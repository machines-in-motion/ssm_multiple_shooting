import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)
from torchnet.meter import AverageValueMeter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from ssm import SSM
from inference_tools import ukf
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--T", default=10000, type=int)
arg = parser.parse_args()

T = arg.T


class transition_model(nn.Module):
    def __init__(self, dim_x, ne=32):
        super(transition_model, self).__init__()
        self.dim_x = dim_x
        self.ne = ne
        self.alpha = nn.Sequential(
            nn.Linear(dim_x, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.ne),
            nn.Softmax(dim=1),
        )
        self.matrix = nn.Linear(dim_x * self.ne, dim_x, bias=False)

    def forward(self, x):
        b = self.alpha(x)
        b = b.unsqueeze(1)
        x_ = x.unsqueeze(2) @ b
        cat = x_.reshape(-1, self.ne * self.dim_x)
        return x + self.matrix(cat)


class observation_model(nn.Module):
    def __init__(self, dim_y):
        super(observation_model, self).__init__()
        self.dim_y = dim_y

    def forward(self, x):
        return x[:, : self.dim_y]


dim_x = 3
dim_y = 1

f_net = transition_model(dim_x)
g_net = observation_model(dim_y)

print("---------- Networks architecture -------------")
print(f_net)
print(g_net)

print("----------    Training data     --------------")
Train_set = np.load("data/datasets/lorenz_train.npy")
Train_set = utils.split(Train_set, T)
Test_set = np.load("data/datasets/lorenz_test.npy")
Test_set_GT = np.load("data/datasets/lorenz_test_GT.npy")
N_train, T, d = Train_set.shape
print(f"{N_train} training trajctories with {T} time steps")
N_test, T_test, d = Test_set.shape
print(f"{N_test} testing trajctories with {T_test} time steps")
print(f"The dimension of the measurement space is {dim_y}")


#  Initialize shooting_nodes with no prior
max_it = 50
ms = int(arg.T / max_it) if arg.T > 100 else int(arg.T / 10)
nodes = np.zeros((N_train, ms, dim_x))
nodes[:, :, :dim_y] = np.transpose(
    np.array([Train_set[:, i * int(T / ms)] for i in range(ms)]), (1, 0, 2)
)

alpha = 100
lr = 1e-3
bs = 40
normalize = True
if normalize:
    mean = np.mean(np.concatenate(Train_set, axis=0), axis=0)
    std = np.std(np.concatenate(Train_set, axis=0), axis=0)
    nodes[:, :, :dim_y] = (nodes[:, :, :dim_y] - mean) / std

model = SSM(
    f_net, g_net, Train_set, nodes, dim_x, ms, lr, bs, alpha, normalize, "lorenz"
)
hist = model.load("lorenz", f"lorenz_{T}_locally_linear")

dtype = torch.float32


class load_data_for_rnn(Dataset):
    def __init__(self, loader, state_traj, T_max=T):
        self.state_traj = state_traj
        self.n = int(T / T_max)
        self.T_max = T_max
        print(self.n)
        self.loader = loader

    def __getitem__(self, index):
        index1, index2 = index // self.n, index % self.n
        ind, data = self.loader.dataset[index1]
        return (
            data[self.T_max * index2 : self.T_max * (index2 + 1)],
            self.state_traj[ind, self.T_max * index2 : self.T_max * (index2 + 1)],
        )

    def __len__(self):
        return len(self.loader.dataset) * self.n


def create_RNN_dataset(model, T, batch_size):
    shooting_nodes = model.nodes.detach()
    N, ms, dx = shooting_nodes.shape
    subtraj = int(T / ms)
    with torch.no_grad():
        state_traj = shooting_nodes.new_empty((N, T, dx))
        for b in range(int(N / batch_size)):
            for i in range(model.ms):
                s = shooting_nodes[b * batch_size : (b + 1) * batch_size, i]
                state_traj[b * batch_size : (b + 1) * batch_size, i * subtraj] = s
                for j in range(subtraj - 1):
                    s = model.f_net(s)
                    state_traj[
                        b * batch_size : (b + 1) * batch_size, i * subtraj + j + 1
                    ] = s
        state_traj = state_traj.detach()
        mean_nodes = torch.mean(state_traj.reshape(-1, dx), axis=0)
        std_nodes = torch.std(state_traj.reshape(-1, dx), axis=0)
        print(mean_nodes)
        print(std_nodes)
        state_traj = (state_traj - mean_nodes) / std_nodes
    trainset = load_data_for_rnn(model.train_loader, state_traj)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True),
        mean_nodes,
        std_nodes,
    )


class RNN(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(RNN, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.lstm = nn.LSTM(1, latent_dim, batch_first=True)
        self.fc2 = nn.Sequential(nn.Linear(self.latent_dim, self.state_dim))

    def forward(self, obs):
        bs, T, _ = obs.shape
        lstm_out, _ = self.lstm(obs)
        lstm_out = lstm_out.reshape(bs * T, -1)
        state = self.fc2(lstm_out)
        return state.view(bs, T, -1)


rnn = RNN(dim_x, 1024)
print(rnn)
epochs = 1000
batch_size = 40
len_filt = 1990


data_test = torch.tensor((Test_set - mean) / std, dtype=dtype)
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    rnn.cuda()
    data_test = data_test.cuda()

if normalize:
    mean, std = data_test.new_tensor(mean), data_test.new_tensor(std)
Test_set_GT = data_test.new_tensor(Test_set_GT)

train_loader, mean_nodes, std_nodes = create_RNN_dataset(model, T, batch_size)
optimizer = optim.Adam(rnn.parameters())
for epoch in range(1, epochs + 1):
    meter_loss = AverageValueMeter()
    for it, (obs, state) in enumerate(train_loader):
        rnn.zero_grad()
        if use_cuda:
            obs, state = obs.cuda(), state.cuda()
        state_hat = rnn(obs)
        loss = F.mse_loss(state_hat, state)
        meter_loss.add(loss.item())
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, meter_loss.mean))

    if epoch % 50 == 0:
        with torch.no_grad():
            mse_list = []
            state_hat = data_test.new_empty((data_test.shape[0], state.shape[2]))
            pred = data_test.new_empty((data_test.shape[0], T_test - len_filt))
            for b in range(int(data_test.shape[0] / batch_size)):
                f = data_test[b * batch_size : (b + 1) * batch_size, :len_filt]
                s = rnn(f)
                s = s[:, -1] * std_nodes + mean_nodes
                state_hat[b * batch_size : (b + 1) * batch_size] = s
            for j in range(data_test.shape[0]):
                obs_GT = data_test[j]
                s = state_hat[j : j + 1]
                for i in range(T_test - len_filt):
                    s = model.f_net(s)
                    pred[j, i] = model.g_net(s)
            pred = pred * std + mean
            gt = Test_set_GT[:, len_filt:].squeeze()
            score = F.mse_loss(gt, pred)
        print(f"MSE over {T_test - len_filt} steps = ", score)

len_filt = 1900
with torch.no_grad():
    index = 0
    f = data_test[index : index + 1, :len_filt]
    if use_cuda:
        f = f.cuda()
    s = rnn(f)
    s = s * std_nodes + mean_nodes
    s_traj = s.new_empty((T_test, model.dim_x))
    s_traj[:len_filt] = s
    s = s[:, -1]
    for i in range(T_test - len_filt):
        s = model.f_net(s)
        s_traj[len_filt + i] = s

    s_traj[:, 0] = s_traj[:, 0] * std + mean
    s_traj = s_traj.cpu().detach().numpy()

# Qualitative plot
dt = 0.005
time = np.linspace(0, dt * T_test, T_test)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(time, Test_set[index, :, 0], "-", linewidth=1, label="Measurements")
ax1.plot(
    time,
    Test_set_GT[index, :, 0].cpu().detach().numpy(),
    "--",
    linewidth=2,
    label="Measurements",
)
ax1.plot(time, s_traj[:, 0], "k")
ax1.axvline(time[len_filt], 0, 1, color="k")
ax1.set_ylabel("x1")
ax2.plot(time, s_traj[:, 1], "k")
ax2.axvline(time[len_filt], 0, 1, color="k")
ax2.set_ylabel("x2")
ax3.plot(time, s_traj[:, 2], "k")
ax3.axvline(time[len_filt], 0, 1, color="k")
ax3.set_ylabel("x3")
ax1.set_xlim([0, dt * T_test])
ax2.set_xlim([0, dt * T_test])
ax3.set_xlim([0, dt * T_test])
plt.savefig("plotting/plots/lorenz_test_rnn.png")
plt.show()

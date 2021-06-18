import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchnet.meter import AverageValueMeter

torch.manual_seed(0)

import time
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, animation

import numpy as np
import sys

dtype = torch.float32

main_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def split(data, rec):
    N_traj, T, d = data.shape
    if rec > T:
        raise ValueError("Dataset has too short trajectories")
    m = int(T / rec)
    split_set = np.zeros((N_traj * m, rec, d))
    for it in range(m):
        split_set[it * N_traj : (it + 1) * N_traj] = data[:, it * rec : (it + 1) * rec]
    return split_set


data_name = "Lorenz"
data_path = os.path.join(main_folder, "ssm_ms/data/datasets")
train_name = os.path.join(data_path, f"lorenz_train.npy")
test_name = os.path.join(data_path, f"lorenz_test.npy")
test_name_GT = os.path.join(data_path, f"lorenz_test_GT.npy")
train_data = np.load(train_name)
train_data = split(train_data, 10000)
test_data = np.load(test_name)
test_data_GT = np.load(test_name_GT)


mean = np.mean(np.concatenate(train_data))
std = np.std(np.concatenate(train_data))
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
test_data_GT = (test_data_GT - mean) / std

train_obs = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
rs = np.random.RandomState()
train_obs_valid = rs.rand(train_data.shape[0], train_data.shape[1], 1) < 0.5
train_obs_valid[:, :1] = True
train_targets = train_obs.copy()
print(
    "Fraction of Valid Images:",
    np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape),
)
train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0
train_input = np.concatenate((train_obs, train_obs_valid), axis=2)


len_filt = 1990
test_obs = test_data.copy()
test_obs = test_obs.reshape((test_data.shape[0], test_data.shape[1], 1))
test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1))
test_obs_valid[:, len_filt:] = 0
test_targets = test_data_GT.copy()
test_targets = test_targets.reshape((test_data.shape[0], test_data.shape[1], 1))
test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0
test_input = np.concatenate((test_obs, test_obs_valid), axis=2)

print(train_targets.shape)
print(test_targets.shape)


class load_data_for_rnn(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


epochs = 1000
bs = 40
train_input = torch.tensor(train_input, dtype=dtype)
train_targets = torch.tensor(train_targets, dtype=dtype)
trainset = load_data_for_rnn(train_input, train_targets)
train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)


test_input = torch.tensor(test_input, dtype=dtype)
test_targets = torch.tensor(test_targets, dtype=dtype)
testset = load_data_for_rnn(test_input, test_targets)
test_loader = DataLoader(testset, batch_size=bs, shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)
        self.fc2 = nn.Sequential(nn.Linear(latent_dim, output_dim))

    def forward(self, obs):
        bs, T, _ = obs.shape
        lstm_out, _ = self.lstm(obs)
        lstm_out = lstm_out.reshape(bs * T, -1)
        state = self.fc2(lstm_out)
        return state.view(bs, T, -1)


rnn = RNN(2, 1024, 1)

mean = torch.tensor(mean, dtype=dtype)
std = torch.tensor(std, dtype=dtype)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    rnn.cuda()
    mean = mean.cuda()
    std = std.cuda()

optimizer = optim.Adam(rnn.parameters())
for epoch in range(1, epochs + 1):
    meter_loss = AverageValueMeter()
    for it, (x, y) in enumerate(train_loader):
        rnn.zero_grad()
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        y_hat = rnn(x)
        loss = F.mse_loss(y, y_hat)
        meter_loss.add(loss.item())
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, meter_loss.mean))

    with torch.no_grad():
        meter_loss = AverageValueMeter()
        for it, (x, y) in enumerate(test_loader):
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            y_hat = rnn(x)
            y_hat = y_hat * std + mean
            y = y * std + mean
            loss = F.mse_loss(y[:, len_filt:], y_hat[:, len_filt:])
            meter_loss.add(loss.item())
        print("Test Epoch: {} \tLoss: {:.6f}".format(epoch, meter_loss.mean))


index = 1
x, y = next(iter(test_loader))
if use_cuda:
    x = x.cuda()
y_hat = rnn(x)
pred = y_hat[index].detach().cpu().numpy()

# Qualitative plot
dt = 0.005
T_test = test_data.shape[1]
time = np.linspace(0, dt * T_test, T_test)
fig = plt.figure()
plt.plot(time, test_obs[index, :, 0], "-", linewidth=1, label="Measurements")
plt.plot(
    time, test_data_GT[index, :, 0], "--", linewidth=2, label="GT",
)
plt.plot(time, pred, label="prediction")
plt.axvline(time[len_filt], 0, 1, color="k")
plt.xlabel("x1")
plt.xlim([9.6, dt * T_test])
plt.legend()
plt.savefig("plots/lorenz_test_rnn.png")
plt.show()

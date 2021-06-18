import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
import numpy as np
import matplotlib.pyplot as plt
from ssm import SSM
from inference_tools import ukf
import argparse
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--T", default=10000, type=int)
parser.add_argument("--ll", default=1, type=int)
arg = parser.parse_args()

if arg.ll:
    log_name = "locally_linear"
    max_it = 50

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


else:
    log_name = "fully_connected"
    max_it = 10

    class transition_model(nn.Module):
        def __init__(self, dim_x):
            super(transition_model, self).__init__()

            self.fc = nn.Sequential(
                nn.Linear(dim_x, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, dim_x),
            )

        def forward(self, x):
            return x + self.fc(x)


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
Train_set = utils.split(Train_set, arg.T)
Test_set = np.load("data/datasets/lorenz_test.npy")
Test_set_GT = np.load("data/datasets/lorenz_test_GT.npy")
N_train, T, d = Train_set.shape
print(f"{N_train} training trajctories with {T} time steps")
N_test, T_test, d = Test_set.shape
print(f"{N_test} testing trajctories with {T_test} time steps")
print(f"The dimension of the measurement space is {dim_y}")


#  Initialize shooting_nodes with no prior
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
hist = model.load("lorenz", f"lorenz_{arg.T}_{log_name}")


# Evaluation on Test set
dt = 0.005
time = np.linspace(0, dt * T_test, T_test)
len_filt = 1990  # 1600 for qualitative plot
Variance = 0.5
state_traj, score0, score = ukf.test_UKF(
    model, Test_set, Test_set_GT, len_filt, Variance
)
print(f"Error at the end of the filter : {score0}")
print(f"AVG error on the next {T_test - len_filt} steps: {score}")


# Qualitative plot UKF
index = 3
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(time, Test_set[index, :, 0], "-", linewidth=1, label="Measurements")
ax1.plot(time, Test_set_GT[index, :, 0], linewidth=3, label="Measurements")
ax1.plot(time, state_traj[index, :, 0], "k", linewidth=1)
ax1.axvline(time[len_filt], 0, 1, color="k")
ax1.set_ylabel("x1")
ax2.plot(time, state_traj[index, :, 1], "k")
ax2.axvline(time[len_filt], 0, 1, color="k")
ax2.set_ylabel("x2")
ax3.plot(time, state_traj[index, :, 2], "k")
ax3.axvline(time[len_filt], 0, 1, color="k")
ax3.set_ylabel("x3")
ax3.set_xlabel("Time [s]")
ax1.set_xlim([0, dt * T_test])
ax2.set_xlim([0, dt * T_test])
ax3.set_xlim([0, dt * T_test])
# ax1.legend(loc="upper left")
plt.savefig("plotting/plots/lorenz_test.png")
plt.show()


#  Training trajectory with nodes
index = 10
traj = np.empty((T, dim_x))
for i in range(ms):
    x = model.nodes[index, i].unsqueeze(0)
    traj[i * int(T / ms)] = x.detach().cpu().numpy()
    for t in range(int(T / ms) - 1):
        x = model.f_net(x)
        traj[i * int(T / ms) + t + 1] = x.detach().cpu().numpy()
if normalize:
    mean, std = model.dict_normalize["mean"], model.dict_normalize["std"]
    traj[:, :dim_y] = traj[:, :dim_y] * std + mean

time = np.linspace(0, dt * T, T)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(time, traj[:, 0], label="x1")
ax1.plot(time, Train_set[index, :, 0], "x", label="x1 GT")
ax1.legend(loc="upper right")
ax2.plot(time, traj[:, 1], label="x2")
#  ax2.plot(time, Train_set[index, :, 1], label='x2 GT')
ax2.legend(loc="upper right")
ax3.plot(time, traj[:, 2], label="x3")
#  ax3.plot(time, Train_set[index, :, 2], label='x3 GT')
ax3.legend(loc="upper right")
ax1.set_xlim([0, 1])
ax2.set_xlim([0, 1])
ax3.set_xlim([0, 1])
plt.savefig("plotting/plots/lorenz_nodes.png")

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Lorenz Attractor")
plt.savefig("plotting/plots/lorenz_nodes_3d.png")


#  Training trajectory without nodes
traj = np.empty((T, dim_x))
x = model.nodes[index, 0].unsqueeze(0)
traj[0] = x.detach().cpu().numpy()
for t in range(T - 1):
    x = model.f_net(x)
    traj[t + 1] = x.detach().cpu().numpy()
if normalize:
    mean, std = model.dict_normalize["mean"], model.dict_normalize["std"]
    traj[:, :dim_y] = traj[:, :dim_y] * std + mean
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
time = np.linspace(0, dt * T, T)
ax1.plot(time, traj[:, 0], label="x1")
ax1.plot(time, Train_set[index, :, 0], "x", label="x1 GT")
ax1.legend(loc="upper right")
ax2.plot(time, traj[:, 1], label="x2")
#  ax2.plot(time, Train_set[index, :, 1], label='x2 GT')
ax2.legend(loc="upper right")
ax3.plot(time, traj[:, 2], label="x3")
#  ax3.plot(time, Train_set[index, :, 2], label='x3 GT')
ax3.legend(loc="upper right")
ax1.set_xlim([0, 1])
ax2.set_xlim([0, 1])
ax3.set_xlim([0, 1])
plt.savefig("plotting/plots/lorenz.png")

fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.5)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Lorenz Attractor")
plt.savefig("plotting/plots/lorenz_3d.png")

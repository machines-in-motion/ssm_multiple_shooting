import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ssm import SSM
from inference_tools import ukf
import matplotlib.pyplot as plt


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
data = np.load("data/datasets/lorenz_train.npy")
data = np.concatenate([data[:, i * 10 : i * 10 + 1] for i in range(1000)], axis=1)

N_train, T, d = data.shape
print(f"{N_train} training trajctories with {T} time steps")
print(f"The dimension of the measurement space is {dim_y}")


#  Initialize shooting_nodes with no prior
ms = 100
nodes = np.zeros((N_train, ms, dim_x))
nodes[:, :, :dim_y] = np.transpose(
    np.array([data[:, i * int(T / ms)] for i in range(ms)]), (1, 0, 2)
)

alpha = 100
lr = 1e-3
bs = 40
normalize = True
if normalize:
    mean = np.mean(np.concatenate(data, axis=0), axis=0)
    std = np.std(np.concatenate(data, axis=0), axis=0)
    nodes[:, :, :dim_y] = (nodes[:, :, :dim_y] - mean) / std

model = SSM(f_net, g_net, data, nodes, dim_x, ms, lr, bs, alpha, normalize, "lorenz")


fig, ax = plt.subplots(1, 2, figsize=(15, 3))
N_epoch = [0, 1000]
count_epoch = ["Initial Guess", "Final Solution"]
for p in range(2):
    model.train(N_epoch[p], loss_name="mse")
    index = 3
    noisy = data[index]
    n = int(T / ms)
    traj = np.empty((ms, n + 1, dim_x))
    for i in range(ms):
        x = model.nodes[index, i].unsqueeze(0)
        traj[i, 0] = x.detach().cpu().numpy()
        for t in range(n):
            x = model.f_net(x)
            traj[i, t + 1] = x.detach().cpu().numpy()
    if normalize:
        traj[:, :, :dim_y] = traj[:, :, :dim_y] * std + mean

    dt = 0.05
    time = np.linspace(0, dt * T, T)
    ax[p].plot(time, noisy, "x")
    #  plt.plot(time, GT)
    for i in range(ms):
        if i == ms - 1:
            ax[p].plot(time[i * n : (i + 1) * n], traj[i, :-1, 0])
        else:
            ax[p].plot(time[i * n : (i + 1) * n + 1], traj[i, :, 0])
    ax[p].title.set_text(count_epoch[p])
    ax[p].set_xlim([0, 10])
    ax[p].set_xlabel("Time")

fig.tight_layout()
plt.savefig("plotting/plots/schematic.png")

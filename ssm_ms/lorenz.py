import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
import numpy as np
from ssm import SSM
import utils
from inference_tools import ukf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--T", default=100, type=int)
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
N_train, T, d = Train_set.shape
print(f"{N_train} training trajctories with {T} time steps")
print(f"The dimension of the measurement space is {dim_y}")


#  Initialize shooting_nodes
ms = int(arg.T / max_it) if arg.T > 100 else int(arg.T / 10)
nodes = np.zeros((N_train, ms, dim_x))
nodes[:, :, :dim_y] = np.transpose(
    np.array([Train_set[:, i * int(T / ms)] for i in range(ms)]), (1, 0, 2)
)

alpha = 100
print(f"Penalty coeffiscient = {alpha}")
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

N_epoch = 1000
model.train(N_epoch)
model.save("lorenz", f"lorenz_{arg.T}_{log_name}")

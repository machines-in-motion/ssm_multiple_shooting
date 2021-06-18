import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
import numpy as np
import matplotlib.pyplot as plt
from ssm import SSM
from inference_tools import rnn_tools
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ll", default=1, type=int)
arg = parser.parse_args()

if arg.ll:
    log_name = "locally_linear"

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

    class transition_model(nn.Module):
        def __init__(self, dim_x):
            super(transition_model, self).__init__()

            self.fc = nn.Sequential(
                nn.Linear(dim_x, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, dim_x),
            )

        def forward(self, x):
            return x + self.fc(x)


class observation_model(nn.Module):
    def __init__(self, dim_x, dim_y, ne=32):
        super(observation_model, self).__init__()
        self.ne = ne
        self.dim_y = dim_y
        self.dim_input = 3
        self.fc_obs = nn.Linear(dim_x, self.ne * self.dim_input ** 2)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.ne, 16, (5, 5), stride=(3, 3)),
            nn.LayerNorm((16, 11, 11)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, (3, 3), stride=(2, 2)),
            nn.LayerNorm((12, 23, 23)),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, (2, 2), stride=(1, 1)),
        )

    def forward(self, x):
        x = F.relu(self.fc_obs(x))
        x = x.view(-1, self.ne, self.dim_input, self.dim_input)
        x = self.conv_layers(x)
        x = torch.sigmoid(x.view(-1, self.dim_y))
        return x


dim_x = 10
im_size = 24
dim_y = im_size ** 2

f_net = transition_model(dim_x)
g_net = observation_model(dim_x, dim_y)

print("---------- Networks architecture -------------")
print(f_net)
print(g_net)

print("----------    Training data     --------------")
Train_set, Test_set = utils.load_pendulum_data(1)
N_train, T, d = Train_set.shape
print(f"{N_train} training trajctories with {T} time steps")
N_test, T_test, d = Test_set.shape
print(f"{N_test} training trajctories with {T_test} time steps")
print(f"The dimension of the measurement space is {dim_y}")


#  Initialize shooting_nodes with no prior
ms = 4
nodes = np.zeros((N_train, ms, dim_x))


alpha = 1e4
print(f"Penalty coeffiscient = {alpha}")
lr = 1e-3
bs = 40
normalize = False
model = SSM(
    f_net, g_net, Train_set, nodes, dim_x, ms, lr, bs, alpha, normalize, "pendulum"
)

N_epoch = 1000
hist = model.load("Pendulum", "locally_linear")
loss = hist["loss"]
penalty = hist["penalty"]

plt.plot(loss, label="loss")
plt.plot(penalty, label="penalty")
plt.yscale("log")
plt.xlabel("Learning iterations")
plt.grid()
plt.legend(loc="upper right")
plt.savefig(f"plotting/plots/loss_ex.png")

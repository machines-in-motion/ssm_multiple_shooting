import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(0)
import numpy as np
from ssm import SSM
from inference_tools import rnn_tools
import utils
from data.moving_mnist import MovingMNIST


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
    def __init__(self, dim_x, dim_y, ne=512):
        super(observation_model, self).__init__()
        self.ne = ne
        self.dim_y = dim_y
        self.dim_input = 3
        self.fc_obs = nn.Linear(dim_x, self.ne * self.dim_input ** 2)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.ne, 512, (3, 3), stride=(2, 2)),
            nn.LayerNorm((512, 7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (3, 3), stride=(2, 2)),
            nn.LayerNorm((256, 15, 15)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (3, 3), stride=(2, 2)),
            nn.LayerNorm((256, 31, 31)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 1, (4, 4), stride=(2, 2)),
        )

    def forward(self, x):
        x = F.relu(self.fc_obs(x))
        x = x.view(-1, self.ne, self.dim_input, self.dim_input)
        x = self.conv_layers(x)
        x = torch.sigmoid(x.view(-1, self.dim_y))
        return x


dim_x = 100
im_size = 64
dim_y = im_size ** 2

f_net = transition_model(dim_x)
g_net = observation_model(dim_x, dim_y)

print("---------- Networks architecture -------------")
print(f_net)
print(g_net)

print("----------    Training data     --------------")
n_frames_input = 10
n_frames_output = 10

train_name = "data/datasets/mnist_train_seq.npy"
train = np.load(train_name, allow_pickle=True)
Train_set = np.array([x.toarray() for x in train])
Train_set /= 255
N_train, T, d = Train_set.shape
print(f"{N_train} training trajctories with {T} time steps")
print(f"The dimension of the measurement space is {dim_y}")


#  Initialize shooting_nodes with no prior
ms = 2
nodes = np.zeros((N_train, ms, dim_x))

alpha = 1e4
print(f"Penalty coeffiscient = {alpha}")
lr = 1e-3
bs = 50
normalize = False
model = SSM(
    f_net, g_net, Train_set, nodes, dim_x, ms, lr, bs, alpha, normalize, "moving_mnist"
)

N_epoch = 250
model.train(N_epoch)
model.save("Moving_mnist", "s100_ms2_a4")


index = 0
s_traj, obs_pred = model.train_prediction(index)
obs_pred = obs_pred.reshape(-1, im_size, im_size).detach().cpu().numpy()
obs_GT = Train_set[index].reshape((-1, im_size, im_size))
rnn_tools.create_gif(obs_pred, obs_GT, 20, "moving_mnist_train", True)

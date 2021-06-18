import torch.nn as nn
import torch.nn.functional as F
import torch
from torchnet.meter import AverageValueMeter

torch.manual_seed(0)
import time
import numpy as np
from ssm import SSM
from inference_tools import rnn_tools
import utils
from data.moving_mnist import MovingMNIST
from tqdm import tqdm


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
root = "data/datasets"
is_train = True
n_frames_input = 10
n_frames_output = 10

train_name = "data/datasets/mnist_train_seq.npy"
train = np.load(train_name, allow_pickle=True)
Train_set = np.array([x.toarray() for x in train])
Train_set /= 255
N_train, T, d = Train_set.shape
print(f"{N_train} training trajctories with {T} time steps")
Test_set = np.load("data/datasets/mnist_test_seq.npy")
Test_set = Test_set.reshape(20, 10000, 64 * 64) / 255
Test_set = Test_set.transpose((1, 0, 2))
N_test, T_test, d = Test_set.shape
print(f"{N_test} testing trajctories with {T_test} time steps")
print(f"The dimension of the measurement space is {dim_y}")


ms = 2
nodes = np.zeros((N_train, ms, dim_x))

alpha = 1e4
lr = 1e-3
bs = 50
normalize = False
model = SSM(
    f_net, g_net, Train_set, nodes, dim_x, ms, lr, bs, alpha, normalize, "moving_mnist"
)

hist = model.load("Moving_mnist", "s100_ms2_a4")
print("s100_ms2_a4")


index = 0
s_traj, obs_pred = model.train_prediction(index)
obs_pred = obs_pred.reshape(-1, im_size, im_size).detach().cpu().numpy()
obs_GT = Train_set[index].reshape((-1, im_size, im_size))
rnn_tools.create_gif(obs_pred, obs_GT, 20, "moving_mnist_train", True)


class RNN(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(RNN, self).__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LayerNorm((64, 62, 62)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, 3),
            nn.LayerNorm((128, 29, 29)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(128, 512, 3),
            nn.LayerNorm((512, 12, 12)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(512, 16, 3),
            nn.ReLU(),
        )
        im_output_size = int((((im_size - 2) / 2 - 2) / 2 - 2) / 2 - 2)
        self.output_dim_cnn = 16 * im_output_size ** 2

        self.lstm = nn.LSTM(self.output_dim_cnn, latent_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(self.latent_dim, self.state_dim))

    def forward(self, obs):
        bs, T, C, H, W = obs.shape
        conv_out = self.conv(obs.reshape(bs * T, C, H, W))
        conv_out = conv_out.view(bs, T, -1)
        lstm_out, _ = self.lstm(conv_out)
        lstm_out = lstm_out.reshape(bs * T, -1)
        state = self.fc(lstm_out)
        return state.view(bs, T, -1)


start_time = time.time()
rnn = RNN(dim_x, 1024)
print(rnn)
rnn_epochs = 25
len_filt = n_frames_input
mean_state, std_state = rnn_tools.train_rnn(
    rnn, rnn_epochs, model, Test_set, len_filt, T, im_size, bs
)
print(f"Training lasted {int((time.time() - start_time)/60)} min")

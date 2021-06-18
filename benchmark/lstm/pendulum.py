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


data_name = "Pendulum"
data_path = os.path.join(main_folder, "ssm_ms/data/datasets")
train_name = os.path.join(data_path, f"1_link_image_train.npy")
test_name = os.path.join(data_path, f"1_link_image_test.npy")
train = np.load(train_name, allow_pickle=True)
test = np.load(test_name, allow_pickle=True)
train_data = np.array([x.toarray() for x in train])
test_data = np.array([x.toarray() for x in test])

im_size = 24

len_filt = 50
train_obs = train_data.reshape((train_data.shape[0], train_data.shape[1], 1, 24, 24))
train_obs_valid = np.ones((train_obs.shape[0], train_obs.shape[1], 1))
train_obs_valid[:, len_filt:] = 0
train_targets = train_obs.copy()
train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0

test_obs = test_data.reshape((test_data.shape[0], test_data.shape[1], 1, 24, 24))
test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1))
test_obs_valid[:, len_filt:] = 0
test_targets = test_obs.copy()
test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0


print(train_targets.shape)
print(test_targets.shape)


class load_data_for_rnn(Dataset):
    def __init__(self, X, mask, y):
        self.X = X
        self.mask = mask
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.mask[index], self.y[index]

    def __len__(self):
        return len(self.X)


epochs = 1000
bs = 40
train_input = torch.tensor(train_obs, dtype=dtype)
train_targets = torch.tensor(train_targets, dtype=dtype)
train_mask = torch.tensor(train_obs_valid, dtype=dtype)
trainset = load_data_for_rnn(train_input, train_mask, train_targets)
train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)


test_input = torch.tensor(test_obs, dtype=dtype)
test_targets = torch.tensor(test_targets, dtype=dtype)
test_mask = torch.tensor(test_obs_valid, dtype=dtype)
testset = load_data_for_rnn(test_input, test_mask, test_targets)
test_loader = DataLoader(testset, batch_size=bs, shuffle=False)


class RNN(nn.Module):
    def __init__(self, latent_dim):
        super(RNN, self).__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.LayerNorm((16, 22, 22)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 16, 3),
            nn.LayerNorm((16, 9, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 16, 3),
        )
        im_output_size = int(((im_size - 2) / 2 - 2) / 2 - 2)
        self.output_dim_cnn = 16 * im_output_size ** 2

        self.lstm = nn.LSTM(self.output_dim_cnn + 1, latent_dim, batch_first=True)

        self.dim_input = 3
        self.ne = 32
        self.fc_obs = nn.Linear(latent_dim, self.ne * self.dim_input ** 2)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(self.ne, 16, (5, 5), stride=(3, 3)),
            nn.LayerNorm((16, 11, 11)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, (3, 3), stride=(2, 2)),
            nn.LayerNorm((12, 23, 23)),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 1, (2, 2), stride=(1, 1)),
        )

    def forward(self, obs, mask):
        bs, T, C, H, W = obs.shape
        conv_out = self.conv(obs.reshape(bs * T, C, H, W))
        conv_out = conv_out.view(bs, T, -1)
        lstm_input = torch.cat((conv_out, mask), axis=2)
        lstm_out, _ = self.lstm(lstm_input)
        x = lstm_out.reshape(bs * T, -1)
        x = F.relu(self.fc_obs(x))
        x = x.view(-1, self.ne, self.dim_input, self.dim_input)
        x = self.deconv_layers(x)
        x = torch.sigmoid(x)
        return x.view(bs, T, C, H, W)


rnn = RNN(1024)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    rnn.cuda()

optimizer = optim.Adam(rnn.parameters())
for epoch in range(1, epochs + 1):
    meter_loss = AverageValueMeter()
    for it, (x, mask, y) in enumerate(train_loader):
        rnn.zero_grad()
        if use_cuda:
            x, mask, y = x.cuda(), mask.cuda(), y.cuda()
        y_hat = rnn(x, mask)
        loss = F.mse_loss(y, y_hat)
        meter_loss.add(loss.item())
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, meter_loss.mean))
    if epoch % 10 == 1:
        with torch.no_grad():
            meter_loss = AverageValueMeter()
            for it, (x, mask, y) in enumerate(test_loader):
                index_mask = np.logical_not(np.squeeze(mask.detach().numpy()))
                if use_cuda:
                    x, mask, y = x.cuda(), mask.cuda(), y.cuda()
                y_hat = rnn(x, mask)
                loss = F.mse_loss(y[index_mask], y_hat[index_mask])
                meter_loss.add(loss.item())
            print("Test Epoch: {} \tLoss: {:.6f}".format(epoch, meter_loss.mean))

with torch.no_grad():
    meter_loss_mse = []
    meter_loss_bce = []
    for i in range(test_input.shape[0]):
        x = test_input[i : i + 1]
        y = test_targets[i : i + 1]
        mask = test_mask[i : i + 1]
        index_mask = np.logical_not(np.squeeze(mask.detach().numpy()))
        if use_cuda:
            x, mask, y = x.cuda(), mask.cuda(), y.cuda()
        y_hat = rnn(x, mask)
        tar = y[0, index_mask].flatten()
        obs_pred = y_hat[0, index_mask].flatten()
        loss_mse = F.mse_loss(obs_pred, tar).cpu().detach().numpy()
        meter_loss_mse.append(loss_mse)
        loss_bce = (
            -torch.mean(
                tar * torch.log(obs_pred + 1e-12)
                + (1 - tar) * torch.log(1 - obs_pred + 1e-12)
            )
            .cpu()
            .detach()
            .numpy()
        )
        meter_loss_bce.append(loss_bce)
    print(
        "\tLoss MSE: {:.6f} +- {:.6f}".format(
            np.mean(meter_loss_mse), np.std(meter_loss_mse)
        )
    )
    print(
        "\tLoss BCE: {:.6f} +- {:.6f}".format(
            np.mean(meter_loss_bce), np.std(meter_loss_bce)
        )
    )


obs_GT = y.squeeze().cpu().detach().numpy()
obs_pred = y_hat.squeeze().cpu().detach().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2)
jump = 1
N = len(obs_GT)


def animate(i):
    ax1.imshow(obs_pred[i * jump], cmap="gray")
    ax1.set_title("Prediction")
    ax2.imshow(obs_GT[i * jump], cmap="gray")
    ax2.set_title("Ground truth")
    if i * jump < len_filt:
        fig.suptitle("Filtering")
    else:
        fig.suptitle("Prediction")


ani = animation.FuncAnimation(fig, animate, frames=int(N / jump), interval=100)
ani.save(f"plots/{data_name}.gif", writer="imagemagick")

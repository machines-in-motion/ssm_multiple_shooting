import numpy as np
import os
from matplotlib import cm, animation
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from torch.utils.data import DataLoader, Dataset
import torch
from torchnet.meter import AverageValueMeter
import torch.nn.functional as F
import torch.optim as optim

dtype = torch.float32


class load_data_for_rnn(Dataset):
    def __init__(self, loader, state_traj, im_size):
        self.state_traj = state_traj
        self.loader = loader
        self.im_size = im_size

    def __getitem__(self, index):
        ind, image_traj = self.loader.dataset[index]
        image_traj = image_traj.reshape(-1, 1, self.im_size, self.im_size)
        return image_traj, self.state_traj[ind]

    def __len__(self):
        return len(self.loader.dataset)


def create_RNN_dataset(model, T, im_size, batch_size):
    shooting_nodes = model.nodes.detach()
    N, ms, dx = shooting_nodes.shape
    subtraj = int(T / ms)
    with torch.no_grad():
        state_traj = shooting_nodes.new_empty((N, T, dx))
        for b in range(int(shooting_nodes.shape[0] / batch_size)):
            for i in range(ms):
                s = shooting_nodes[b * batch_size : (b + 1) * batch_size, i]
                state_traj[b * batch_size : (b + 1) * batch_size, i * subtraj] = s
                for j in range(subtraj - 1):
                    s = model.f_net(s)
                    state_traj[
                        b * batch_size : (b + 1) * batch_size, i * subtraj + j + 1
                    ] = s
        state_traj = state_traj.detach()
        mean = torch.mean(state_traj.reshape(-1, dx), axis=0)
        std = torch.std(state_traj.reshape(-1, dx), axis=0)
        print(mean)
        print(std)
        state_traj = (state_traj - mean) / std
    trainset = load_data_for_rnn(model.train_loader, state_traj, im_size)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True), mean, std


def train_rnn(rnn, epochs, model, Test_set, len_filt, T, im_size, batch_size):
    data_test = torch.tensor(
        Test_set.reshape((-1, T, 1, im_size, im_size)), dtype=dtype
    )
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        rnn.cuda()
        data_test = data_test.cuda()

    train_loader, mean, std = create_RNN_dataset(model, T, im_size, batch_size)
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

        if epoch % 50 == 0 or epoch == epochs:
            with torch.no_grad():
                # Evaluate on test
                bce_list = []
                mse_list = []
                state_hat = data_test.new_empty((data_test.shape[0], state.shape[2]))
                for b in range(int(data_test.shape[0] / batch_size)):
                    f = data_test[b * batch_size : (b + 1) * batch_size, :len_filt]
                    s = rnn(f)
                    s = s[:, -1] * std + mean
                    state_hat[b * batch_size : (b + 1) * batch_size] = s
                for j in range(data_test.shape[0]):
                    score_mse = 0
                    score_bce = 0
                    obs_GT = data_test[j]
                    s = state_hat[j : j + 1]
                    for i in range(T - len_filt):
                        s = model.f_net(s)
                        obs_pred = model.g_net(s).squeeze()
                        tar = obs_GT[len_filt + i].flatten()
                        score_mse += F.mse_loss(
                            obs_pred, obs_GT[len_filt + i].flatten()
                        )
                        score_bce += -torch.mean(
                            tar * torch.log(obs_pred + 1e-12)
                            + (1 - tar) * torch.log(1 - obs_pred + 1e-12)
                        )

                    bce_list.append(score_bce.cpu().detach().numpy() / (T - len_filt))
                    mse_list.append(score_mse.cpu().detach().numpy() / (T - len_filt))

                print("Test MSE = ", np.mean(mse_list), "  +-  ", np.std(mse_list))
                print("Test BCE = ", np.mean(bce_list), "  +-  ", np.std(bce_list))
                print(
                    "Test MSE not normalized = ",
                    np.mean(mse_list) * im_size ** 2,
                    "  +-  ",
                    np.std(mse_list) * im_size ** 2,
                )
                print(
                    "Test BCE not normalized = ",
                    np.mean(bce_list) * im_size ** 2,
                    "  +-  ",
                    np.std(bce_list) * im_size ** 2,
                )

    with torch.no_grad():
        index = 1
        obs_GT = data_test[index].squeeze().cpu().numpy()
        f = data_test[index : index + 1, :len_filt]
        if use_cuda:
            f = f.cuda()
        s = rnn(f)
        s = s * std + mean
        s_traj = s.new_empty((T, model.dim_x))
        s_traj[:len_filt] = s
        s = s[:, -1]
        for i in range(T - len_filt):
            s = model.f_net(s)
            s_traj[len_filt + i] = s

        obs_pred = model.g_net(s_traj).cpu().detach().numpy()
        obs_pred = obs_pred.reshape((T, im_size, im_size))
        create_gif(obs_pred, obs_GT, len_filt, model.data_name)
        mosaic(obs_pred, obs_GT, len_filt, model.data_name)
    return mean, std


def mosaic(obs_pred, obs_GT, len_filt, data_name):
    if data_name == "moving_mnist":
        l = 10
        jump = 1
        off_set = 10
    else:
        l = 10
        jump = 10
        off_set = 0
    f, ax = plt.subplots(2, l, figsize=(8, 2))
    for i in range(0, l):
        ax[1, i].imshow(obs_pred[off_set + (i + 1) * jump - 1], cmap="gray")
        ax[0, i].imshow(obs_GT[off_set + (i + 1) * jump - 1], cmap="gray")
        ax[1, i].axis("off")
        ax[0, i].axis("off")
        t = 0.1 * (i + 1) * jump
        ax[0, i].set_title(f"{t:.1f}")
    plt.savefig(f"plotting/plots/{data_name}_mosaic1.png")

    if data_name != "moving_mnist":
        l = 100
        f, ax = plt.subplots(10, 10, figsize=(20, 20))
        for i in range(0, l):
            im = obs_pred[i]
            r = i % 10
            c = i // 10
            ax[c, r].imshow(im, cmap="gray")
            ax[c, r].axis("off")
            t = 0.1 * (i + 1)
            ax[c, r].set_title(f"{t:.1f}", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"plotting/plots/{data_name}_mosaic2_pred.png")

        l = 100
        f, ax = plt.subplots(10, 10, figsize=(20, 20))
        for i in range(0, l):
            im = obs_GT[i]
            r = i % 10
            c = i // 10
            ax[c, r].imshow(im, cmap="gray")
            ax[c, r].axis("off")
            t = 0.1 * (i + 1)
            ax[c, r].set_title(f"{t:.1f}", fontsize=30)
        plt.tight_layout()
        plt.savefig(f"plotting/plots/{data_name}_mosaic2_GT.png")


def create_gif(obs_pred, obs_GT, len_filt, data_name, train=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    jump = 1

    def animate(i):
        ax1.imshow(obs_pred[i * jump], cmap="gray")
        ax2.imshow(obs_GT[i * jump], cmap="gray")
        ax2.set_title("Ground truth")
        if i * jump < len_filt and not train:
            fig.suptitle("Filtering")
            ax1.set_title("Belief")
        else:
            ax1.set_title("Prediction")
            fig.suptitle("Prediction")

    ani = animation.FuncAnimation(
        fig, animate, frames=int(len(obs_pred) / jump), interval=100
    )
    if not os.path.exists("plotting/plots"):
        os.makedirs("plotting/plots")
    ani.save(f"plotting/plots/{data_name}.gif", writer="imagemagick")

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import utils
import time
from torchnet.meter import AverageValueMeter
from os.path import join
import os
import pickle
from itertools import product
import matplotlib.pyplot as plt

dtype = torch.float32


class SSM(object):
    def __init__(
        self, f, g, Train, nodes, dim_x, ms, lr, bs, alpha, normalize, data_name
    ):
        self.dim_x = dim_x
        self.ms = ms
        self.batch_size = bs
        self.alpha = alpha
        self.lr = lr
        self.T = Train.shape[1]
        self.normalize = normalize
        self.data_name = data_name
        self.train_loader, self.dict_normalize = utils.ms_loading(Train, bs, normalize)
        self.f_net = f
        self.g_net = g

        nodes = torch.tensor(nodes, dtype=dtype)
        self.use_cuda = False
        if torch.cuda.is_available():
            self.use_cuda = True
            print("\n---------- Using GPU ------")
            self.f_net.cuda()
            self.g_net.cuda()
            nodes = nodes.cuda()

        print("\n---------- Multiple Shooting Parameters ------")
        print(f"{self.ms} shooting nodes")
        print(f"Total sequence of length {self.T}")
        assert (
            self.T % self.ms == 0
        ), "The number of subtrajectories must divide the length of the trajectory"
        assert nodes.shape == (
            Train.shape[0],
            self.ms,
            dim_x,
        ), "The shape of the shooting_node should be equal to (N_train, Number of shooting node, dim_x)"
        self.subtraj = int(self.T / self.ms)
        print(f"subtraj are of length {self.subtraj}\n")

        self.nodes = nodes.detach().requires_grad_(True)
        self.optimizer = optim.Adam(
            list(self.f_net.parameters())
            + list(self.g_net.parameters())
            + [self.nodes],
            self.lr,
        )

    def train(self, N_epochs, loss_name="mse"):
        if loss_name == "cross_entropy":
            self.loss = nn.BCELoss(reduction="mean")
        else:
            self.loss = F.mse_loss

        self.train_hist = {}
        self.train_hist["loss"] = []
        self.train_hist["penalty"] = []
        self.train_hist["total_loss"] = []
        self.train_hist["total_time"] = []

        start_time = time.time()
        self.beta = 1
        for epoch in range(N_epochs):
            if epoch == 200:
                self.optimizer = optim.Adam(
                    list(self.f_net.parameters())
                    + list(self.g_net.parameters())
                    + [self.nodes],
                    self.lr / 10,
                )
                self.beta = self.alpha * self.beta
                print("penalty increased and decay lr")
            elif epoch == 600:
                self.optimizer = optim.Adam(
                    list(self.f_net.parameters())
                    + list(self.g_net.parameters())
                    + [self.nodes],
                    self.lr / 100,
                )
                print("Decay lr")

            self.f_net.train()
            self.g_net.train()
            meter_loss = AverageValueMeter()
            meter_penalty = AverageValueMeter()
            meter_total_loss = AverageValueMeter()
            epoch_start_time = time.time()
            for it, (indexes, data) in enumerate(self.train_loader):
                if self.use_cuda:
                    data = data.cuda()
                self.optimizer.zero_grad()
                loss, penalty = self.loss_function(indexes, data)
                total_loss = loss + self.beta * penalty
                meter_loss.add(loss.item())
                meter_penalty.add(penalty.item())
                meter_total_loss.add(total_loss.item())
                total_loss.backward()
                self.optimizer.step()

            self.train_hist["loss"].append(meter_loss.mean)
            self.train_hist["penalty"].append(meter_penalty.mean)
            self.train_hist["total_loss"].append(meter_total_loss.mean)
            print(
                "Train Epoch: {} \tLoss: {:.6f} \tpenalty: {:.8f}".format(
                    epoch, meter_loss.mean, meter_penalty.mean
                )
            )

        self.train_hist["total_time"].append(time.time() - start_time)
        print(f"Training lasted {int((time.time() - start_time)/60)} min")
        self.f_net.eval()
        self.g_net.eval()

    def loss_function(self, indexes, Y):
        # Flatten Y to parralelize computations on each subtraj
        flat_Y = torch.cat(
            [Y[:, it * self.subtraj : (it + 1) * self.subtraj] for it in range(self.ms)]
        )

        # Find the corresponding nodes
        L = list(range(self.ms))
        X = np.array([list(a) for a in product(L, indexes.numpy())])
        x = self.nodes[X[:, 1], X[:, 0]]

        # Compute loss
        score = self.loss(self.g_net(x), flat_Y[:, 0])
        for i in range(self.subtraj - 1):
            x = self.f_net(x)
            score += self.loss(self.g_net(x), flat_Y[:, i + 1])

        # Compute gaps
        if self.ms > 1:
            x = self.f_net(x)
            ind = X[:, 0] < self.ms - 1
            Xn = X[ind]
            penalty = F.mse_loss(x[ind], self.nodes[Xn[:, 1], Xn[:, 0] + 1])
        else:
            penalty = torch.tensor(0.0)

        return score / self.subtraj, penalty

    def train_prediction(self, index):
        s = self.nodes[index : index + 1, 0]
        s_traj = s.new_empty(self.T, self.dim_x)
        s_traj[0] = s
        for t in range(self.T - 1):
            s = self.f_net(s)
            s_traj[t + 1] = s
        return s_traj, self.g_net(s_traj)

    def save(self, data_dir, exp_name):
        save_dir = join("exp_dir", data_dir, exp_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.f_net.state_dict(), join(save_dir, "transition_net.pkl"))
        torch.save(self.g_net.state_dict(), join(save_dir, "observation_net.pkl"))
        torch.save(self.nodes, join(save_dir, "shooting_nodes.pt"))

        with open(join(save_dir, "history.pkl"), "wb") as f:
            pickle.dump(self.train_hist, f)

    def load(self, data_dir, exp_name):
        save_dir = join("exp_dir", data_dir, exp_name)
        self.f_net.load_state_dict(torch.load(join(save_dir, "transition_net.pkl")))
        self.g_net.load_state_dict(torch.load(join(save_dir, "observation_net.pkl")))
        self.nodes = torch.load(join(save_dir, "shooting_nodes.pt"))
        with open(join(save_dir, "history.pkl"), "rb") as f:
            hist = pickle.load(f)
        return hist

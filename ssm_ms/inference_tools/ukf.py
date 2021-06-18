import torch
import numpy as np
from scipy.linalg import sqrtm
import torch.nn.functional as F
from tqdm import tqdm

dtype = torch.float32


def UKF_step(model, x, P, y, R, R_P):
    S = np.array([sqrtm(model.dim_x * p) for p in P])
    if np.sum(np.iscomplex(S)) > 0:
        return x, P, True
    S = x.new_tensor(S)
    x = x.unsqueeze(1)
    X = torch.cat((x + S, x - S), axis=1).type(dtype)
    X_hat = x.new_zeros(X.shape)
    for i in range(X.shape[1]):
        X_hat[:, i] = model.f_net(X[:, i])

    x_hat = torch.sum(X_hat, axis=1) / (2 * model.dim_x)

    Y_hat = x.new_zeros((X.shape[0], X.shape[1], y.shape[1]))
    for i in range(X.shape[1]):
        Y_hat[:, i] = model.g_net(X_hat[:, i])
    y_hat = torch.sum(Y_hat, axis=1) / (2 * model.dim_x)
    x_hat = x_hat.unsqueeze(1)
    y_hat = y_hat.unsqueeze(1)
    Pxx_hat = (
        torch.matmul(torch.transpose(X_hat - x_hat, 1, 2), (X_hat - x_hat))
        / (2 * model.dim_x)
        + R_P
    )
    Pxy_hat = torch.matmul(torch.transpose(X_hat - x_hat, 1, 2), (Y_hat - y_hat)) / (
        2 * model.dim_x
    )
    Pyy_hat = torch.matmul(torch.transpose(Y_hat - y_hat, 1, 2), (Y_hat - y_hat)) / (
        2 * model.dim_x
    ) + R.unsqueeze(0)
    K = torch.matmul(Pxy_hat, torch.inverse(Pyy_hat))
    x = (
        x_hat.squeeze()
        + torch.matmul(K, y.unsqueeze(-1) - torch.transpose(y_hat, 1, 2)).squeeze()
    )
    #  Pxx = Pxx_hat - torch.matmul(K, torch.transpose(Pxy_hat, 1, 2))
    Pxx = Pxx_hat - torch.matmul(K, torch.matmul(Pyy_hat, torch.transpose(K, 1, 2)))
    return x, Pxx.cpu().detach().numpy(), False


def test_UKF(model, test, test_GT, len_filt, Variance):
    N_test, T, dobs = test.shape
    if model.normalize:
        mean, std = model.dict_normalize["mean"], model.dict_normalize["std"]
        test = (test - mean) / std
    Y = model.nodes.new_tensor(test)
    test_GT = model.nodes.new_tensor(test_GT)
    x = Y.new_zeros(N_test, model.dim_x)
    x[:, :dobs] = Y[:, 0]
    P = [np.diag([0.1, 0.1, 0.1]) for i in range(N_test)]
    R = x.new_tensor(Variance * np.eye(dobs))
    R_P = x.new_tensor(1e-6 * np.eye(model.dim_x))
    state_traj = Y.new_zeros(N_test, T, model.dim_x)
    with torch.no_grad():
        for i in range(len_filt):
            z = Y[:, i]
            x, P, stop = UKF_step(model, x, P, z, R, R_P)
            state_traj[:, i] = x
            if stop:
                print("filter exploded")
        for i in range(T - len_filt):
            x = model.f_net(x)
            state_traj[:, len_filt + i] = x

    if model.normalize:
        mean, std = Y.new_tensor(mean), Y.new_tensor(std)
        state_traj[:, :, :dobs] = state_traj[:, :, :dobs] * std + mean
        Y = Y * std + mean
    score0 = F.mse_loss(
        state_traj[:, len_filt - 1, 0], test_GT[:, len_filt - 1].squeeze()
    )
    score = F.mse_loss(state_traj[:, len_filt:, 0], test_GT[:, len_filt:].squeeze())
    state_traj = state_traj.detach().cpu().numpy()
    return state_traj, score0.detach().cpu().numpy(), score.detach().cpu().numpy()

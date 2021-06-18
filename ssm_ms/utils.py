import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

dtype = torch.float32


def ms_loading(train, bs, normalize):
    dict_normalize = {}
    if normalize:
        if train.shape[0] == 1:
            mean = np.mean(train[0], axis=0)
            std = np.std(train[0], axis=0)
        else:
            mean = np.mean(np.concatenate(train, axis=0), axis=0)
            std = np.std(np.concatenate(train, axis=0), axis=0)
        train = (train - mean) / std
        dict_normalize["mean"] = mean
        dict_normalize["std"] = std
    train = torch.tensor(train, dtype=dtype)
    train = list(enumerate(train))
    train_loader = DataLoader(train, batch_size=bs, shuffle=True)

    return train_loader, dict_normalize


def load_pendulum_data(N):
    train_name = f"data/datasets/{N}_link_image_train.npy"
    test_name = f"data/datasets/{N}_link_image_test.npy"
    train = np.load(train_name, allow_pickle=True)
    test = np.load(test_name, allow_pickle=True)
    train_set = np.array([x.toarray() for x in train])
    test_set = np.array([x.toarray() for x in test])
    return train_set, test_set


def split(data, rec):
    N_traj, T, d = data.shape
    if rec > T:
        raise ValueError("Dataset has too short trjactories")
    m = int(T / rec)
    split_set = np.zeros((N_traj * m, rec, d))
    for it in range(m):
        split_set[it * N_traj : (it + 1) * N_traj] = data[:, it * rec : (it + 1) * rec]
    return split_set

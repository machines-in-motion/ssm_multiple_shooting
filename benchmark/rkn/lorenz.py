from tensorflow import keras as k
import tensorflow as tf

tf.random.set_random_seed(0)
import time
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, animation

import numpy as np
import sys

main_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

sys.path.append("../../../RKN/rkn")
from rkn.RKN import RKN
from util.LayerNormalization import LayerNormalization


def split(data, rec):
    N_traj, T, d = data.shape
    if rec > T:
        raise ValueError("Dataset has too short trajectories")
    m = int(T / rec)
    split_set = np.zeros((N_traj * m, rec, d))
    for it in range(m):
        split_set[it * N_traj : (it + 1) * N_traj] = data[:, it * rec : (it + 1) * rec]
    return split_set


data_name = "Lorenz"
data_path = os.path.join(main_folder, "ssm_ms/data/datasets")
train_name = os.path.join(data_path, f"lorenz_train.npy")
test_name = os.path.join(data_path, f"lorenz_test.npy")
test_name_GT = os.path.join(data_path, f"lorenz_test_GT.npy")
train_data = np.load(train_name)
train_data = split(train_data, 10000)
test_data = np.load(test_name)
test_data_GT = np.load(test_name_GT)


mean = np.mean(np.concatenate(train_data))
std = np.std(np.concatenate(train_data))
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
test_data_GT = (test_data_GT - mean) / std


train_obs = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
rs = np.random.RandomState()
train_obs_valid = rs.rand(train_data.shape[0], train_data.shape[1], 1) < 0.5
train_obs_valid[:, :5] = True
train_targets = train_obs.copy()
print(
    "Fraction of Valid Images:",
    np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape),
)
train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0


# len_filt = 1990
# train_obs = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
# train_obs_valid = np.ones((train_obs.shape[0], train_obs.shape[1], 1))
# train_obs_valid[:, len_filt:] = 0
# train_targets = train_obs.copy()
# train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0

len_filt = 1990
test_obs = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))
test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1))
test_obs_valid[:, len_filt:] = 0
test_targets = test_obs.copy()
test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0


print(train_targets.shape)
print(test_targets.shape)

# Implement Encoder and Decoder hidden layers
class LorenzRKN(RKN):
    def build_encoder_hidden(self):
        return [
            # 3: Dense Layer
            k.layers.Dense(256, activation=k.activations.relu),
            k.layers.Dense(256, activation=k.activations.relu),
        ]

    def build_decoder_hidden(self):
        return [
            k.layers.Dense(256, activation=k.activations.relu),
            k.layers.Dense(256, activation=k.activations.relu),
        ]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=256, activation=k.activations.relu)]


# Build Model
rkn = LorenzRKN(
    observation_shape=1,
    latent_observation_dim=60,
    output_dim=1,
    num_basis=30,
    bandwidth=3,
    never_invalid=False,
)
rkn.compile(
    optimizer=k.optimizers.Adam(clipnorm=5.0), loss=rkn.gaussian_nll, metrics=[rkn.rmse]
)

# Train Model
start = time.time()
rkn.fit(
    (train_obs, train_obs_valid),
    train_targets,
    batch_size=50,
    epochs=100,
    validation_data=((test_obs, test_obs_valid), test_targets),
)
print(" Training lasted " + str((time.time() - start) / 60) + " minutes")


print("Train")
predictions = rkn.predict((train_obs, train_obs_valid)).squeeze()
index = 1
obs_pred = predictions[index] * std + mean
obs_GT = train_targets[index] * std + mean

plt.figure()
time = np.linspace(0, obs_GT.shape[0] * 0.005, obs_GT.shape[0])
plt.plot(time, obs_pred[:, 0], label="pred")
plt.plot(time, obs_GT, "-", label="GT")
plt.legend()
plt.savefig("plots/lorenz_train.png")

print("Test")
predictions = rkn.predict((test_obs, test_obs_valid)).squeeze()

# Compute metric
Y = predictions[:, len_filt:] * std + mean
Target = test_data_GT[:, len_filt:] * std + mean
score = np.linalg.norm(Y - Target) ** 2 / Target.shape[0] / Target.shape[1]
print(f"test score: {score}")

index = 1
obs_pred = predictions[index] * std + mean
obs_GT = test_data_GT[index] * std + mean

plt.figure()
time = np.linspace(0, obs_GT.shape[0] * 0.005, obs_GT.shape[0])
plt.plot(time, obs_pred[:, 0], label="pred")
plt.plot(time, obs_GT, "-", label="GT")
plt.axvline(time[len_filt], 0, 1, color="k")
plt.legend()
plt.savefig("plots/lorenz_test.png")

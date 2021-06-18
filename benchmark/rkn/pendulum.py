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

data_name = "Pendulum"
data_path = os.path.join(main_folder, "ssm_ms/data/datasets")
train_name = os.path.join(data_path, f"1_link_image_train.npy")
test_name = os.path.join(data_path, f"1_link_image_test.npy")
train = np.load(train_name, allow_pickle=True)
test = np.load(test_name, allow_pickle=True)
train_data = np.array([x.toarray() for x in train]) * 255.0
test_data = np.array([x.toarray() for x in test]) * 255.0


len_filt = 50
train_obs = train_data.reshape((train_data.shape[0], train_data.shape[1], 24, 24, 1))
train_obs_valid = np.ones((train_obs.shape[0], train_obs.shape[1], 1))
train_obs_valid[:, len_filt:] = 0
train_targets = train_obs.copy()
train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0

test_obs = test_data.reshape((test_data.shape[0], test_data.shape[1], 24, 24, 1))
test_obs_valid = np.ones((test_obs.shape[0], test_obs.shape[1], 1))
test_obs_valid[:, len_filt:] = 0
test_targets = test_obs.copy()
test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0


print(train_targets.shape)
print(test_targets.shape)

# Implement Encoder and Decoder hidden layers
class PendulumImageImputationRKN(RKN):
    def build_encoder_hidden(self):
        return [
            # 1: Conv Layer
            k.layers.Conv2D(12, kernel_size=5, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            k.layers.Conv2D(12, kernel_size=3, padding="same", strides=2),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            k.layers.Flatten(),
            # 3: Dense Layer
            k.layers.Dense(30, activation=k.activations.relu),
        ]

    def build_decoder_hidden(self):
        return [
            k.layers.Dense(144, activation=k.activations.relu),
            k.layers.Lambda(lambda x: tf.reshape(x, [-1, 3, 3, 16])),
            k.layers.Conv2DTranspose(16, kernel_size=5, strides=4, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),
            k.layers.Conv2DTranspose(12, kernel_size=3, strides=2, padding="same"),
            LayerNormalization(),
            k.layers.Activation(k.activations.relu),
        ]


# Build Model
rkn = PendulumImageImputationRKN(
    observation_shape=train_obs.shape[-3:],
    latent_observation_dim=30,
    output_dim=train_targets.shape[-3:],
    num_basis=15,
    bandwidth=3,
    never_invalid=False,
)
rkn.compile(
    optimizer=k.optimizers.Adam(clipnorm=5.0),
    loss=lambda t, p: rkn.bernoulli_nll(t, p, uint8_targets=True),
)

# Train Model
start = time.time()
rkn.fit(
    (train_obs, train_obs_valid),
    train_targets,
    batch_size=50,
    epochs=1000,
    validation_data=((test_obs[:10], test_obs_valid[:10]), test_targets[:10]),
)
print(" Training lasted " + str((time.time() - start) / 60) + " minutes")


def evaluate(pred, GT):
    mse_list = []
    bce_list = []
    for j in range(len(pred)):
        mse_score = 0
        bce_score = 0
        obs_pred = pred[j]
        obs_GT = GT[j].squeeze() / 255.0
        for i in range(pred.shape[1] - len_filt):
            hat = obs_pred[len_filt + i]
            tar = obs_GT[len_filt + i]
            mse_score += np.linalg.norm(tar - hat) ** 2 / (24 * 24)
            bce_score += -np.mean(
                tar * np.log(hat + 1e-12) + (1 - tar) * np.log(1 - hat + 1e-12)
            )
        mse_list.append(mse_score / (pred.shape[1] - len_filt))
        bce_list.append(bce_score / (pred.shape[1] - len_filt))
    print("MSE = ", np.mean(mse_list), "  +-  ", np.std(mse_list))
    print("BCE = ", np.mean(bce_list), "  +-  ", np.std(bce_list))


print("Train")
predictions = rkn.predict((train_obs, train_obs_valid)).squeeze()
evaluate(predictions, train_targets)


print("Test")
predictions = rkn.predict((test_obs, test_obs_valid)).squeeze()
n_traj_test, N, _, _ = predictions.shape
evaluate(predictions, test_targets)


index = 1
obs_pred = predictions[index]
obs_GT = test_targets[index].squeeze() / 255.0


fig, (ax1, ax2) = plt.subplots(1, 2)
jump = 1


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

import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm
from moving_mnist import MovingMNIST

np.random.seed(0)

root = "datasets"
is_train = True
n_frames_input = 10
n_frames_output = 10
num_object = [2]
dataset = MovingMNIST(root, is_train, n_frames_input, n_frames_output, num_object)

N_train = 100000
Train_set = np.empty((N_train, n_frames_input + n_frames_output, 64 * 64))
for i in tqdm(range(N_train)):
    inp, out = dataset[i]
    Train_set[i, :n_frames_input] = inp.reshape(n_frames_input, 64 * 64)
    Train_set[i, n_frames_input:] = out.reshape(n_frames_output, 64 * 64)

Train_set = np.array([csc_matrix(x) for x in Train_set])

np.save("datasets/mnist_train_seq", Train_set)

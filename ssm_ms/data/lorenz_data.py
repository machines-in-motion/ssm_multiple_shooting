import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

np.random.seed(0)


class Lorentz:
    """
    Environment class Lorentz system
    """

    def __init__(self, p, dt_sim):
        super(Lorentz, self).__init__()
        self.p = p
        self.dt = 0.005
        self.d = 3
        self.dobs = 1
        self.n = int(dt_sim / self.dt)

    def f(self, state):
        sigma, beta, rho = self.p
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz])

    def RK4(self, f, x):  ## RK4 step
        k1 = f(x) * self.dt
        k2 = f(x + k1 / 2.0) * self.dt
        k3 = f(x + k2 / 2.0) * self.dt
        k4 = f(x + k3) * self.dt
        y = x + (k1 + 2 * (k2 + k3) + k4) / 6
        return y

    def obs(self, x):
        return x[: self.dobs]

    def step(self, state):
        for i in range(self.n):
            state = self.RK4(self.f, state)
        return state

    def reset(self):
        return 10 * (2 * np.random.rand(self.d) - 1)


p = np.array([10.0, 8 / 3, 28])
dt_sim = 0.005  #  Needs to be larger or equal to 0.005.
model = Lorentz(p, dt_sim)


def generate_data(N_data, T, plot=False):
    N_it = int(T / dt_sim)
    noise = np.random.normal(loc=0, scale=2.5, size=(N_data, N_it, model.dobs))
    dataset = np.empty((N_data, N_it, model.dobs))
    for n in tqdm(range(N_data)):
        x = model.reset()
        dataset[n, 0] = model.obs(x)
        for k in range(N_it - 1):
            x = model.step(x)
            dataset[n, k + 1] = model.obs(x)
        if plot:
            time = np.linspace(0, T, N_it)
            plt.plot(time, dataset[n, :, 0], label="x1")
            #  plt.plot(time, dataset[n, :, 1], label='x2')
            #  plt.plot(time, dataset[n, :, 2], label='x3')
            plt.legend()
            plt.show()
    return dataset + noise, dataset


Train, _ = generate_data(1000, 50, False)
Test, gt = generate_data(1000, 10, False)

np.save(f"datasets/lorenz_train", Train)
np.save(f"datasets/lorenz_test", Test)
np.save(f"datasets/lorenz_test_GT", gt)

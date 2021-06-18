import pinocchio as pin
import hppfcl as fcl
import os
from tqdm import tqdm
import numpy as np
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--n_pendulum", default=1, type=int)
arg = parser.parse_args()
N = arg.n_pendulum

if N == 1:
    n_train, n_test = 1000, 1000
    img_size = 24
    v0 = 1
    friction_coeff = 0
else:
    n_train, n_test = 2000, 2000
    img_size = 48
    v0 = 0
    friction_coeff = 0.1

dt_im, T = 0.1, 10
std = 0.2

model = pin.Model()
geom_model = pin.GeometryModel()

parent_id = 0
joint_placement = pin.SE3.Identity()
body_mass = 1.0
body_radius = 0.1

shape0 = fcl.Sphere(body_radius)
geom0_obj = pin.GeometryObject("base", 0, shape0, pin.SE3.Identity())
geom0_obj.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
geom_model.addGeometryObject(geom0_obj)

for k in range(N):
    joint_name = "joint_" + str(k + 1)
    joint_id = model.addJoint(
        parent_id, pin.JointModelRY(), joint_placement, joint_name
    )

    body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
    body_placement = joint_placement.copy()
    body_placement.translation[2] = 1.0
    model.appendBodyToJoint(joint_id, body_inertia, body_placement)

    geom1_name = "ball_" + str(k + 1)
    shape1 = fcl.Sphere(body_radius)
    geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
    geom1_obj.meshColor = np.ones((4))
    geom_model.addGeometryObject(geom1_obj)

    geom2_name = "bar_" + str(k + 1)
    shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
    shape2_placement = body_placement.copy()
    shape2_placement.translation[2] /= 2.0

    geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
    geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
    geom_model.addGeometryObject(geom2_obj)

    parent_id = joint_id
    joint_placement = body_placement.copy()


plt_width = 8
img_size_internal = 128
cx, cy = img_size_internal / 2, img_size_internal / 2
plt_length = img_size_internal / 2 / N


def create_image(q):
    img = Image.new("F", (img_size_internal, img_size_internal), 0.0)
    draw = ImageDraw.Draw(img)
    x0, y0 = 0, 0
    for k in range(N):
        x1 = x0 - np.sin(np.sum(q[: k + 1]) - np.pi)
        y1 = y0 - np.cos(np.sum(q[: k + 1]) - np.pi)

        draw.line(
            [
                (cx + plt_length * x0, cy - plt_length * y0),
                (cx + plt_length * x1, cy - plt_length * y1),
            ],
            fill=1.0,
            width=plt_width,
        )
        x0, y0 = x1, y1
    img = img.resize((img_size, img_size), resample=Image.ANTIALIAS)
    img = np.asarray(img)
    noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
    img = np.clip(img + noise, 0, 1)
    return img


dt = 0.001
N_it = math.floor(T / dt)

model.lowerPositionLimit.fill(-math.pi)
model.upperPositionLimit.fill(+math.pi)


def generate_data(N_data, plot=False):
    dataset = np.zeros((N_data, int(T / dt_im), img_size * img_size))
    for n in tqdm(range(N_data)):
        q = (2 * np.random.rand(model.nv) - 1) * math.pi
        v = v0 * (2 * np.random.rand(model.nv) - 1)
        data_sim = model.createData()
        i = 0
        for k in range(N_it):
            tau_control = -friction_coeff * v
            a = pin.aba(model, data_sim, q, v, tau_control)  # Forward dynamics
            v += a * dt
            q = pin.integrate(model, q, v * dt)
            if k % int(dt_im / dt) == 0:
                im = create_image(q)
                if plot:
                    plt.imshow(im, cmap="gray")
                    plt.pause(dt)
                    plt.draw()
                dataset[n, i] = im.flatten()
                i += 1
    return dataset


train_set = generate_data(n_train)
test_set = generate_data(n_test)
train_set = np.array([csc_matrix(x) for x in train_set])
test_set = np.array([csc_matrix(x) for x in test_set])
if not os.path.exists("datasets"):
    os.makedirs("datasets")
np.save(f"datasets/{N}_link_image_train", train_set)
np.save(f"datasets/{N}_link_image_test", test_set)

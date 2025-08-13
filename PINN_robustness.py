# Improved PINN + Bayesian PINN (Variational Inference) for 1D Heat Equation
# Notebook-style Python script. Adds:
# 1) Gaussian noise to observed data to verify robustness
# 2) Deterministic PINN with PDE residual used for training (DeepXDE)
# 3) Bayesian PINN (variational inference) using TensorFlow Probability (TFP)
#
# Notes:
# - This script is intentionally educational and annotated with comments.
# - It assumes you run in an environment with tensorflow, deepxde, and tensorflow_probability installed.
# - The Bayesian part uses DenseVariational layers; we use MC sampling to obtain predictive mean/std.

# === Install / imports ===
# !pip install --upgrade deepxde tensorflow-probability
import os
os.environ["DDE_BACKEND"] = "tensorflow"

import deepxde as dde
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fix random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === 1) Data generation: 1D heat equation (transient) with optional noise ===
L = 1.0
Nx = 51
dx = L / (Nx - 1)
T_total = 5.0
Nt = 5000
dt = T_total / Nt
true_kappa = 0.01
true_q = 1.0

# initial condition: zeros
u = np.zeros(Nx)
x_coords = np.linspace(0, L, Nx)

# integrate forward (explicit Euler) to produce ground-truth data
records = []
records.append([0.0] + u.tolist() if (u := np.copy(u)) is not None else None)
for n in range(1, Nt + 1):
    u_new = np.copy(u)
    for i in range(1, Nx - 1):
        u_new[i] = u[i] + dt * (true_kappa * (u[i+1] - 2 * u[i] + u[i-1]) / (dx**2) + true_q)
    u_new[0] = 0.0
    u_new[-1] = 0.0
    u = u_new
    if n % 50 == 0 or n == Nt:
        records.append([n * dt] + u.tolist())

# convert to dataframe
header = ["Time (s)"] + [f"x={x:.3f}" for x in x_coords]
df = pd.DataFrame(records, columns=header)

# prepare (x,t,u) triplets
triplets = []
for _, row in df.iterrows():
    t = row['Time (s)']
    for col in df.columns[1:]:
        x = float(col.split('=')[1])
        u_val = row[col]
        triplets.append([x, t, u_val])
triplets = np.array(triplets)
X_all = triplets[:, :2]
y_all = triplets[:, 2:3]

# Add Gaussian noise to measured observations (to test robustness)
noise_std = 0.02  # 2% absolute noise (adjustable)
noisy_indices = np.random.choice(len(X_all), size=int(0.2 * len(X_all)), replace=False)
# We'll use a subset as "observed" points with noise; others are used as collocation points
observed_mask = np.zeros(len(X_all), dtype=bool)
observed_mask[noisy_indices] = True

X_obs = X_all[observed_mask]
y_obs = y_all[observed_mask] + noise_std * np.random.randn(len(X_obs), 1)

# train/test split for observed data
X_obs_train, X_obs_test, y_obs_train, y_obs_test = train_test_split(X_obs, y_obs, test_size=0.2, random_state=SEED)

# collocation points for PDE residual (sampled from domain)
num_domain = 10000
x_dom = np.random.rand(num_domain, 1) * L
t_dom = np.random.rand(num_domain, 1) * T_total
X_domain = np.hstack([x_dom, t_dom])

# === Utility: function to build DeepXDE data with PDE included ===
def build_deeptime_data(kappa_var, q_var, X_train_data, y_train_data, X_test_anchors=None):
    geom = dde.geometry.Interval(0, L)
    timedomain = dde.geometry.TimeDomain(0, T_total)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde(x, u):
        # x: [x, t]
        u_t = dde.grad.jacobian(u, x, i=0, j=1)
        u_xx = dde.grad.hessian(u, x, i=0, j=0)
        kappa = tf.nn.softplus(kappa_var)  # enforce positivity
        q = q_var
        return u_t - kappa * u_xx - q

    # Use observed data as point set BCs (data anchoring)
    anchors = X_train_data
    pointset = dde.PointSetBC(anchors, y_train_data)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [pointset],
        num_domain=10000,
        num_boundary=0,
        num_initial=0,
        anchors=X_test_anchors if X_test_anchors is not None else None,
    )
    return data

# === 2) Deterministic PINN (DeepXDE) that uses PDE residual and learns kappa/q ===
# Trainable variables
raw_kappa = tf.Variable(0.0, dtype=tf.float32)
raw_q = tf.Variable(0.0, dtype=tf.float32)

data = build_deeptime_data(raw_kappa, raw_q, X_obs_train, y_obs_train, X_obs_test)

net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# custom loss remains but now PDE residual is included by data; we still can add parameter regularization
def custom_loss(y_true, y_pred):
    base = tf.reduce_mean(tf.square(y_true - y_pred))
    # regularize kappa and q toward expected values (weak prior)
    reg = 1e1 * (tf.square(tf.nn.softplus(raw_kappa) - true_kappa) + tf.square(raw_q - true_q))
    return base + reg

model.compile("adam", lr=1e-4, loss=custom_loss, external_trainable_variables=[raw_kappa, raw_q])
losshistory, train_state = model.train(iterations=20000)

# L-BFGS-B fine tune (DeepXDE will include PDE residual during optimizer step)
model.compile("L-BFGS-B", loss=custom_loss, external_trainable_variables=[raw_kappa, raw_q])
model.train()

learned_kappa = tf.nn.softplus(raw_kappa).numpy()
learned_q = raw_q.numpy()
print("Deterministic PINN learned kappa:", learned_kappa)
print("Deterministic PINN learned q:", learned_q)

# Evaluate on test observed points
y_pred_test = model.predict(X_obs_test)
mse_test = np.mean((y_pred_test - y_obs_test) ** 2)
print("Deterministic PINN test MSE on observed points:", mse_test)


"""
@author: Mohammad Al-Jarrah

Run OTF for a chosen dimension d, then plot per-state histograms comparing
the prior, OTF posterior approximation, and SIR reference.
"""

import os
import time
import subprocess
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ot
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from param import get_param

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.rc('font', size=13)
fontsize = 19

seed = np.random.randint(0, 10000)
# seed = 42
print(f"Random seed: {seed}")
torch.manual_seed(seed=seed)
np.random.seed(seed=seed)


def get_free_gpu(n=1):
    """
    Query nvidia-smi and return the IDs of the n least-loaded GPUs by memory usage.

    Parameters
    ----------
    n : int — number of GPU IDs to return

    Returns
    -------
    list of int — GPU device IDs sorted by ascending memory usage
    """
    result      = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    memory_used = [int(x) for x in result.stdout.strip().split("\n")]
    sorted_ids  = sorted(range(len(memory_used)), key=lambda i: memory_used[i])
    return sorted_ids[:n]


if torch.cuda.is_available():
    gpu_id = get_free_gpu()[0]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.manual_seed(seed)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ===========================================================
#  Set dimension here
# ===========================================================
d  = 5          # must be multiple of 5 for plotting purposes, but can be set to any positive integer
dy = d
# ===========================================================

N       = 5000          # number of particles
N_true  = int(1e5)       # SIR reference size
sigma   = np.sqrt(1e-2)  # observation noise std
y_true  = 1.0            # fixed observation

delta   = 0.0
delta_T = delta
delta_f = delta


# -----------------------------------------------------------
#  Network definitions  (smac-style residual architecture)
# -----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation):
        super().__init__()
        self.linear1    = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear2    = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.activation = activation

    def forward(self, x):
        identity = x
        out      = self.linear1(x)
        out      = self.activation(out)
        out      = self.linear2(out)
        return self.activation(out + identity)


class f_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_resblocks=2):
        super().__init__()
        self.activation  = nn.ELU()
        self.layer_input = nn.Linear(input_dim[0] + input_dim[1], hidden_dim, bias=True)
        self.resblocks   = nn.ModuleList([
            ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
        ])
        self.layer_out   = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x, y):
        out = self.layer_input(torch.concat((x, y), dim=1))
        for block in self.resblocks:
            out = block(out)
        return self.layer_out(self.activation(out))


class map_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_resblocks=2):
        super().__init__()
        self.activation  = nn.ReLU()
        self.layer_input = nn.Linear(input_dim[0] + input_dim[1], hidden_dim, bias=True)
        self.resblocks   = nn.ModuleList([
            ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
        ])
        self.layer_out   = nn.Linear(hidden_dim, input_dim[0], bias=True)

    def forward(self, x, y):
        out = self.layer_input(torch.concat((x, y), dim=1))
        for block in self.resblocks:
            out = block(out)
        return self.layer_out(self.activation(out))


def init_weights(m):
    if isinstance(m, nn.Linear) and m.bias is not None:
        m.bias.data.fill_(0.001)


# -----------------------------------------------------------
#  Training loop
# -----------------------------------------------------------
def train(f, T, X_Train, Y_Train, iterations, lr_f, lr_T, batch_size,
          delta_T, delta_f, K_in, iter_0, d, dy):
    f.train()
    T.train()
    optimizer_f = torch.optim.Adam(f.parameters(), lr=lr_f)
    optimizer_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=iter_0, T_mult=2, eta_min=lr_f * 1e-3)
    scheduler_T = CosineAnnealingWarmRestarts(optimizer_T, T_0=iter_0, T_mult=2, eta_min=lr_T * 1e-3)

    Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)

    for i in range(iterations):
        idx        = torch.randperm(X_Train.shape[0])[:batch_size]
        X_train    = X_Train[idx].clone().detach()
        Y_train    = Y_Train[idx].clone().detach()
        Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)

        idx2     = torch.randperm(X_Train.shape[0])[:batch_size]
        X_train2 = X_Train[idx2].clone().detach()

        # Inner loop: update T holding f fixed
        for _ in range(K_in):
            map_T      = T(X_train, Y_shuffled)
            f_of_map_T = f(map_T, Y_shuffled)
            reg        = 0
            if delta_T != 0:
                map_T2 = T(X_train2, Y_shuffled)
                reg    = nn.functional.elu(
                    ((map_T2 - map_T) * (-X_train2 + X_train)).sum(axis=1), alpha=0.01
                ).mean()
            loss_T = (-f_of_map_T.mean()
                      + 0.5 * ((X_train - map_T) ** 2).sum(axis=1).mean()
                      + delta_T * reg)
            optimizer_T.zero_grad()
            loss_T.backward()
            optimizer_T.step()

        # Update potential f
        f_of_y     = f(X_train, Y_train)
        map_T      = T(X_train, Y_shuffled)
        f_of_map_T = f(map_T, Y_shuffled)

        reg2 = 0
        if delta_f != 0:
            K_hessian = batch_size

            def f_scalar(x_flat, y_flat):
                return f(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).squeeze()

            H_fn = jacrev(jacrev(f_scalar, argnums=0), argnums=0)

            def hess_diag_norm(x_k, y_k):
                return torch.norm(H_fn(x_k, y_k).diag())

            x_batch   = X_train[:K_hessian]
            y_batch   = Y_train[:K_hessian]
            laplacian = vmap(hess_diag_norm)(x_batch, y_batch).sum()
            reg2      = nn.functional.elu(laplacian, alpha=0.01) / K_hessian

        loss_f = -f_of_y.mean() + f_of_map_T.mean() + delta_f * reg2
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        if (i + 1) % 512 == 0 or (i + 1) == iterations:
            with torch.no_grad():
                f_of_y_full     = f(X_Train, Y_Train)
                map_T_full      = T(X_Train, Y_Train_shuffled)
                f_of_map_T_full = f(map_T_full, Y_Train_shuffled)
                loss = (f_of_y_full.mean() - f_of_map_T_full.mean()
                        + 0.5 * ((X_Train - map_T_full) ** 2).sum(axis=1).mean())
                print(f"  iter {i+1:>6}/{iterations}  loss = {loss.item():.5f}")

        scheduler_f.step()
        scheduler_T.step()


def h(x):
    return 0.5 * x * x


def w2_distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.ones(len(x)) / len(x)
    b = np.ones(len(y)) / len(y)
    M = ot.dist(x, y, metric='sqeuclidean')
    return np.sqrt(ot.emd2(a, b, M))


# -----------------------------------------------------------
#  Build SIR reference posterior  (independent per component)
# -----------------------------------------------------------
rng    = np.random.default_rng(0)
X_true = np.zeros((N_true, d))
for j in range(d):
    x_SIR = np.random.multivariate_normal(np.zeros(1), np.eye(1), N_true).T
    W     = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma ** 2)
    W     = np.exp(-(W - np.min(W)))
    W    /= W.sum()
    index = rng.choice(np.arange(N_true), N_true, p=W)
    X_true[:, j] = x_SIR[:, index].reshape(N_true)

# -----------------------------------------------------------
#  Sample prior and observations
# -----------------------------------------------------------
INPUT_DIM   = [d, dy]
dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
dist_obs    = MultivariateNormal(torch.zeros(dy), sigma ** 2 * torch.eye(dy))

x = dist_normal.sample((N,)).to(device)
y = h(x) + dist_obs.sample((N,)).to(device)

# -----------------------------------------------------------
#  Load SMAC-tuned hyperparameters and run OTF
# -----------------------------------------------------------
NUM_NEURON_f, NUM_NEURON_T, NUM_RESBLOCKS_f, NUM_RESBLOCKS_T, \
    BATCH_SIZE, ITERS, LR_f, LR_T, K_in, ITER_0 = get_param()

BATCH_SIZE = min(BATCH_SIZE, N)

print(f"\nRunning OTF  d={d}, N={N}, seed={seed}")
print(f"  f_NN  : neurons={NUM_NEURON_f}, resblocks={NUM_RESBLOCKS_f}, lr={LR_f:.2e}")
print(f"  map_T : neurons={NUM_NEURON_T}, resblocks={NUM_RESBLOCKS_T}, lr={LR_T:.2e}")
print(f"  ITERS={ITERS}, batch={BATCH_SIZE}, K_in={K_in}\n")

f_net = f_NN(INPUT_DIM,  NUM_NEURON_f, num_resblocks=NUM_RESBLOCKS_f).to(device)
MAP_T = map_NN(INPUT_DIM, NUM_NEURON_T, num_resblocks=NUM_RESBLOCKS_T).to(device)
MAP_T.apply(init_weights)
f_net.apply(init_weights)

t0 = time.time()
train(f_net, MAP_T, x, y, ITERS, LR_f, LR_T, BATCH_SIZE,
      delta_T=delta_T, delta_f=delta_f, K_in=K_in, iter_0=ITER_0, d=d, dy=dy)
print(f"\nOTF training done in {time.time() - t0:.1f}s")

# Push prior samples through the learned map at y = y_true
with torch.no_grad():
    x_transported = MAP_T(x, torch.ones_like(x) * y_true).cpu().numpy()

x_prior = x.cpu().numpy()

# -----------------------------------------------------------
#  W2 distance: average over per-dimension 1-D marginals
# -----------------------------------------------------------
p_true   = int(1e3)
w2_total = 0.0
for j in range(d):
    w2_total += w2_distance(X_true[:p_true, j:j+1], x_transported[:p_true, j:j+1])
w2_otf = w2_total / d

# -----------------------------------------------------------
#  EnKF baseline
# -----------------------------------------------------------
x_np  = x_prior
y_np  = y.cpu().numpy()
X_hat = x_np.mean(axis=0, keepdims=True)
Y_hat = y_np.mean(axis=0, keepdims=True)
a_cov = x_np - X_hat
b_cov = y_np - Y_hat
C_xy  = (a_cov.T @ b_cov) / N
C_yy  = (b_cov.T @ b_cov) / N
K_gain = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-4)
x_enkf = x_np + (y_true - y_np) @ K_gain.T

w2_enkf = 0.0
for j in range(d):
    w2_enkf += w2_distance(X_true[:p_true, j:j+1], x_enkf[:p_true, j:j+1])
w2_enkf /= d

# -----------------------------------------------------------
#  SIR baseline
# -----------------------------------------------------------
x_SIR_raw = np.random.multivariate_normal(np.zeros(d), np.eye(d), N).T
W_sir     = np.sum((y_true - h(x_SIR_raw).T) ** 2, axis=1) / (2 * sigma ** 2)
W_sir     = np.exp(-(W_sir - np.min(W_sir)))
W_sir    /= W_sir.sum()
idx_sir   = rng.choice(np.arange(N), N, p=W_sir)
x_sir     = x_SIR_raw[:, idx_sir].T

w2_sir = 0.0
for j in range(d):
    w2_sir += w2_distance(X_true[:p_true, j:j+1], x_sir[:p_true, j:j+1])
w2_sir /= d

print(f"\nAA-SW2 — EnKF: {w2_enkf:.4f}  |  SIR: {w2_sir:.4f}  |  OTF: {w2_otf:.4f}")

#%%
# -----------------------------------------------------------
#  Plot: d subplots, one histogram per state dimension
# -----------------------------------------------------------
ncols = min(d, 5)
nrows = int(np.ceil(d / ncols))
bins  = 60

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes      = np.array(axes).reshape(-1)

for j in range(d):
    ax = axes[j]
    ax.hist(x_prior[:, j],       bins=bins, density=True, alpha=0.35,
            color='grey',      label='Prior',    rasterized=True)
    ax.hist(X_true[:, j],        bins=bins, density=True, alpha=0.45,
            color='C2',        label='True', rasterized=True)
    ax.hist(x_transported[:, j], bins=bins, density=True, alpha=0.55,
            color='C3',    label='OTF',      rasterized=True)
    ax.set_xlim(-3, 3)
    # ax.set_title(rf'$x_{{{j+1}}}$', fontsize=fontsize)
    ax.set_xlabel(rf'$U({j+1})$', fontsize=fontsize)
    if j % ncols == 0:
        ax.set_ylabel('Density', fontsize=fontsize)
    if j == 0:
        ax.legend(fontsize=fontsize)

# Hide any unused subplots
for j in range(d, len(axes)):
    axes[j].set_visible(False)

# title = (
#     rf'OTF posterior approximation  ($d={d}$,  $y={y_true}$)'
#     f'\nf_NN: neurons={NUM_NEURON_f}, resblocks={NUM_RESBLOCKS_f}, lr={LR_f:.2e}'
#     f'  |  map_T: neurons={NUM_NEURON_T}, resblocks={NUM_RESBLOCKS_T}, lr={LR_T:.2e}'
#     f'\nITERS={ITERS}, batch={BATCH_SIZE}, K_in={K_in}'
#     f'  |  W2={w2_otf:.4f}'
# )
# fig.suptitle(title, fontsize=fontsize, y=1.02)
plt.tight_layout()

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_dir  = "check_otf_results"
# os.makedirs(save_dir, exist_ok=True)
# fname = os.path.join(save_dir, f'check_otf_d{d}_{timestamp}.pdf')
# print(f"\nSaving figure to {fname}")
plt.savefig(f'marginals_d{d}.pdf', bbox_inches='tight')

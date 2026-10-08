"""
@author: Mohammad Al-Jarrah

Run OTF for a chosen dimension d, then plot per-state histograms comparing
the prior, OTF posterior approximation, and SIR reference.
"""

import os
import time
import subprocess

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
matplotlib.rcParams['savefig.transparent'] = False   # True would also blank the axes patch
matplotlib.rcParams['savefig.facecolor']   = 'none'  # transparent outside the axes
matplotlib.rcParams['savefig.edgecolor']   = 'none'
matplotlib.rcParams['axes.facecolor']      = 'white' # white inside the axes
matplotlib.rcParams['figure.facecolor']    = 'none'
plt.rc('font', size=19)
fontsize = 50  # large, since the d marginals are tiled into a wide grid of subplots

seed = np.random.randint(0, 10000)  # fresh seed every run; printed so a run can be reproduced
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
    visible     = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        # nvidia-smi ignores CUDA_VISIBLE_DEVICES, so restrict its output to the
        # visible devices and re-index locally to stay valid for torch.
        ids         = [int(i) for i in visible.split(",") if i.strip() != ""]
        memory_used = [memory_used[i] for i in ids]
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
d  = 10  # must be multiple of 5 for plotting purposes, but can be set to any positive integer
dy = d   # every state component is observed
# ===========================================================

N      = 5000           # number of particles
N_true = int(1e5)       # SIR reference size
sigma  = np.sqrt(1e-2)  # observation noise std
y_true = 1.0            # fixed observation; with h(x) = x^2/2 the posterior of each component is bimodal at about ±sqrt(2)

n_projections = 500  # number of random slices used by the regular SW2
proj_seed     = 0    # seed fixing the random slices (shared by all methods)

delta   = 0.0    # OTF regularization weight (λ = 0: unregularized)
delta_T = delta  # weight of the monotonicity penalty on the map T
delta_f = delta  # weight of the Hessian penalty on the potential f


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
    # Small positive bias, so ReLU units start active instead of dead
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

    # Shuffling Y breaks the (x, y) pairing, giving samples of the product of marginals P_X ⊗ P_Y
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

        # Report the full-data loss every 512 iterations and at the end
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
    # Quadratic observation function: the sign of x is unobservable, hence the bimodal posterior
    return 0.5 * x * x


def get_projections(d, n_proj, seed):
    """
    Draw n_proj directions uniformly on the unit sphere of R^d.

    Parameters
    ----------
    d      : int -- ambient dimension
    n_proj : int -- number of slices
    seed   : int -- seed of the generator, so the slices are reproducible

    Returns
    -------
    ndarray of shape (d, n_proj) -- unit-norm column directions
    """
    rng_proj = np.random.default_rng(seed)
    theta    = rng_proj.standard_normal((d, n_proj))
    return theta / np.linalg.norm(theta, axis=0, keepdims=True)


def sw2_distance(X, Y, projections):
    """
    Compute the regular sliced W2 distance between two d-dimensional point clouds.

    This is a distance between the full joint distributions, not between their
    marginals: each direction theta mixes all d coordinates, so the scalars
    theta.x carry the joint structure of the cloud.

    Parameters
    ----------
    X, Y        : ndarray of shape (n, d) -- the two empirical d-dimensional clouds
    projections : ndarray of shape (d, n_proj) -- unit-norm directions in R^d

    Returns
    -------
    float -- (mean over theta of W2^2(theta.X, theta.Y))^(1/2)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    return ot.sliced_wasserstein_distance(X, Y, projections=projections, p=2)


# -----------------------------------------------------------
#  Build SIR reference posterior  (independent per component)
# -----------------------------------------------------------
rng    = np.random.default_rng(0)
X_true = np.zeros((N_true, d))  # reference posterior samples, (N_true, d)
# The prior and likelihood factorize over components, so each coordinate is resampled independently
for j in range(d):
    x_SIR = np.random.multivariate_normal(np.zeros(1), np.eye(1), N_true).T  # (1, N_true)
    W     = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma ** 2)   # negative log-likelihood
    W     = np.exp(-(W - np.min(W)))  # shift by the minimum before exponentiating, to avoid underflow
    W    /= W.sum()
    index = rng.choice(np.arange(N_true), N_true, p=W)
    X_true[:, j] = x_SIR[:, index].reshape(N_true)

# -----------------------------------------------------------
#  Sample prior and observations
# -----------------------------------------------------------
INPUT_DIM   = [d, dy]
dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
dist_obs    = MultivariateNormal(torch.zeros(dy), sigma ** 2 * torch.eye(dy))

x = dist_normal.sample((N,)).to(device)      # prior samples, (N, d)
y = h(x) + dist_obs.sample((N,)).to(device)  # simulated observations, (N, dy)

# -----------------------------------------------------------
#  Load SMAC-tuned hyperparameters and run OTF
# -----------------------------------------------------------
NUM_NEURON_f, NUM_NEURON_T, NUM_RESBLOCKS_f, NUM_RESBLOCKS_T, \
    BATCH_SIZE, ITERS, LR_f, LR_T, K_in, ITER_0 = get_param()

BATCH_SIZE = min(BATCH_SIZE, N)  # the tuned batch size cannot exceed the number of particles

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
    x_transported = MAP_T(x, torch.ones_like(x) * y_true).cpu().numpy()  # OTF posterior samples, (N, d)

x_prior = x.cpu().numpy()

# -----------------------------------------------------------
#  SW2 distance on the full joint clouds, sliced along random directions
# -----------------------------------------------------------
p_true      = int(5e3)  # reference particles used in the SW2 computation
projections = get_projections(d, n_projections, proj_seed)
sw2_otf     = sw2_distance(X_true[:p_true], x_transported[:p_true], projections)

# -----------------------------------------------------------
#  EnKF baseline
# -----------------------------------------------------------
x_np   = x_prior
y_np   = y.cpu().numpy()
X_hat  = x_np.mean(axis=0, keepdims=True)
Y_hat  = y_np.mean(axis=0, keepdims=True)
a_cov  = x_np - X_hat
b_cov  = y_np - Y_hat
C_xy   = (a_cov.T @ b_cov) / N                            # state-observation cross-covariance, (d, dy)
C_yy   = (b_cov.T @ b_cov) / N                            # observation covariance, (dy, dy)
K_gain = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-4)   # small jitter keeps C_yy invertible
x_enkf = x_np + (y_true - y_np) @ K_gain.T                # perturbed-observation EnKF update

sw2_enkf = sw2_distance(X_true[:p_true], x_enkf[:p_true], projections)

# -----------------------------------------------------------
#  SIR baseline
# -----------------------------------------------------------
x_SIR_raw = np.random.multivariate_normal(np.zeros(d), np.eye(d), N).T
W_sir     = np.sum((y_true - h(x_SIR_raw).T) ** 2, axis=1) / (2 * sigma ** 2)
W_sir     = np.exp(-(W_sir - np.min(W_sir)))
W_sir    /= W_sir.sum()
idx_sir   = rng.choice(np.arange(N), N, p=W_sir)
x_sir     = x_SIR_raw[:, idx_sir].T

sw2_sir = sw2_distance(X_true[:p_true], x_sir[:p_true], projections)

print(f"\nSW2 — EnKF: {sw2_enkf:.4f}  |  SIR: {sw2_sir:.4f}  |  OTF: {sw2_otf:.4f}")

#%%
# -----------------------------------------------------------
#  Plot: d subplots, one histogram per state dimension
# -----------------------------------------------------------
ncols = min(d, 10)               # at most 10 marginals per row
nrows = int(np.ceil(d / ncols))
bins  = 60                       # histogram bins per marginal

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8 * nrows))
axes      = np.array(axes).reshape(-1)

for j in range(d):
    ax = axes[j]
    ax.hist(x_prior[:, j],       bins=bins, density=True, alpha=0.35,
            color='grey', label='Prior', rasterized=True)
    ax.hist(X_true[:, j],        bins=bins, density=True, alpha=0.45,
            color='C2',   label='True',  rasterized=True)
    ax.hist(x_transported[:, j], bins=bins, density=True, alpha=0.55,
            color='C3',   label='OTF',   rasterized=True)
    ax.set_xlim(-3, 3)
    # ax.set_title(rf'$x_{{{j+1}}}$', fontsize=fontsize)
    ax.set_xlabel(rf'$U({j+1})$', fontsize=fontsize)
    if j % ncols == 0:  # y-label only on the first column
        ax.set_ylabel('Density', fontsize=fontsize)

# Hide any unused subplots
for j in range(d, len(axes)):
    axes[j].set_visible(False)

# title = (
#     rf'OTF posterior approximation  ($d={d}$,  $y={y_true}$)'
#     f'\nf_NN: neurons={NUM_NEURON_f}, resblocks={NUM_RESBLOCKS_f}, lr={LR_f:.2e}'
#     f'  |  map_T: neurons={NUM_NEURON_T}, resblocks={NUM_RESBLOCKS_T}, lr={LR_T:.2e}'
#     f'\nITERS={ITERS}, batch={BATCH_SIZE}, K_in={K_in}'
#     f'  |  SW2={sw2_otf:.4f}'
# )
# fig.suptitle(title, fontsize=fontsize, y=1.02)
plt.tight_layout()

# Single shared legend above the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
           ncol=len(labels), fontsize=fontsize, frameon=False)

plt.savefig(f'marginals_d{d}.pdf', bbox_inches='tight')

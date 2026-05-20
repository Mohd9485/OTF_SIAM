"""
@author: Mohammad Al-Jarrah
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
from torch.func import vmap, jacrev
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.distributions.multivariate_normal import MultivariateNormal
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from smac import Scenario, HyperparameterOptimizationFacade as HPO
import ot

# --- Plot settings ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- Reproducibility ---
np.random.seed(0)
torch.manual_seed(0)

# --- Device selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------------------------------------
#  Network definitions  (smac-style residual architecture)
# -----------------------------------------------------------

class ResidualBlock(nn.Module):
    """A residual block with two linear layers and an activation-gated skip connection."""

    def __init__(self, hidden_dim, activation):
        super().__init__()
        self.linear1    = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear2    = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.activation = activation

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out + identity)
        return out


class f_NN(nn.Module):
    """Neural network representing the Kantorovich potential f(x, y)."""

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
    """Neural network representing the optimal transport map T(x, y)."""

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
    """Initialize biases of linear layers to a small positive constant."""
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.fill_(0.001)


def train(f, T, X_Train, Y_Train, iterations, lr_f, lr_T, batch_size, delta_T, delta_f, K_in, iter_0):
    """
    Train the Kantorovich potential f and transport map T jointly.

    Alternates between an inner loop that updates T (holding f fixed) and an
    outer update of f, following a regularized dual OT formulation. Regularization
    strength for both networks is controlled by delta_T and delta_f respectively.
    """
    f.train()
    T.train()
    optimizer_f = torch.optim.Adam(f.parameters(), lr=lr_f)
    optimizer_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=iter_0, T_mult=2, eta_min=lr_f * 1e-3)
    scheduler_T = CosineAnnealingWarmRestarts(optimizer_T, T_0=iter_0, T_mult=2, eta_min=lr_T * 1e-3)

    Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)

    for _ in range(iterations):
        idx        = torch.randperm(X_Train.shape[0])[:batch_size]
        X_train    = X_Train[idx].clone().detach()
        Y_train    = Y_Train[idx].clone().detach()
        Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)

        idx2     = torch.randperm(X_Train.shape[0])[:batch_size]
        X_train2 = X_Train[idx2].clone().detach()

        # Inner loop: update transport map T while holding f fixed
        for _ in range(K_in):
            map_T      = T(X_train, Y_shuffled)
            f_of_map_T = f(map_T, Y_shuffled)
            reg = 0
            if delta_T != 0:
                map_T2 = T(X_train2, Y_shuffled)
                # Monotonicity regularization: penalize violations of (T(x2)-T(x))*(x2-x) >= 0
                reg = nn.functional.elu(
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

        # Optional Hessian regularization to promote convexity of f
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

        scheduler_f.step()
        scheduler_T.step()


def h(x):
    """Forward model: y = 0.5 * x^2, applied element-wise."""
    return 0.5 * x * x


def w2_distance(x, y):
    """Compute the 2-Wasserstein distance between two empirical distributions using exact OT."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError("x and y must each contain at least 2 samples.")
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(x, y, metric='sqeuclidean')  # squared Euclidean cost matrix
    return np.sqrt(ot.emd2(a, b, M))         # returns W2 (not squared)


# --- Problem setup ---
d         = 20              # state dimension
lamda     = 0              # regularization strength (λ=0 for SMAC tuning)
dy        = d              # observation dimension
INPUT_DIM = [d, dy]        # input dimensions for both networks: [state dim, observation dim]
sigma     = np.sqrt(1e-2)  # observation noise standard deviation
N         = 5000           # number of particles used for training and evaluation
N_true    = int(1e5)       # number of reference posterior samples
AVG_SIM   = 2              # number of evaluations averaged per config to reduce noise
y_true    = 1.0            # observed value at which to evaluate the posterior

_SHARED_DATA_FILE = f"smac_shared_data_d{d}.npz"

# Lazy cache: each worker process loads the shared data once on first call
_cache = {}


def _load_data():
    """Load training data and reference posterior from disk, caching after first read."""
    if not _cache:
        data = np.load(_SHARED_DATA_FILE)
        _cache['X_true'] = data['X_true']
        _cache['x_data'] = torch.tensor(data['x_data']).to(device)
        _cache['y_data'] = torch.tensor(data['y_data']).to(device)
    return _cache


# ----------------------------------------------------------
#  SMAC Configuration Space
# ----------------------------------------------------------
cs = ConfigurationSpace(seed=42)
cs.add([
    Float(  "lr1",        (1e-5, 1e-2), log=True, default=6e-4),  # learning rate for f
    Float(  "lr2",        (1e-5, 1e-2), log=True, default=6e-3),  # learning rate for T
    Integer("nns1",       (1, 14),      default=12),  # x32 → hidden width of f
    Integer("nns2",       (1, 14),      default=12),  # x32 → hidden width of T
    Integer("nbs1",       (1, 3),       default=1),   # number of residual blocks in f
    Integer("nbs2",       (1, 3),       default=1),   # number of residual blocks in T
    Integer("batch_size", (1, 10),      default=12),   # x32 → mini-batch size
    Categorical("ITERATION", [1, 3, 7], default=3),   # x512 → total iterations
    Integer("K_in",       (1, 6),       default=5),   # x5 → inner-loop updates per outer step
])


# ----------------------------------------------------------
#  SMAC target function
# ----------------------------------------------------------
def target_fun(config, seed: int = 0) -> float:
    """
    Train OTF with the given hyperparameter config and return the average W2
    distance to the SIR reference posterior. SMAC minimises this value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Pin each worker to a separate GPU if CUDA is available
    local_device = device
    if device.type == 'cuda':
        try:
            from dask.distributed import get_worker
            worker    = get_worker()
            name      = str(worker.name)
            worker_id = int(name) if name.isdigit() else hash(name) % torch.cuda.device_count()
        except Exception:
            worker_id = 0
        local_device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")

    # Load shared data (each worker process reads once, then caches in memory)
    shared = _load_data()
    X_true = shared['X_true']
    x_data = shared['x_data'].to(local_device)
    y_data = shared['y_data'].to(local_device)

    NUM_NEURON_f    = int(config["nns1"] * 32)        # hidden layer width of f
    NUM_NEURON_T    = int(config["nns2"] * 32)        # hidden layer width of T
    NUM_RESBLOCKS_f = int(config["nbs1"])              # number of residual blocks in f
    NUM_RESBLOCKS_T = int(config["nbs2"])              # number of residual blocks in T
    BATCH_SIZE      = int(config["batch_size"] * 32)  # mini-batch size per iteration
    ITERS           = int(config["ITERATION"] * 512)  # total number of training iterations
    LR_f            = float(config["lr1"])             # learning rate for f
    LR_T            = float(config["lr2"])             # learning rate for T
    K_in            = int(config["K_in"] * 5)         # inner-loop updates per outer iteration
    ITER_0          = 512                               # base period for cosine annealing warm restarts

    print(
        f"\n[SMAC] seed={seed}  {time.strftime('%H:%M:%S')}\n"
        f"  f_NN  : neurons={NUM_NEURON_f}, resblocks={NUM_RESBLOCKS_f}, lr={LR_f:.2e}\n"
        f"  map_T : neurons={NUM_NEURON_T}, resblocks={NUM_RESBLOCKS_T}, lr={LR_T:.2e}\n"
        f"  ITERS={ITERS}, batch={BATCH_SIZE}, K_in={K_in}, iter_0={ITER_0}",
        flush=True
    )

    try:
        w2     = 0.0
        p_true = int(1e3)  # subsample size for W2 estimation
        for sim_idx in range(AVG_SIM):
            sim_start = time.time()
            f_net = f_NN(INPUT_DIM, NUM_NEURON_f, num_resblocks=NUM_RESBLOCKS_f).to(local_device)
            MAP_T = map_NN(INPUT_DIM, NUM_NEURON_T, num_resblocks=NUM_RESBLOCKS_T).to(local_device)
            MAP_T.apply(init_weights)
            f_net.apply(init_weights)

            train(f_net, MAP_T, x_data, y_data, ITERS, LR_f, LR_T, BATCH_SIZE, lamda, lamda, K_in, ITER_0)

            # Push prior samples through the learned map at y=1 to approximate the posterior
            y_true_OT     = torch.ones_like(x_data, device=local_device)
            x_transported = MAP_T(x_data, y_true_OT).detach().cpu().numpy()

            # Compute W2 per dimension and sum (each marginal is 1D)
            sim_w2 = 0.0
            for j in range(d):
                sim_w2 += w2_distance(X_true[:p_true, j:j+1], x_transported[:p_true, j:j+1])
            w2 += sim_w2

            elapsed = time.time() - sim_start
            print(f"[SMAC] seed={seed}  sim {sim_idx + 1}/{AVG_SIM} done  "
                  f"W2={sim_w2:.4f}  time={elapsed:.1f}s  "
                  f"{time.strftime('%H:%M:%S')}", flush=True)

        loss = w2 / AVG_SIM
    except Exception as e:
        print(f"[SMAC] Trial failed: {e}")
        loss = 1e6  # penalise failed runs
    finally:
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"[SMAC] config={dict(config)}  W2={loss:.6f}")
    return loss


# ----------------------------------------------------------
#  Run SMAC optimisation
# ----------------------------------------------------------
if __name__ == "__main__":
    # Build and save shared data once — worker processes load from disk
    if not os.path.exists(_SHARED_DATA_FILE):
        print("Building SIR reference posterior and training data...")
        rng    = np.random.default_rng(0)
        X_true = np.zeros((N_true, d))
        for j in range(d):
            x_SIR = np.random.multivariate_normal(np.zeros(1), np.eye(1), N_true).T
            W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
            W = np.exp(-(W - np.min(W)))
            W = W / np.sum(W)
            index = rng.choice(np.arange(N_true), N_true, p=W)
            X_true[:, j] = x_SIR[:, index].reshape(N_true)

        dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
        dist_obs    = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))
        x_data = dist_normal.sample((N,))
        y_data = h(x_data) + dist_obs.sample((N,))

        np.savez(_SHARED_DATA_FILE,
                 X_true=X_true,
                 x_data=x_data.numpy(),
                 y_data=y_data.numpy())
        print(f"Saved shared data to {_SHARED_DATA_FILE}")
    else:
        print(f"Shared data found at {_SHARED_DATA_FILE}, skipping rebuild.")

    scenario = Scenario(
        configspace   = cs,
        name          = f"OTF_static_bimodal_d{d}",
        deterministic = True,
        n_trials      = 500,
        n_workers     = 3,
        seed          = 42,
    )

    smac = HPO(
        scenario        = scenario,
        target_function = target_fun,
        overwrite       = False,
    )

    print(f"\n=== Starting SMAC optimisation for OTF (static bimodal, d={d}) ===\n")
    print(f"Number of CPU cores: {os.cpu_count()}")
    print(f"Using device: {device}")
    print(f"Using n_workers={scenario.n_workers} for parallel evaluation.\n")
    try:
        incumbent       = smac.optimize()
        validation_loss = target_fun(incumbent, seed=42)
    finally:
        try:
            smac._runner.close()
        except Exception:
            pass

    print("\n=== Best configuration found ===")
    print(incumbent)
    print(f"Validation W2 loss: {validation_loss:.6f}")

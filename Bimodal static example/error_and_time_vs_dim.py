"""
@author: Mohammad Al-Jarrah
"""

import subprocess
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import ot
from torch.func import vmap, jacrev
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.distributions.multivariate_normal import MultivariateNormal
from param import get_param

# --- Plot settings ---
plt.close('all')
plt.rc('font', size=13)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontsize = 19

# --- Reproducibility ---
seed = np.random.randint(0, 10000)
print(f"Random seed: {seed}")
torch.manual_seed(seed)
np.random.seed(seed)


def get_free_gpu():
    """Return the GPU index with the lowest current memory usage."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    memory_used = [int(x) for x in result.stdout.strip().split("\n")]
    return memory_used.index(min(memory_used))


# --- Device selection ---
if torch.cuda.is_available():
    gpu_id = get_free_gpu()
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.manual_seed(seed)
else:
    device = torch.device("cpu")
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

    for i in range(iterations):
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

        if (i + 1) % 1024 == 0 or (i + 1) == iterations:
            with torch.no_grad():
                f_of_y     = f(X_Train, Y_Train)
                map_T      = T(X_Train, Y_Train_shuffled)
                f_of_map_T = f(map_T, Y_Train_shuffled)
                loss = (f_of_y.mean() - f_of_map_T.mean()
                        + 0.5 * ((X_Train - map_T) ** 2).sum(axis=1).mean())
                print(f"  iter {i+1:>6}/{iterations}  loss = {loss.item():.5f}")

        scheduler_f.step()
        scheduler_T.step()


def h(x):
    """Forward model: y = 0.5 * x^2, applied element-wise."""
    return 0.5 * x * x


def w2_distance(x, y):
    """Compute the 2-Wasserstein distance between two empirical distributions using exact OT."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.sqrt(ot.emd2_1d(x, y, metric='sqeuclidean'))


# --- Fixed settings ---
N       = 5000            # number of training samples per run
AVG_SIM = 10              # number of independent runs to average W2 over
sigma   = np.sqrt(1e-2)   # observation noise standard deviation

# Dimensions must match tuned parameters in param.py
D      = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50]  # state/observation dimensions to sweep
Lambda = [0, 0.01, 0.1]                          # regularization strengths to compare

# Compute SIR reference samples for W2 evaluation (ground truth posterior samples)
rng    = np.random.default_rng(0)
y_true = 1         # observed value at which to evaluate the posterior
N_true = int(1e5)  # number of reference samples for ground truth posterior
X_true = np.zeros((N_true, max(D)))
for j in range(max(D)):
    x_SIR = np.random.multivariate_normal(np.zeros(1), np.eye(1), N_true).T
    W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
    W = np.exp(-(W - np.min(W)))
    W = W / np.sum(W)
    index = rng.choice(np.arange(N_true), N_true, p=W)
    X_true[:, j] = x_SIR[:, index].reshape(N_true)

# Storage for W2 distances and timing
distance_ot = {str(l): [] for l in Lambda}
x_otf       = {str(l): {} for l in Lambda}
time_save   = {str(l): [] for l in Lambda}

# --- OTF training loop over dimensions and regularization weights ---
for dim in D:
    d  = dim
    dy = dim
    INPUT_DIM = [d, dy]

    # Load SMAC-tuned hyperparameters for this dimension
    NUM_NEURON_f, NUM_NEURON_T, NUM_RESBLOCKS_f, NUM_RESBLOCKS_T, \
    BATCH_SIZE, ITERS, LR_f, LR_T, K_in, ITER_0 = get_param()

    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist        = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))

    x = dist_normal.sample((N,)).to(device)
    y = h(x) + dist.sample((N,)).to(device)

    for lamda in Lambda:
        print('dim = ', dim, ', lambda = ', lamda)
        Delta_T = lamda
        Delta_f = lamda

        w2         = 0
        track_time = 0
        for k in range(AVG_SIM):
            f_net = f_NN(INPUT_DIM, NUM_NEURON_f, num_resblocks=NUM_RESBLOCKS_f).to(device)
            MAP_T = map_NN(INPUT_DIM, NUM_NEURON_T, num_resblocks=NUM_RESBLOCKS_T).to(device)
            MAP_T.apply(init_weights)
            f_net.apply(init_weights)

            start_time = time.time()
            train(f_net, MAP_T, x, y, ITERS, LR_f, LR_T, BATCH_SIZE, Delta_T, Delta_f, K_in, ITER_0)

            x_transported = MAP_T(x, torch.ones_like(x)).detach().cpu().numpy()
            track_time += (time.time() - start_time)
            print("--- OT time : %s seconds ---" % (time.time() - start_time))

            # Compute W2 as sum of marginal W2 distances (matches smac evaluation)
            p_true = int(1e3)
            sim_w2 = 0.0
            for j in range(d):
                sim_w2 += w2_distance(X_true[:p_true, j], x_transported[:p_true, j])
            w2 += sim_w2 / d

        distance_ot[str(lamda)].append(w2 / AVG_SIM)
        x_otf[str(lamda)][str(dim) + '_' + str(k)] = x_transported
        time_save[str(lamda)].append(track_time / AVG_SIM)

# --- SIR and EnKF baselines ---
X_SIR  = {}
X_EnKF = {}
distance_sir        = []
distance_enkf       = []
y_save              = {}
time_save_baselines = {'sir': [], 'enkf': []}

for dim in D:
    print("dim = : ", dim)
    d  = dim
    dy = dim

    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist        = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))

    x = dist_normal.sample((N,))
    y = h(x) + dist.sample((N,))
    y_save[str(dim)] = y

    w2_sir  = 0
    w2_enkf = 0
    track_time_sir  = 0
    track_time_enkf = 0

    for k in range(AVG_SIM):
        # SIR: sequential importance resampling
        start_time = time.time()
        x_SIR = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), N).T
        W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
        W = np.exp(-(W - np.min(W)))
        W = W / np.sum(W)
        index = rng.choice(np.arange(N), N, p=W)
        X_SIR[str(dim) + '_' + str(k)] = x_SIR[:, index].T
        track_time_sir += (time.time() - start_time)
        print("--- SIR time : %s seconds ---" % (time.time() - start_time))

        # EnKF: ensemble Kalman filter update
        x_hatEnKF = x.detach().numpy()
        y_hatEnKF = y.detach().numpy()
        start_time = time.time()
        X_hat  = x_hatEnKF.mean(axis=0, keepdims=True)
        Y_hat  = y_hatEnKF.mean(axis=0, keepdims=True)
        a_cov  = x_hatEnKF - X_hat
        b_cov  = y_hatEnKF - Y_hat
        C_xy   = (a_cov.T @ b_cov) / N
        C_yy   = (b_cov.T @ b_cov) / N
        K_EnKF = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-4)
        X_EnKF[str(dim) + '_' + str(k)] = x_hatEnKF + (y_true - y_hatEnKF) @ K_EnKF.T
        track_time_enkf += (time.time() - start_time)
        print("--- EnKF time : %s seconds ---" % (time.time() - start_time))

        # Compute W2 as sum of marginal W2 distances
        p_true  = int(1e3)
        sir_w2  = 0.0
        enkf_w2 = 0.0
        for j in range(dim):
            sir_w2  += w2_distance(X_true[:p_true, j], X_SIR[str(dim)  + '_' + str(k)][:p_true, j])
            enkf_w2 += w2_distance(X_true[:p_true, j], X_EnKF[str(dim) + '_' + str(k)][:p_true, j])
        w2_sir  += sir_w2  / dim
        w2_enkf += enkf_w2 / dim

    distance_sir.append(w2_sir / AVG_SIM)
    distance_enkf.append(w2_enkf / AVG_SIM)
    time_save_baselines['sir'].append(track_time_sir   / AVG_SIM)
    time_save_baselines['enkf'].append(track_time_enkf / AVG_SIM)

lambda_styles = {
    Lambda[0]: {'color': 'blue',  'ls': '--',  'marker': 'v'},
    Lambda[1]: {'color': 'green', 'ls': '-.',  'marker': 's'},
    Lambda[2]: {'color': 'red',   'ls': '-.',  'marker': 'o'},
}

# --- Plot W2 vs dimension and computational time vs dimension ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: W2 distance vs dimension
ax = axes[0]
for lamda in Lambda:
    s = lambda_styles[lamda]
    ax.semilogy(D, distance_ot[str(lamda)], marker=s['marker'], linestyle=s['ls'],
                color=s['color'], label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2)
ax.semilogy(D, distance_enkf, color="C1", marker='D', linestyle=':', label=r"EnKF", lw=2.5)
ax.semilogy(D, distance_sir,  color="C2", marker='^', linestyle=':', label=r"SIR",  lw=2.5)
ax.set_xlabel(r'$dim$', fontsize=fontsize)
ax.set_ylabel(r'$\mathrm{AA\text{-}SW}_2$', fontsize=fontsize)
ax.legend(fontsize=fontsize)

# Subplot 2: computational time vs dimension
ax = axes[1]
for lamda in Lambda:
    s = lambda_styles[lamda]
    ax.semilogy(D, time_save[str(lamda)], marker=s['marker'], linestyle=s['ls'],
                color=s['color'], label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2)
ax.semilogy(D, time_save_baselines['enkf'], color="C1", marker='D', linestyle=':', label=r"EnKF", lw=2.5)
ax.semilogy(D, time_save_baselines['sir'],  color="C2", marker='^', linestyle=':', label=r"SIR",  lw=2.5)
ax.set_xlabel(r'$dim$', fontsize=fontsize)
ax.set_ylabel(r'computational time', fontsize=fontsize)
ax.legend(fontsize=fontsize)

plt.tight_layout()
plt.savefig('error_and_time_vs_dim.pdf', bbox_inches='tight')
plt.show()

# --- Save all data needed to regenerate plots and particles ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_data = {
    # Metadata
    'seed':    seed,
    'D':       D,
    'Lambda':  Lambda,
    'N':       N,
    'AVG_SIM': AVG_SIM,
    'sigma':   sigma,
    'y_true':  y_true,
    # Ground truth (SIR reference)
    'X_true': X_true,
    # OTF results
    'distance_ot':   distance_ot,
    'x_otf':         x_otf,
    'time_save_otf': time_save,
    # Baseline particles
    'X_SIR':   X_SIR,
    'X_EnKF':  X_EnKF,
    # Baseline W2 distances
    'distance_sir':  distance_sir,
    'distance_enkf': distance_enkf,
    # Baseline timings
    'time_save_baselines': time_save_baselines,
    # Observations used for baselines
    'y_save': y_save,
}
save_path = f'error_and_time_vs_dim.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f"Data saved to {save_path}")
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

    for i in range(iterations):
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

        if (i + 1) % 1024 == 0 or (i + 1) == iterations:
            with torch.no_grad():
                f_of_y     = f(X_Train, Y_Train)
                map_T      = T(X_Train, Y_Train_shuffled)
                f_of_map_T = f(map_T, Y_Train_shuffled)
                loss = (f_of_y.mean() - f_of_map_T.mean()
                        + 0.5 * ((X_Train - map_T) ** 2).sum(axis=1).mean())
                print(f"  iter {i+1:>6}/{iterations}  loss = {loss.item():.5f}")

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
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(x, y, metric='sqeuclidean')
    return np.sqrt(ot.emd2(a, b, M))


# --- Fixed settings ---
N       = 5000            # number of training samples per run
AVG_SIM = 10              # number of independent runs to average W2 over
sigma   = np.sqrt(1e-2)   # observation noise standard deviation

# Dimensions must match tuned parameters in param.py
D      = [2, 4, 6, 8, 10, 15, 20, 30, 40, 50]  # state/observation dimensions to sweep
Lambda = [0, 0.01, 0.1]                          # regularization strengths to compare

# Compute SIR reference samples for W2 evaluation (ground truth posterior samples)
rng    = np.random.default_rng(0)
y_true = 1         # observed value at which to evaluate the posterior
N_true = int(1e5)  # number of reference samples for ground truth posterior
X_true = np.zeros((N_true, max(D)))
for j in range(max(D)):
    x_SIR = np.random.multivariate_normal(np.zeros(1), np.eye(1), N_true).T
    W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
    W = np.exp(-(W - np.min(W)))
    W = W / np.sum(W)
    index = rng.choice(np.arange(N_true), N_true, p=W)
    X_true[:, j] = x_SIR[:, index].reshape(N_true)

# Storage for W2 distances and timing
distance_ot = {str(l): [] for l in Lambda}
x_otf       = {str(l): {} for l in Lambda}
time_save   = {str(l): [] for l in Lambda}

# --- OTF training loop over dimensions and regularization weights ---
for dim in D:
    d  = dim
    dy = dim
    INPUT_DIM = [d, dy]

    # Load SMAC-tuned hyperparameters for this dimension
    NUM_NEURON_f, NUM_NEURON_T, NUM_RESBLOCKS_f, NUM_RESBLOCKS_T, \
    BATCH_SIZE, ITERS, LR_f, LR_T, K_in, ITER_0 = get_param()

    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist        = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))

    x = dist_normal.sample((N,)).to(device)
    y = h(x) + dist.sample((N,)).to(device)

    for lamda in Lambda:
        print('dim = ', dim, ', lambda = ', lamda)
        Delta_T = lamda
        Delta_f = lamda

        w2         = 0
        track_time = 0
        for k in range(AVG_SIM):
            f_net = f_NN(INPUT_DIM, NUM_NEURON_f, num_resblocks=NUM_RESBLOCKS_f).to(device)
            MAP_T = map_NN(INPUT_DIM, NUM_NEURON_T, num_resblocks=NUM_RESBLOCKS_T).to(device)
            MAP_T.apply(init_weights)
            f_net.apply(init_weights)

            start_time = time.time()
            train(f_net, MAP_T, x, y, ITERS, LR_f, LR_T, BATCH_SIZE, Delta_T, Delta_f, K_in, ITER_0)

            x_transported = MAP_T(x, torch.ones_like(x)).detach().cpu().numpy()
            track_time += (time.time() - start_time)
            print("--- OT time : %s seconds ---" % (time.time() - start_time))

            # Compute W2 as sum of marginal W2 distances (matches smac evaluation)
            p_true = int(1e3)
            sim_w2 = 0.0
            for j in range(d):
                sim_w2 += w2_distance(X_true[:p_true, j:j+1], x_transported[:p_true, j:j+1])
            w2 += sim_w2 / d

        distance_ot[str(lamda)].append(w2 / AVG_SIM)
        x_otf[str(lamda)][str(dim) + '_' + str(k)] = x_transported
        time_save[str(lamda)].append(track_time / AVG_SIM)

# --- SIR and EnKF baselines ---
X_SIR  = {}
X_EnKF = {}
distance_sir        = []
distance_enkf       = []
y_save              = {}
time_save_baselines = {'sir': [], 'enkf': []}

for dim in D:
    print("dim = : ", dim)
    d  = dim
    dy = dim

    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist        = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))

    x = dist_normal.sample((N,))
    y = h(x) + dist.sample((N,))
    y_save[str(dim)] = y

    w2_sir  = 0
    w2_enkf = 0
    track_time_sir  = 0
    track_time_enkf = 0

    for k in range(AVG_SIM):
        # SIR: sequential importance resampling
        start_time = time.time()
        x_SIR = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), N).T
        W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
        W = np.exp(-(W - np.min(W)))
        W = W / np.sum(W)
        index = rng.choice(np.arange(N), N, p=W)
        X_SIR[str(dim) + '_' + str(k)] = x_SIR[:, index].T
        track_time_sir += (time.time() - start_time)
        print("--- SIR time : %s seconds ---" % (time.time() - start_time))

        # EnKF: ensemble Kalman filter update
        x_hatEnKF = x.detach().numpy()
        y_hatEnKF = y.detach().numpy()
        start_time = time.time()
        X_hat  = x_hatEnKF.mean(axis=0, keepdims=True)
        Y_hat  = y_hatEnKF.mean(axis=0, keepdims=True)
        a_cov  = x_hatEnKF - X_hat
        b_cov  = y_hatEnKF - Y_hat
        C_xy   = (a_cov.T @ b_cov) / N
        C_yy   = (b_cov.T @ b_cov) / N
        K_EnKF = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-4)
        X_EnKF[str(dim) + '_' + str(k)] = x_hatEnKF + (y_true - y_hatEnKF) @ K_EnKF.T
        track_time_enkf += (time.time() - start_time)
        print("--- EnKF time : %s seconds ---" % (time.time() - start_time))

        # Compute W2 as sum of marginal W2 distances
        p_true  = int(1e3)
        sir_w2  = 0.0
        enkf_w2 = 0.0
        for j in range(dim):
            sir_w2  += w2_distance(X_true[:p_true, j:j+1], X_SIR[str(dim)  + '_' + str(k)][:p_true, j:j+1])
            enkf_w2 += w2_distance(X_true[:p_true, j:j+1], X_EnKF[str(dim) + '_' + str(k)][:p_true, j:j+1])
        w2_sir  += sir_w2  / dim
        w2_enkf += enkf_w2 / dim

    distance_sir.append(w2_sir / AVG_SIM)
    distance_enkf.append(w2_enkf / AVG_SIM)
    time_save_baselines['sir'].append(track_time_sir   / AVG_SIM)
    time_save_baselines['enkf'].append(track_time_enkf / AVG_SIM)

lambda_styles = {
    Lambda[0]: {'color': 'blue',  'ls': '--',  'marker': 'v'},
    Lambda[1]: {'color': 'green', 'ls': '-.',  'marker': 's'},
    Lambda[2]: {'color': 'red',   'ls': '-.',  'marker': 'o'},
}

# --- Plot W2 vs dimension and computational time vs dimension ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: W2 distance vs dimension
ax = axes[0]
for lamda in Lambda:
    s = lambda_styles[lamda]
    ax.semilogy(D, distance_ot[str(lamda)], marker=s['marker'], linestyle=s['ls'],
                color=s['color'], label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2)
ax.semilogy(D, distance_enkf, color="C1", marker='D', linestyle=':', label=r"EnKF", lw=2.5)
ax.semilogy(D, distance_sir,  color="C2", marker='^', linestyle=':', label=r"SIR",  lw=2.5)
ax.set_xlabel(r'$dim$', fontsize=fontsize)
ax.set_ylabel(r'$\mathrm{AA\text{-}SW}_2$', fontsize=fontsize)
ax.legend(fontsize=fontsize)

# Subplot 2: computational time vs dimension
ax = axes[1]
for lamda in Lambda:
    s = lambda_styles[lamda]
    ax.semilogy(D, time_save[str(lamda)], marker=s['marker'], linestyle=s['ls'],
                color=s['color'], label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2)
ax.semilogy(D, time_save_baselines['enkf'], color="C1", marker='D', linestyle=':', label=r"EnKF", lw=2.5)
ax.semilogy(D, time_save_baselines['sir'],  color="C2", marker='^', linestyle=':', label=r"SIR",  lw=2.5)
ax.set_xlabel(r'$dim$', fontsize=fontsize)
ax.set_ylabel(r'computational time', fontsize=fontsize)
ax.legend(fontsize=fontsize)

plt.tight_layout()
plt.savefig('error_and_time_vs_dim.pdf', bbox_inches='tight')
plt.show()

# --- Save all data needed to regenerate plots and particles ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_data = {
    # Metadata
    'seed':    seed,
    'D':       D,
    'Lambda':  Lambda,
    'N':       N,
    'AVG_SIM': AVG_SIM,
    'sigma':   sigma,
    'y_true':  y_true,
    # Ground truth (SIR reference)
    'X_true': X_true,
    # OTF results
    'distance_ot':   distance_ot,
    'x_otf':         x_otf,
    'time_save_otf': time_save,
    # Baseline particles
    'X_SIR':   X_SIR,
    'X_EnKF':  X_EnKF,
    # Baseline W2 distances
    'distance_sir':  distance_sir,
    'distance_enkf': distance_enkf,
    # Baseline timings
    'time_save_baselines': time_save_baselines,
    # Observations used for baselines
    'y_save': y_save,
}
save_path = f'error_and_time_vs_dim.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(save_data, f)
print(f"Data saved to {save_path}")

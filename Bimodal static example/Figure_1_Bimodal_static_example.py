"""
@author: Mohammad Al-Jarrah
"""

import os
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from torch.func import vmap, jacrev
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# --- Plot settings ---
plt.close('all')
plt.rc('font', size=13)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontsize = 19

# --- Reproducibility ---
seed = np.random.randint(0, 10000)
print(f"Random seed: {seed}")
torch.manual_seed(seed=seed)
np.random.seed(seed=seed)


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
        return self.activation(out + identity)


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
    if isinstance(m, nn.Linear) and m.bias is not None:
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
            map_T2     = T(X_train2, Y_shuffled)
            reg        = nn.functional.elu(
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

            x_batch = X_train[:K_hessian]
            y_batch = Y_train[:K_hessian]
            laplacian = vmap(hess_diag_norm)(x_batch, y_batch).sum()
            reg2 = nn.functional.elu(laplacian, alpha=0.01) / K_hessian

        loss_f = -f_of_y.mean() + f_of_map_T.mean() + delta_f * reg2
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        if (i + 1) == iterations or (i + 1) % 512 == 0:
            with torch.no_grad():
                f_of_y_full     = f(X_Train, Y_Train)
                map_T_full      = T(X_Train, Y_Train_shuffled)
                f_of_map_T_full = f(map_T_full, Y_Train_shuffled)
                loss = (f_of_y_full.mean() - f_of_map_T_full.mean()
                        + 0.5 * ((X_Train - map_T_full) ** 2).sum(axis=1).mean())
                print(f"  iter {i+1:>6}/{iterations}  loss = {loss.item():.5f}")

        scheduler_f.step()
        scheduler_T.step()


def h_1D(x):
    """Forward model: y = 0.5 * x^2."""
    return 0.5 * x * x


# --- Problem setup ---
d     = 1               # state dimension
dy    = 1               # observation dimension
N     = 1000            # number of training samples
sigma = np.sqrt(1e-2)   # observation noise standard deviation

x = torch.randn((N, d), device=device)
y = h_1D(x) + sigma * torch.randn((N, dy), device=device)

# --- Training hyperparameters ---
ITERS         = int(512 * 7)  # total number of training iterations
LR_f          = 2e-4          # learning rate for the potential network f
LR_T          = 8e-4          # learning rate for the transport map network T
INPUT_DIM     = [d, dy]       # input dimensions for both networks: [state dim, observation dim]
NUM_NEURON_f  = 32 * 9        # hidden layer width of f
NUM_NEURON_T  = 32 * 9        # hidden layer width of T
NUM_RESBLOCKS = 1             # number of residual blocks in each network
BATCH_SIZE    = 32 * 7        # mini-batch size per iteration
K_in          = 10            # number of inner-loop updates for T per outer iteration
ITER_0        = 512           # base period for cosine annealing warm restarts

# --- Exact posterior p(x|y=1) on a grid ---
y_true = 1
xx     = np.linspace(-3, 3, 1000)
dx     = 6. / 1000
px     = np.exp(-xx * xx / 2)
px     = px / np.sum(px * dx)
pyx    = np.exp(-(y_true - h_1D(xx)) ** 2 / (2 * sigma * sigma))
pxy    = px * pyx
pxy    = pxy / np.sum(pxy * dx)

# Compute exact CDF-based transport map via inverse CDF
sum_matrix    = np.tril(np.ones((px.size, px.size)))
F_px          = sum_matrix @ (pxy * dx)
F_pxy         = sum_matrix @ (px * dx)
F_inv_of_F_px = np.interp(F_px, F_pxy, xx)
f_x_y_1       = sum_matrix @ (F_inv_of_F_px * dx)
T_exact        = np.interp(F_pxy, F_px, xx)

# --- Train for each regularization strength lambda ---
Lambda  = [0, 0.01, 0.1]
results = {}

for lamda in Lambda:
    Delta_T = lamda
    Delta_f = lamda

    start_time = time.time()

    f_net = f_NN(INPUT_DIM, NUM_NEURON_f, num_resblocks=NUM_RESBLOCKS).to(device)
    MAP_T = map_NN(INPUT_DIM, NUM_NEURON_T, num_resblocks=NUM_RESBLOCKS).to(device)
    MAP_T.apply(init_weights)
    f_net.apply(init_weights)

    train(f_net, MAP_T, x, y, ITERS, LR_f, LR_T, BATCH_SIZE, Delta_T, Delta_f, K_in, ITER_0)

    # Push prior samples through the learned map at y=1 to approximate the posterior
    x_transported = torch.randn((N * 10, d), device=device)
    y_obs = torch.ones_like(x_transported)
    x_transported = MAP_T(x_transported, y_obs).detach().cpu().numpy()

    print("--- OT time : %s seconds ---" % (time.time() - start_time))

    y_plot    = torch.ones(1000, 1, device=device)
    xx_tensor = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=device)
    f_plot    = f_net(xx_tensor, y_plot).detach().cpu().numpy()[:, 0]
    T_plot    = MAP_T(xx_tensor, y_plot).detach().cpu().numpy()[:, 0]

    results[lamda] = {'x_transported': x_transported, 'f_plot': f_plot, 'T_plot': T_plot}

#%%
# --- Plotting ---
bw       = 1.0 / 5  # KDE bandwidth multiplier: < 1 sharper, > 1 smoother

plt.figure(figsize=(24, 8))

lambda_styles = {
    Lambda[0]: ('blue',  '--'),
    Lambda[1]: ('green', '-.'),
    Lambda[2]: ('red',   '-.'),
}

# Subplot 1: transported density vs exact posterior
plt.subplot(1, 3, 1)
plt.plot(xx, pxy, color='k', label=r"$P_{U|Y=1}$", lw=2)
for lamda, res in results.items():
    color, ls = lambda_styles[lamda]
    plt.hist(res['x_transported'][:, 0], bins=50, density=True, alpha=0.25, color=color)
    sns.kdeplot(data=res['x_transported'][:, 0], color=color, linestyle=ls,
                label=rf'$OT_{{(\lambda={lamda})}}$', lw=2, bw_adjust=bw)
plt.xlabel('U', fontsize=fontsize)
plt.ylabel('Density', fontsize=fontsize)
plt.legend(loc=0, fontsize=fontsize)

# Subplot 2: learned Kantorovich potential 0.5|u|^2 - f(u, y=1)
plt.subplot(1, 3, 2)
plt.plot(xx, f_x_y_1 - f_x_y_1.mean(), 'k-', lw=2)
for lamda, res in results.items():
    color, ls = lambda_styles[lamda]
    potential = 0.5 * xx * xx - res['f_plot']
    potential -= potential.mean()
    plt.plot(xx, potential, color=color, linestyle=ls, lw=2)
plt.xlabel('U', fontsize=fontsize)
plt.ylabel(r'$0.5\|U\|^2 - \phi(Y=1,U)$', fontsize=fontsize)

# Subplot 3: learned vs exact transport map T(u, y=1)
plt.subplot(1, 3, 3)
plt.plot(xx[:-1], T_exact[:-1], 'k-', lw=2)
for lamda, res in results.items():
    color, ls = lambda_styles[lamda]
    plt.plot(xx, res['T_plot'], color=color, linestyle=ls, lw=2)
plt.xlabel('U', fontsize=fontsize)
plt.ylabel(r"$T(Y=1,U)$", fontsize=fontsize)

# --- Save figure ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.dirname(os.path.abspath(__file__))  # same folder as this script
plt.savefig(os.path.join(save_dir, f'Figure_1_Bimodal_static_example_{timestamp}.pdf'), bbox_inches='tight')
plt.show()

"""
@author: Mohammad Al-Jarrah
"""


import torch
import numpy as np
import time
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.close('all')
plt.rc('font', size=13)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

torch.manual_seed(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for reproducibility
print(f"Using device: {device}")


# f(x, y) — scalar output, 2-block residual network with ELU
class f_NN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(f_NN, self).__init__()
        self.layer_input = nn.Linear(input_dim[0] + input_dim[1], hidden_dim, bias=False)
        self.layer11 = nn.Linear(hidden_dim, hidden_dim)
        self.layer12 = nn.Linear(hidden_dim, hidden_dim)
        self.layer21 = nn.Linear(hidden_dim, hidden_dim)
        self.layer22 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, 1, bias=False)
        self.act = nn.ELU()

    def forward(self, x, y):
        X = self.layer_input(torch.concat((x, y), dim=1))
        X = self.act(self.layer12(self.act(self.layer11(X))) + X)
        return self.layer_out(self.act(self.layer22(self.act(self.layer21(X))) + X))


# Transport map T(x, y) — maps prior samples to posterior given observation y, ReLU residual network
class map_NN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(map_NN, self).__init__()
        self.layer_input = nn.Linear(input_dim[0] + input_dim[1], hidden_dim, bias=False)
        self.layer11 = nn.Linear(hidden_dim, hidden_dim)
        self.layer12 = nn.Linear(hidden_dim, hidden_dim)
        self.layer21 = nn.Linear(hidden_dim, hidden_dim)
        self.layer22 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, input_dim[0], bias=False)
        self.act = nn.ReLU()

    def forward(self, x, y):
        X = self.layer_input(torch.concat((x, y), dim=1))
        X = self.act(self.layer12(self.act(self.layer11(X))) + X)
        return self.layer_out(self.act(self.layer22(self.act(self.layer21(X))) + X))


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def train(f, T, X_Train, Y_Train, iterations, learning_rate, batch_size, delta_T, delta_f):
    f.train()
    T.train()
    optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate)
    optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate)
    scheduler_f = ExponentialLR(optimizer_f, gamma=0.999)
    scheduler_T = ExponentialLR(optimizer_T, gamma=0.999)

    inner_iterations = 10
    Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)

    for i in range(iterations):
        idx = torch.randperm(X_Train.shape[0])[:batch_size]
        X_train = X_Train[idx].clone().detach()
        Y_train = Y_Train[idx].clone().detach()
        Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)

        idx2 = torch.randperm(X_Train.shape[0])[:batch_size]
        X_train2 = X_Train[idx2].clone().detach()

        # Inner loop: update transport map T while holding f fixed
        for _ in range(inner_iterations):
            map_T = T(X_train, Y_shuffled)
            f_of_map_T = f(map_T, Y_shuffled)
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
        f_of_y = f(X_train, Y_train)
        map_T = T(X_train, Y_shuffled)
        f_of_map_T = f(map_T, Y_shuffled)

        # Optional Hessian regularization to promote convexity of f
        reg2 = 0
        if delta_f != 0:
            laplacian = 0
            K_hessian = 4
            for kk in range(K_hessian):
                x_k = X_train[kk].view(1, d)
                y_k = Y_train[kk].view(1, dy)
                hessian = torch.autograd.functional.hessian(f, (x_k, y_k), create_graph=True)
                laplacian += torch.norm(hessian[0][0].reshape(d, d).diag())
            reg2 = nn.functional.elu(laplacian, alpha=0.01) / K_hessian

        loss_f = -f_of_y.mean() + f_of_map_T.mean() + delta_f * reg2
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        if (i + 1) == iterations or i % 1000 == 0:
            with torch.no_grad():
                f_of_y = f(X_Train, Y_Train)
                map_T = T(X_Train, Y_Train_shuffled)
                f_of_map_T = f(map_T, Y_Train_shuffled)
                loss = (f_of_y.mean() - f_of_map_T.mean()
                        + 0.5 * ((X_Train - map_T) ** 2).sum(axis=1).mean())
                print("Iteration: %d/%d, loss = %.4f" % (i + 1, iterations, loss.item()))

        scheduler_f.step()
        scheduler_T.step()


# --- Problem setup ---
d = 1    # source/state dimension
dy = 1   # observation dimension
N = 1000
sigma = np.sqrt(1e-2)

x = torch.randn((N, d), device=device)
y = 0.5 * x * x + sigma * torch.randn((N, dy), device=device)  # nonlinear observation: y = 0.5*x^2 + noise

ITERS = int(1e3 * 5)
LR = 1e-3
INPUT_DIM = [d, dy]
NUM_NEURON = 32
BATCH_SIZE = 64

def h_1D(x):
    return 0.5 * x * x

# Compute exact posterior p(x|y=1) via Bayes rule on a grid
y_true = 1
xx = np.linspace(-3, 3, 1000)
dx = 6. / 1000
px = np.exp(-xx * xx / 2)
px = px / np.sum(px * dx)
pyx = np.exp(-(y_true - h_1D(xx)) ** 2 / (2 * sigma * sigma))
pxy = px * pyx
pxy = pxy / np.sum(pxy * dx)

Lambda = [0, 0.01, 0.1]


# Compute exact CDF-based transport once — doesn't depend on lambda
sum_matrix = np.tril(np.ones((px.size, px.size)))
F_px = sum_matrix @ (pxy * dx)       # CDF of posterior p(x|y=1)
F_pxy = sum_matrix @ (px * dx)       # CDF of prior p(x)
F_inv_of_F_px = np.interp(F_px, F_pxy, xx)
f_x_y_1 = sum_matrix @ (F_inv_of_F_px * dx)
T_exact = np.interp(F_pxy, F_px, xx)

results = {}

for lamda in Lambda:
    Delta_T = lamda
    Delta_f = lamda

    start_time = time.time()

    f = f_NN(INPUT_DIM, NUM_NEURON).to(device)
    MAP_T = map_NN(INPUT_DIM, NUM_NEURON).to(device)
    MAP_T.apply(init_weights)
    f.apply(init_weights)

    train(f, MAP_T, x, y, ITERS, LR, BATCH_SIZE, Delta_T, Delta_f)

    # Push prior samples through the learned map at y=1 to approximate the posterior
    x_transported = torch.randn((N * 10, d), device=device)
    y_obs = torch.ones_like(x_transported)
    x_transported = MAP_T(x_transported, y_obs).detach().cpu().numpy()

    print("--- OT time : %s seconds ---" % (time.time() - start_time))

    y_plot = torch.ones(1000, 1, device=device)
    xx_tensor = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=device)
    f_plot = f(xx_tensor, y_plot).detach().cpu().numpy()[:, 0]
    T_plot = MAP_T(xx_tensor, y_plot).detach().cpu().numpy()[:, 0]

    results[lamda] = {'x_transported': x_transported, 'f_plot': f_plot, 'T_plot': T_plot}

#%%
bw = 1.0/5  # KDE bandwidth multiplier: < 1 sharper, > 1 smoother
fontsize = 18
# --- Plotting ---
plt.figure(figsize=(25, 6))

# Subplot 1: transported density vs exact posterior
plt.subplot(1, 3, 1)
plt.plot(xx, pxy, color='k', label=r"$P_{U|Y=1}$", lw=2)
for lamda, res in results.items():
    if lamda == 0:
        sns.kdeplot(data=res['x_transported'][:, 0], label=r'$OT_{(\lambda=0)}$', color="blue", linestyle='--', lw=2, bw_adjust=bw)
    elif lamda == 0.01:
        sns.kdeplot(data=res['x_transported'][:, 0], color='green', label=r'$OT_{(\lambda=0.01)}$', linestyle='-.', lw=2, bw_adjust=bw)
    else:
        sns.kdeplot(data=res['x_transported'][:, 0], color='red', label=r'$OT_{(\lambda=0.1)}$', linestyle='-.', lw=2, bw_adjust=bw)
plt.xlabel('U', fontsize=fontsize)
plt.ylabel('Density', fontsize=fontsize)
plt.legend(loc=0, fontsize=fontsize)

# Subplot 2: learned Kantorovich potential 0.5|u|^2 - f(u, y=1)
plt.subplot(1, 3, 2)
plt.plot(xx, f_x_y_1 - f_x_y_1.mean(), 'k-', lw=2)
for lamda, res in results.items():
    potential = 0.5 * xx * xx - res['f_plot']
    potential -= potential.mean()
    if lamda == 0:
        plt.plot(xx, potential, 'b--', lw=2)
    elif lamda == 0.01:
        plt.plot(xx, potential, 'g-.', lw=2)
    else:
        plt.plot(xx, potential, 'r-.', lw=2)
plt.xlabel('U', fontsize=fontsize)
plt.ylabel(r'$0.5\|U\|^2 - \phi(Y=1,U)$', fontsize=fontsize)

# Subplot 3: learned vs exact transport map T(u, y=1)
plt.subplot(1, 3, 3)
plt.plot(xx[:-1], T_exact[:-1], 'k-', lw=2)
for lamda, res in results.items():
    if lamda == 0:
        plt.plot(xx, res['T_plot'], 'b--', lw=2)
    elif lamda == 0.01:
        plt.plot(xx, res['T_plot'], 'g-.', lw=2)
    else:
        plt.plot(xx, res['T_plot'], 'r-.', lw=2)
plt.xlabel('U', fontsize=fontsize)
plt.ylabel(r"$T(Y=1,U)$", fontsize=fontsize)

plt.tight_layout()
plt.savefig('Figure_1_Bimodal_static_example.pdf', bbox_inches='tight')
plt.savefig('Figure_1_Bimodal_static_example.png', bbox_inches='tight', dpi=300)
# plt.savefig('Figure_1_Bimodal_static_example.pdf')
# plt.savefig('Figure_1_Bimodal_static_example.png', dpi=300)
plt.show()


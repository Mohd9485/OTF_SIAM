#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:14:56 2024

@author: Mohammad Al-Jarrah
"""

import torch
import numpy as np
import time
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import matplotlib
import ot
from torch.distributions.multivariate_normal import MultivariateNormal

plt.close('all')
plt.rc('font', size=13)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontsize = 16

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
        if m.bias is not None:
            m.bias.data.fill_(0.0)


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

        if (i + 1) == iterations:
            with torch.no_grad():
                f_of_y = f(X_Train, Y_Train)
                map_T = T(X_Train, Y_Train_shuffled)
                f_of_map_T = f(map_T, Y_Train_shuffled)
                loss = (f_of_y.mean() - f_of_map_T.mean()
                        + 0.5 * ((X_Train - map_T) ** 2).sum(axis=1).mean())
                print("Iteration: %d/%d, loss = %.4f" % (i + 1, iterations, loss.item()))

        scheduler_f.step()
        scheduler_T.step()


def h(x):
    return 0.5 * x * x


# --- Problem setup ---
ITERS = int(1e4 / 2)
LR = 1e-3
NUM_NEURON = int(32 * 3)
BATCH_SIZE = int(32 * 1)

NN = [100, 500, 1000, 5000, 10000, 20000, 50000]
sigma = np.sqrt(1e-2)
d = 10
dy = d
INPUT_DIM = [d, dy]
Lambda = [0, 0.01, 0.1]
AVG_SIM = 10

y_true = 1
rng = np.random.default_rng()

# Generate reference posterior samples via SIR for W2 ground truth
N_true = int(1e5)
X_true = np.zeros((N_true, d))
for j in range(d):
    print(j)
    x_SIR = np.random.multivariate_normal(np.zeros(1), np.eye(1), N_true).T
    W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
    W = np.exp(-(W - np.min(W)))
    W = W / np.sum(W)
    index = rng.choice(np.arange(N_true), N_true, p=W)
    X_true[:, j] = x_SIR[:, index].reshape(N_true)

distance_ot = {str(l): [] for l in Lambda}
x_otf = {str(l): {} for l in Lambda}

# --- OT: train map and compute W2 for each particle count and regularization weight ---
for N in NN:
    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))
    x = dist_normal.sample((N,))
    y = h(x) + dist.sample((N,))

    for lamda in Lambda:
        print('# particles = ', N, ', lambda = ', lamda)
        Delta_T = lamda
        Delta_f = lamda

        w2 = 0
        for k in range(AVG_SIM):
            start_time = time.time()
            f = f_NN(INPUT_DIM, NUM_NEURON)
            MAP_T = map_NN(INPUT_DIM, NUM_NEURON)
            train(f, MAP_T, x, y, ITERS, LR, BATCH_SIZE, Delta_T, Delta_f)

            # Push prior samples through the learned map at y=1 to approximate the posterior
            y_true_OT = torch.ones_like(x)
            x_transported = MAP_T(x, y_true_OT).detach().numpy()
            print("--- OT time : %s seconds ---" % (time.time() - start_time))

            # Compute W2 between transported samples and reference posterior
            p_true = int(1e3)
            M_ot = ot.dist(X_true[:p_true, :], x_transported[:p_true, :])
            a = np.ones(p_true) / p_true
            b = np.ones(p_true) / p_true if N > p_true else np.ones(N) / N
            w2 += np.sqrt(ot.emd2(a, b, M_ot))
            x_otf[str(lamda)][str(N) + '_' + str(k)] = x_transported

        distance_ot[str(lamda)].append(w2 / AVG_SIM)

# --- Baselines: SIR and EnKF W2 vs particle count ---
X_SIR = {}
X_EnKF = {}
distance_sir = []
distance_enkf = []

for N in NN:
    print('# particles = ', N)
    dist_normal = MultivariateNormal(torch.zeros(d), torch.eye(d))
    dist = MultivariateNormal(torch.zeros(dy), sigma * sigma * torch.eye(dy))
    x = dist_normal.sample((N,))
    y = h(x) + dist.sample((N,))

    w2_sir = 0
    w2_enkf = 0
    for k in range(AVG_SIM):
        # SIR: sequential importance resampling
        start_time = time.time()
        x_SIR = np.random.multivariate_normal(np.zeros(d), np.eye(d), N).T
        W = np.sum((y_true - h(x_SIR).T) ** 2, axis=1) / (2 * sigma * sigma)
        W = np.exp(-(W - np.min(W)))
        W = W / np.sum(W)
        index = rng.choice(np.arange(N), N, p=W)
        X_SIR[str(N) + '_' + str(k)] = x_SIR[:, index].T
        print("--- SIR time : %s seconds ---" % (time.time() - start_time))

        # EnKF: ensemble Kalman filter update
        x_hatEnKF = x.detach().numpy()
        y_hatEnKF = y.detach().numpy()
        start_time = time.time()
        X_hat = x_hatEnKF.mean(axis=0, keepdims=True)
        Y_hat = y_hatEnKF.mean(axis=0, keepdims=True)
        a_cov = x_hatEnKF - X_hat
        b_cov = y_hatEnKF - Y_hat
        C_xy = (a_cov.T @ b_cov) / N
        C_yy = (b_cov.T @ b_cov) / N
        K_EnKF = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-4)
        X_EnKF[str(N) + '_' + str(k)] = x_hatEnKF + (y_true - y_hatEnKF) @ K_EnKF.T
        print("--- EnKF time : %s seconds ---" % (time.time() - start_time))

        # Compute W2 for SIR and EnKF
        p_true = int(1e3)
        a = np.ones(p_true) / p_true
        b = np.ones(N) / N
        M_sir = ot.dist(X_true[:p_true, :d], X_SIR[str(N) + '_' + str(k)])
        w2_sir += np.sqrt(ot.emd2(a, b, M_sir))
        M_enkf = ot.dist(X_true[:p_true, :d], X_EnKF[str(N) + '_' + str(k)])
        w2_enkf += np.sqrt(ot.emd2(a, b, M_enkf))

    distance_sir.append(w2_sir / AVG_SIM)
    distance_enkf.append(w2_enkf / AVG_SIM)

# --- Plot W2 vs number of particles ---
plt.figure(figsize=(12, 6))
for lamda in x_otf.keys():
    if lamda in ('0.0', '0'):
        plt.semilogy(NN, distance_ot[str(lamda)], 'v--', color="blue", label=r"$OT_{(\lambda=0)}$", lw=2)
    elif lamda == '0.01':
        plt.semilogy(NN, distance_ot[str(lamda)], 's-.', color="green", label=r'$OT_{(\lambda=0.01)}$', lw=2)
    elif lamda == '0.1':
        plt.semilogy(NN, distance_ot[str(lamda)], 'o-.', color="red", label=r'$OT_{(\lambda=0.1)}$', lw=2)

plt.semilogy(NN, distance_enkf, 'D:', color="C4", label=r"EnKF", lw=2.5)
plt.semilogy(NN, distance_sir, '^:', color="C5", label=r"SIR", lw=2.5)
plt.xlabel('# of particles', fontsize=fontsize)
plt.ylabel(r'$W_2$', fontsize=fontsize)
plt.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig('Figure_2_W2_vs_particles.pdf', bbox_inches='tight')
plt.show()

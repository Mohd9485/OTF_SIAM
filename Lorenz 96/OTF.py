"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# --- Integrator ---

def rk4_step(f, t, x, tau):
    k1 = f(t,         x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau,   x + tau   * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)


# --- Optimal Transport Filter ---


def OTF(Y, X0_const, parameters, A, h, t, tau, Noise, rk4, delta, device):
    """
    On-the-fly optimal transport filter (OTF) for sequential state estimation.

    At each assimilation step, trains a dual pair of networks — a convex potential f
    and a transport map T — to push the forecast ensemble to the posterior via the
    Kantorovich dual formulation of optimal transport.

    Parameters
    ----------
    Y           : ndarray, shape (AVG_SIM, N, dy)  — observations across all simulations
    X0_const    : ndarray, shape (AVG_SIM, L, J)   — initial ensemble states
    parameters  : dict                             — network and training hyperparameters
    A           : callable                         — dynamics function A(t, x)
    h           : callable                         — observation operator h(x)
    t           : ndarray                          — time grid
    tau         : float                            — time step size
    Noise       : list or tuple                    — noise standard deviations [sigmma, gamma]
    rk4         : bool                             — use RK4 integrator if True, else forward Euler
    delta       : list or tuple                    — regularization weights [delta_T, delta_f]
    device      : torch.device                     — compute device (cpu/cuda/mps)

    Returns
    -------
    ndarray, shape (AVG_SIM, N, L, J) — filtered ensemble trajectories
    """

    # --- Dimensions ---
    AVG_SIM     = X0_const.shape[0]  # number of independent simulation runs
    L           = X0_const.shape[1]  # state space dimension
    SAMPLE_SIZE = X0_const.shape[2]  # ensemble / particle size

    N  = Y.shape[1]  # number of observation time steps
    dy = Y.shape[2]  # observation space dimension

    # --- Noise Parameters ---
    sigmma = Noise[0]  # std of process noise in the hidden state
    gamma  = Noise[1]  # std of observation noise

    # --- Regularization Weights ---
    delta_T = delta[0]  # monotonicity regularization weight for transport map T
    delta_f = delta[1]  # Hessian regularization weight for potential f

    # --- Network Hyperparameters ---
    NUM_NEURON_f, NUM_NEURON_T = parameters['NUM_NEURON']              # hidden layer widths for f and T networks
    INPUT_DIM                  = parameters['INPUT_DIM']               # [state_dim, obs_dim]
    BATCH_SIZE                 = parameters['BATCH_SIZE']              # mini-batch size for training
    LR_f, LR_T                 = parameters['LearningRate']            # learning rates for f and T
    ITERATION                  = parameters['ITERATION']               # initial number of training iterations per step
    Final_Number_ITERATION     = parameters['Final_Number_ITERATION']  # minimum iterations after decay
    inner_iterations           = parameters['inner_iterations']        # inner loop count for T update per outer step


    # --- Network Definitions ---

    class NeuralNet(nn.Module):
        """Convex potential network f(x, y) used in the dual OT objective."""

        def __init__(self, input_dim, hidden_dim):
            super(NeuralNet, self).__init__()
            self.input_dim  = input_dim   # [state_dim, obs_dim]
            self.hidden_dim = hidden_dim  # number of neurons per hidden layer
            self.activation = nn.ELU()

            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=True)
            self.layer11     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out   = nn.Linear(self.hidden_dim, 1, bias=True)

        def forward(self, x, y):
            X  = self.layer_input(torch.concat((x, y), dim=1))  # joint (x, y) input, (batch, L + dy) -> (batch, hidden)
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12(xy)
            xy = self.layer_out(self.activation(xy) + X)  # residual skip connection from input layer
            return xy                                      # scalar potential per sample, (batch, 1)


    class T_NeuralNet(nn.Module):
        """Transport map network T(x, y) that pushes forecast particles to the posterior."""

        def __init__(self, input_dim, hidden_dim):
            super(T_NeuralNet, self).__init__()
            self.input_dim  = input_dim   # [state_dim, obs_dim]
            self.hidden_dim = hidden_dim  # number of neurons per hidden layer
            self.activation = nn.ReLU()

            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=True)
            self.layer11     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out   = nn.Linear(self.hidden_dim, input_dim[0], bias=True)

        def forward(self, x, y):
            X  = self.layer_input(torch.concat((x, y), dim=1))  # joint (x, y) input, (batch, L + dy) -> (batch, hidden)
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12(xy)
            xy = self.layer_out(self.activation(xy) + X)  # residual connection before output
            return xy                                      # transported state per sample, (batch, L)


    # --- Weight Initialization ---

    def init_weights(m):
        """Initialize Linear layer weights with Xavier uniform and biases to 0.001."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.001)


    # --- Training Function ---

    def train(f, T, X_Train, Y_Train, iterations, lr_f, lr_T, inner_iters, ts, Ts, batch_size, k, K):
        """
        Train networks f and T for one assimilation step using the dual OT objective.

        Alternates between inner updates to T (holding f fixed) and outer updates to f.
        Applies optional monotonicity regularization on T and Hessian regularization on f.

        Parameters
        ----------
        f, T        : nn.Module  — potential network and transport map network
        X_Train     : Tensor     — forecast ensemble, shape (SAMPLE_SIZE, L)
        Y_Train     : Tensor     — forecast observations, shape (SAMPLE_SIZE, dy)
        iterations  : int        — number of outer training iterations
        lr_f, lr_T  : float      — learning rates for f and T
        inner_iters : int        — number of inner T updates per outer step
        ts          : int        — current time step index (for logging)
        Ts          : int        — total number of time steps (for logging)
        batch_size  : int        — mini-batch size
        k           : int        — current simulation index (for logging)
        K           : int        — total number of simulations (for logging)
        """
        f.train()
        T.train()
        optimizer_T = torch.optim.Adam(T.parameters(), lr=lr_T)  # optimizer for transport map T
        optimizer_f = torch.optim.Adam(f.parameters(), lr=lr_f)  # optimizer for potential f

        # Cosine annealing with warm restarts; T_0 = iterations, so each call runs a single cosine cycle
        scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=iterations, T_mult=1, eta_min=lr_f * 1e-3)
        scheduler_T = CosineAnnealingWarmRestarts(optimizer_T, T_0=iterations, T_mult=1, eta_min=lr_T * 1e-3)

        inner_iterations = inner_iters                                                     # inner loop count for T update
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)  # fixed shuffled Y for the final loss report (its randperm also advances the torch RNG)

        for i in range(iterations):
            idx  = torch.randperm(X_Train.shape[0])[:batch_size]  # random batch indices for primary batch
            idx2 = torch.randperm(X_Train.shape[0])[:batch_size]  # random batch indices for monotonicity regularization

            X_train  = X_Train[idx].clone().detach()   # primary batch of forecast states
            Y_train  = Y_Train[idx].clone().detach()   # primary batch of forecast observations
            X_train2 = X_Train[idx2].clone().detach()  # secondary batch used only for monotonicity reg

            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)  # shuffled Y breaks the (x, y) pairing: samples of the product of marginals

            # Inner loop: update T while holding f fixed
            for _ in range(inner_iterations):
                map_T      = T.forward(X_train, Y_shuffled)
                f_of_map_T = f.forward(map_T, Y_shuffled)

                reg = 0
                if delta_T != 0:
                    map_T2 = T(X_train2, Y_shuffled)
                    # Monotonicity regularization: penalize violations of (T(x2)-T(x))*(x2-x) >= 0
                    reg = nn.functional.elu(
                        ((map_T2 - map_T) * (-X_train2 + X_train)).sum(axis=1), alpha=0.01
                    ).mean()

                # T loss: maximize f(T(x)) subject to transport cost and monotonicity penalty
                loss_T = -f_of_map_T.mean() + 0.5 * ((X_train - map_T) * (X_train - map_T)).sum(axis=1).mean() + delta_T * reg

                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()

            # Outer update: compute dual OT loss and update f
            f_of_xy    = f.forward(X_train, Y_train)
            map_T      = T.forward(X_train, Y_shuffled)
            f_of_map_T = f.forward(map_T, Y_shuffled)

            reg2 = 0
            if delta_f != 0:
                K_hessian = batch_size  # number of samples used for Hessian approximation

                def f_scalar(x_flat, y_flat):
                    return f(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).squeeze()  # f on a single sample, as a scalar for jacrev

                H_fn = jacrev(jacrev(f_scalar, argnums=0), argnums=0)  # full Hessian via double-nested autodiff

                def hess_diag_norm(x_k, y_k):
                    return torch.norm(H_fn(x_k, y_k).diag())  # norm of Hessian diagonal as convexity proxy

                x_batch   = X_train[:K_hessian]                            # (K_hessian, L)
                y_batch   = Y_train[:K_hessian]                            # (K_hessian, dy)
                laplacian = vmap(hess_diag_norm)(x_batch, y_batch).sum()  # summed over the batch
                reg2      = nn.functional.elu(laplacian, alpha=0.01) / K_hessian

            # f loss: maximize Kantorovich dual E[f(x)] - E[f(T(x))] with Hessian regularization
            loss_f = -f_of_xy.mean() + f_of_map_T.mean() + delta_f * reg2

            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()

            scheduler_f.step()
            scheduler_T.step()

            # Log full-batch loss at the final iteration
            if (i + 1) == iterations:
                with torch.no_grad():
                    f_of_xy    = f.forward(X_Train, Y_Train)
                    map_T      = T.forward(X_Train, Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T, Y_Train_shuffled)
                    loss       = f_of_xy.mean() - f_of_map_T.mean() + 0.5 * ((X_Train - map_T) * (X_Train - map_T)).sum(axis=1).mean()
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" % (k + 1, K, ts, Ts - 1, i + 1, iterations, loss.item()))


    # --- Main Filter Loop ---

    start_time    = time.time()
    SAVE_all_X_OT = np.zeros((AVG_SIM, N, SAMPLE_SIZE, L))  # storage for filtered ensemble states

    for k in range(AVG_SIM):

        y = Y[k,]  # observations for simulation k, shape (N, dy)

        ITERS = ITERATION  # iteration budget, halved over time steps until reaching Final_Number_ITERATION

        # Initialize networks and apply weight initialization schemes
        convex_f = NeuralNet(INPUT_DIM, NUM_NEURON_f).to(device)
        MAP_T    = T_NeuralNet(INPUT_DIM, NUM_NEURON_T).to(device)

        convex_f.apply(init_weights)
        MAP_T.apply(init_weights)

        X0                        = X0_const[k,].T  # initial ensemble for simulation k, shape (SAMPLE_SIZE, L)
        SAVE_all_X_OT[k, 0, :, :] = X0

        # --- Assimilation Loop ---
        for i in range(N - 1):

            # Process noise for ensemble propagation
            sai_train = np.random.multivariate_normal(np.zeros(L), sigmma * sigmma * np.eye(L), SAMPLE_SIZE)  # (SAMPLE_SIZE, L)

            # Forecast step: propagate ensemble forward by one time step.
            # The dynamics A act on the flattened (L * SAMPLE_SIZE,) state (vectorized ensemble).
            if rk4:
                X1 = (rk4_step(A, t[i], (X0.T).reshape(-1), tau).reshape(L, SAMPLE_SIZE) + sai_train.T).T  # RK4 forecast with noise, shape (SAMPLE_SIZE, L)
            else:
                X1 = X0 + ((A(t[i], X0.T)).reshape(L, SAMPLE_SIZE) * tau).T + sai_train  # Euler forecast with noise

            # Ensemble predicted observations with perturbed noise
            eta_train = np.random.multivariate_normal(np.zeros(dy), gamma * gamma * np.eye(dy), SAMPLE_SIZE)  # (SAMPLE_SIZE, dy)
            Y1        = np.array(h(X1.T).T + eta_train)  # shape (SAMPLE_SIZE, dy)

            # Convert forecast ensemble to float32 tensors on device
            X1_train = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_train = torch.from_numpy(Y1).to(torch.float32).to(device)

            # Train f and T on the current forecast ensemble
            train(convex_f, MAP_T, X1_train, Y1_train, ITERS, LR_f, LR_T, inner_iterations, i + 1, N, BATCH_SIZE, k, AVG_SIM)

            # Halve iteration budget each step until the minimum is reached (i >= 0 always holds).
            # With get_params(), Final_Number_ITERATION = ITERATION = 512, so the budget never decays.
            if ITERS > Final_Number_ITERATION and i >= 0:
                ITERS = int(ITERS / 2)
                if ITERS < Final_Number_ITERATION:
                    ITERS = Final_Number_ITERATION

            # Prepare true observation at next time step, broadcast to ensemble shape
            Y1_true = torch.from_numpy(y[i + 1, :]).to(torch.float32).to(device)
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)  # forecast ensemble as inference input

            # Apply trained transport map to push forecast ensemble to posterior
            map_T = MAP_T.forward(X1_test, Y1_true * torch.ones((X1_test.shape[0], dy), device=device))

            # Store results and advance ensemble to next step
            X0                            = map_T.cpu().detach().numpy()  # posterior ensemble, (SAMPLE_SIZE, L)
            SAVE_all_X_OT[k, i + 1, :, :] = map_T.cpu().detach().numpy()

    SAVE_all_X_OT = SAVE_all_X_OT.transpose((0, 1, 3, 2))  # (AVG_SIM, N, SAMPLE_SIZE, L) -> (AVG_SIM, N, L, SAMPLE_SIZE)
    print("--- OTF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT

"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy.integrate import RK45


def OTF(Y, X0_const, parameters, A, h, t, tau, Noise, rk45, delta, device):
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
    Noise       : tuple                            — noise parameters (sigmma, gamma)
    rk45        : bool                             — use RK45 integrator if True, else forward Euler
    delta       : tuple                            — regularization weights (delta_T, delta_f)
    device      : torch.device                    — compute device (cpu/cuda/mps)

    Returns
    -------
    ndarray, shape (AVG_SIM, N, L, J) — filtered ensemble trajectories
    """

    # --- Dimensions ---
    AVG_SIM     = X0_const.shape[0]  # number of independent simulation runs
    L           = X0_const.shape[1]  # state space dimension
    SAMPLE_SIZE = X0_const.shape[2]  # ensemble / particle size

    N  = Y.shape[1]                  # number of observation time steps
    dy = Y.shape[2]                  # observation space dimension

    # --- Noise Parameters ---
    sigmma = Noise[0]  # std of process noise in the hidden state
    gamma  = Noise[1]  # std of observation noise

    T = tau * N                      # total integration time horizon

    # --- Regularization Weights ---
    delta_T = delta[0]               # monotonicity regularization weight for transport map T
    delta_f = delta[1]               # Hessian regularization weight for potential f

    # --- Network Hyperparameters ---
    normalization              = parameters['normalization']            # normalization type for training data
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
            X  = self.layer_input(torch.concat((x, y), dim=1))
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12(xy)
            xy = self.layer_out(self.activation(xy) + X)  # residual skip connection from input layer
            return xy


    class T_NeuralNet(nn.Module):
        """
        Transport map network T(x, y) that pushes forecast particles to the posterior.

        Uses an internal EnKF correction (x_kf) as a warm start, then refines it
        through learned residual layers.
        """

        def __init__(self, input_dim, hidden_dim):
            super(T_NeuralNet, self).__init__()
            self.input_dim  = input_dim   # [state_dim, obs_dim]
            self.hidden_dim = hidden_dim  # number of neurons per hidden layer
            self.activation = nn.ReLU()

            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=True)
            self.layer11     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer12     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.layer_out   = nn.Linear(self.hidden_dim, input_dim[0], bias=True)

            # Pre-built observation noise distribution for perturbed observation sampling
            self.dist = MultivariateNormal(
                torch.zeros(self.input_dim[1]).to(device),
                gamma * gamma * torch.eye(self.input_dim[1]).to(device)
            )
            self.x_kf = 0  # stores Kalman-corrected state; written on each forward pass for loss access

        def forward(self, x, y):
            y_hat     = h(x.T).T + self.dist.sample((x.shape[0],))  # perturbed predicted observations, shape (J, dy)
            self.x_kf = x + (K @ (y - y_hat).T).T                   # EnKF warm start using precomputed Kalman gain K

            X  = self.layer_input(torch.concat((self.x_kf, y), dim=1))
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12(xy)
            xy = self.layer_out(self.activation(xy) + X) + self.x_kf  # residual: correction added on top of x_kf
            return xy


    # --- Weight Initialization ---

    def init_weights(m):
        """Initialize linear weights with small normal noise and constant bias (used for T)."""
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.001)
            if m.bias is not None:
                m.bias.data.fill_(0.001)

    def init_weights_f(m):
        """Initialize linear weights with orthogonal init and constant bias (used for f)."""
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
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

        # Cosine annealing with warm restarts for smooth LR decay throughout training
        scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=iterations, T_mult=2, eta_min=lr_f * 1e-3)
        scheduler_T = CosineAnnealingWarmRestarts(optimizer_T, T_0=iterations, T_mult=2, eta_min=lr_T * 1e-3)

        inner_iterations = inner_iters                                                      # inner loop count for T update
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)   # fixed shuffled Y for final validation pass

        for i in range(iterations):
            idx  = torch.randperm(X_Train.shape[0])[:batch_size]  # random batch indices for primary batch
            idx2 = torch.randperm(X_Train.shape[0])[:batch_size]  # random batch indices for monotonicity regularization

            X_train  = X_Train[idx].clone().detach()   # primary batch of forecast states
            Y_train  = Y_Train[idx].clone().detach()   # primary batch of forecast observations
            X_train2 = X_Train[idx2].clone().detach()  # secondary batch used only for monotonicity reg

            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)  # shuffled Y to break x-y dependence

            # Inner loop: update T while holding f fixed
            for j in range(inner_iterations):
                map_T      = T.forward(X_train, Y_shuffled)
                f_of_map_T = f.forward(map_T, Y_shuffled)
                x_kf_1     = T.x_kf.detach().clone()  # Kalman state for primary batch, detached

                reg = 0
                if delta_T != 0:
                    map_T2 = T(X_train2, Y_shuffled)
                    x_kf_2 = T.x_kf.detach()
                    # Penalize violations of (T(x2)-T(x)) · (x_kf_2-x_kf_1) >= 0 (monotonicity)
                    reg = nn.functional.elu(((map_T2 - map_T) * (x_kf_1 - x_kf_2)).sum(axis=1), alpha=0.01).mean()

                # T loss: maximize f(T(x)) subject to transport cost and monotonicity penalty
                loss_T = -f_of_map_T.mean() + 0.5 * ((x_kf_1 - map_T) * (x_kf_1 - map_T)).sum(axis=1).mean() + delta_T * reg

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
                    return f(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).squeeze()

                H_fn = jacrev(jacrev(f_scalar, argnums=0), argnums=0)  # full Hessian via double-nested autodiff

                def hess_diag_norm(x_k, y_k):
                    return torch.norm(H_fn(x_k, y_k).diag())  # norm of Hessian diagonal as convexity proxy

                x_batch   = X_train[:K_hessian]
                y_batch   = Y_train[:K_hessian]
                laplacian = vmap(hess_diag_norm)(x_batch, y_batch).sum()
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
                    x_kf_val   = T.x_kf
                    f_of_map_T = f.forward(map_T, Y_Train_shuffled)
                    loss_f     = f_of_xy.mean() - f_of_map_T.mean()
                    loss       = f_of_xy.mean() - f_of_map_T.mean() + 0.5 * ((x_kf_val - map_T) * (x_kf_val - map_T)).sum(axis=1).mean()
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" % (k + 1, K, ts, Ts - 1, i + 1, iterations, loss.item()))


    # --- Normalization Utilities ---

    def Normalization(X, Type='None'):
        """Normalize data: 'MinMax' scales to [0,1], 'Mean' standardizes to zero mean and unit std."""
        if Type == 'None':
            return 0, 0, X
        elif Type == 'Mean':
            Mean_X_training_data = torch.mean(X)  # sample mean across all elements
            Std_X_training_data  = torch.std(X)   # sample std across all elements
            return Mean_X_training_data, Std_X_training_data, (X - Mean_X_training_data) / Std_X_training_data
        elif Type == 'MinMax':
            Min = torch.min(X)  # global minimum
            Max = torch.max(X)  # global maximum
            return Min, Max, (X - Min) / (Max - Min)

    def Transfer(M, S, X, Type='None'):
        """Apply normalization to test data using statistics M (mean/min) and S (std/max) from training data."""
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return (X - M) / S
        elif Type == 'MinMax':
            return (X - M) / (S - M)

    def deTransfer(M, S, X, Type='None'):
        """Invert normalization to recover data in the original scale."""
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return X * S + M
        elif Type == 'MinMax':
            return X * (S - M) + M


    # --- Main Filter Loop ---

    start_time    = time.time()
    SAVE_all_X_OT = np.zeros((AVG_SIM, N, SAMPLE_SIZE, L))  # storage for filtered ensemble states

    for k in range(AVG_SIM):

        y = Y[k,]  # observations for simulation k, shape (N, dy)

        ITERS = ITERATION  # iteration budget, halved over time steps until reaching Final_Number_ITERATION

        # Initialize networks and apply weight initialization schemes
        convex_f = NeuralNet(INPUT_DIM, NUM_NEURON_f).to(device)
        MAP_T    = T_NeuralNet(INPUT_DIM, NUM_NEURON_T).to(device)

        convex_f.apply(init_weights_f)
        MAP_T.apply(init_weights)
        torch.nn.init.orthogonal_(MAP_T.layer_input.weight)  # override input layer of T with orthogonal init

        X0 = X0_const[k,].T              # initial ensemble for simulation k, shape (SAMPLE_SIZE, L)
        X1 = np.zeros((SAMPLE_SIZE, L))  # forecast ensemble placeholder
        Y1 = np.zeros((SAMPLE_SIZE, dy)) # ensemble predicted observations placeholder

        x_OT       = np.zeros((N, L))   # mean filtered trajectory (local diagnostic)
        x_OT[0, :] = X0.mean(axis=0)
        SAVE_all_X_OT[k, 0, :, :] = X0

        # --- Assimilation Loop ---
        for i in range(N - 1):

            # Process noise for ensemble propagation
            sai_train = np.random.multivariate_normal(np.zeros(L), sigmma * sigmma * np.eye(L), SAMPLE_SIZE)

            # Forecast step: propagate ensemble forward by one time step
            if rk45:
                sai_train = sai_train.T
                solver    = RK45(A, t[i], (X0.T).reshape(-1), T, first_step=tau)
                solver.step()
                X1 = (solver.y.reshape(L, SAMPLE_SIZE) + sai_train).T  # RK45 forecast with noise, shape (SAMPLE_SIZE, L)
            else:
                X1 = X0 + ((A(t[i], X0.T)).reshape(L, SAMPLE_SIZE) * tau).T + sai_train  # Euler forecast with noise

            # Ensemble predicted observations with perturbed noise
            eta_train = np.random.multivariate_normal(np.zeros(dy), gamma * gamma * np.eye(dy), SAMPLE_SIZE)
            Y1        = np.array(h(X1.T).T + eta_train)  # shape (SAMPLE_SIZE, dy)

            # Convert forecast ensemble to float32 tensors on device
            X1_train = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_train = torch.from_numpy(Y1).to(torch.float32).to(device)

            # --- Kalman Gain (warm start for T) ---
            m_hat = X1_train.mean(axis=0, keepdims=True)  # ensemble mean state,       shape (1, L)
            o_hat = Y1_train.mean(axis=0, keepdims=True)  # ensemble mean observation, shape (1, dy)

            a = X1_train - m_hat  # state anomalies,       shape (SAMPLE_SIZE, L)
            b = Y1_train - o_hat  # observation anomalies, shape (SAMPLE_SIZE, dy)

            C_xy = 1 / X1_train.shape[0] * a.T @ b  # state-observation cross-covariance, shape (L, dy)
            C_yy = 1 / X1_train.shape[0] * b.T @ b  # observation error covariance,       shape (dy, dy)

            K = C_xy @ torch.linalg.inv(C_yy + torch.eye(dy, device=device) * 1e-6)  # Kalman gain with Tikhonov regularization

            # Train f and T on the current forecast ensemble
            train(convex_f, MAP_T, X1_train, Y1_train, ITERS, LR_f, LR_T, inner_iterations, i + 1, N, BATCH_SIZE, k, AVG_SIM)

            # Halve iteration budget each step until the minimum is reached
            if ITERS > Final_Number_ITERATION and i % 1 == 0:
                ITERS = int(ITERS / 2)
                if ITERS < Final_Number_ITERATION:
                    ITERS = Final_Number_ITERATION

            # Prepare true observation at next time step, broadcast to ensemble shape
            Y1_true = torch.from_numpy(y[i + 1, :]).to(torch.float32).to(device)
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)  # forecast ensemble as inference input

            # Apply trained transport map to push forecast ensemble to posterior
            map_T = MAP_T.forward(X1_test, Y1_true * torch.ones((X1_test.shape[0], dy), device=device))

            # Store results and advance ensemble to next step
            X0                             = map_T.cpu().detach().numpy()
            x_OT[i + 1, :]                = torch.mean(map_T, dim=0).cpu().detach().numpy()
            SAVE_all_X_OT[k, i + 1, :, :] = map_T.cpu().detach().numpy()

    SAVE_all_X_OT = SAVE_all_X_OT.transpose((0, 1, 3, 2))
    print("--- OT time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT

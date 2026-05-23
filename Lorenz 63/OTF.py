"""
@author: Mohammad Al-Jarrah
"""

# --- Imports ---

import numpy as np
import time
import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.integrate import odeint


def OTF(Y, X0_C, parameters, A, h, t, tau, Noise, Odeint, delta, device):
    """
    Run the Optimal Transport Filter (OTF).

    Args:
        Y          (ndarray):  observations, shape (AVG_SIM, N, dy)
        X0_C       (ndarray):  initial particle ensembles, shape (AVG_SIM, L, J)
        parameters (dict):     network and training hyperparameters from get_params()
        A          (callable): state transition function A(x, t) returning dx/dt
        h          (callable): observation function mapping state to observation space
        t          (ndarray):  time grid of length N
        tau        (float):    time step size
        Noise      (list):     [noise, sigmma, sigmma0, gamma, x0_amp]
        Odeint     (bool):     if True use odeint for propagation, else use Euler
        delta      (list):     regularization weights [delta_T, delta_f]
        device:                torch device for network computations

    Returns:
        ndarray: filtered particle ensemble, shape (AVG_SIM, N, L, J)
    """
    AVG_SIM = X0_C.shape[0]  # number of simulation runs
    L       = X0_C.shape[1]  # state dimension
    J       = X0_C.shape[2]  # ensemble size

    N  = Y.shape[1]  # number of time steps
    dy = Y.shape[2]  # observation dimension

    sigmma = Noise[0]  # process noise std for the hidden state
    gamma  = Noise[1]  # observation noise std
    T       = N * tau   # total simulation time

    delta_T = delta[0]  # regularization weight for the transport map
    delta_f = delta[1]  # regularization weight for the potential

    # --- Unpack Network Parameters ---
    normalization = parameters['normalization']          # input normalization type
    NUM_NEURON    = parameters['NUM_NEURON']             # hidden layer width
    INPUT_DIM     = parameters['INPUT_DIM']              # [L, dy] model input shape
    BATCH_SIZE    = parameters['BATCH_SIZE']             # mini-batch size
    LearningRate  = parameters['LearningRate']           # base learning rate
    ITERATION     = parameters['ITERATION']              # initial training iterations per step
    final_iter    = parameters['Final_Number_ITERATION'] # minimum iteration floor after decay


    # --- Network Definitions ---

    class NeuralNet(nn.Module):
        """Potential network f used in the OT dual formulation.

        Takes concatenated (x, y) as input and outputs a scalar potential.
        """

        def __init__(self, input_dim, hidden_dim):
            super(NeuralNet, self).__init__()
            self.input_dim  = input_dim   # [L, dy] input dimensions
            self.hidden_dim = hidden_dim  # number of hidden units per layer
            self.activation  = nn.ELU()
            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=True)  # input projection
            self.layer11     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)                        # first hidden layer
            self.layer12     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)                        # second hidden layer
            self.layer_out   = nn.Linear(self.hidden_dim, 1, bias=True)                                      # scalar output

        def forward(self, x, y):
            X  = self.layer_input(torch.concat((x, y), dim=1))
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12(xy)
            xy = self.layer_out(self.activation(xy) + X)  # residual connection before output
            return xy


    class T_NeuralNet(nn.Module):
        """Transport map network T used in the OT dual formulation.

        Takes concatenated (x, y) as input and outputs a mapped state vector of size L.
        """

        def __init__(self, input_dim, hidden_dim):
            super(T_NeuralNet, self).__init__()
            self.input_dim  = input_dim   # [L, dy] input dimensions
            self.hidden_dim = hidden_dim  # number of hidden units per layer
            self.activation  = nn.ReLU()
            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=True)  # input projection
            self.layer11     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)                        # first hidden layer
            self.layer12     = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)                        # second hidden layer
            self.layer_out   = nn.Linear(self.hidden_dim, input_dim[0], bias=True)                           # L-dimensional output

        def forward(self, x, y):
            X  = self.layer_input(torch.concat((x, y), dim=1))
            xy = self.layer11(X)
            xy = self.activation(xy)
            xy = self.layer12(xy)
            xy = self.layer_out(self.activation(xy) + X)  # residual connection before output
            return xy


    # --- Weight Initialization ---

    def init_weights(m):
        """Initialize Linear layer weights with Xavier uniform and biases to 0.001."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.001)


    # --- Training Loop ---

    def train(f, T, X_Train, Y_Train, iterations, learning_rate, ts, Ts, batch_size, k, K):
        """
        Train the potential f and transport map T networks for one time step.

        Args:
            f             (NeuralNet):   potential network
            T             (T_NeuralNet): transport map network
            X_Train       (Tensor): propagated particles, shape (J, L)
            Y_Train       (Tensor): predicted observations, shape (J, dy)
            iterations    (int): number of outer training iterations
            learning_rate (float): base learning rate
            ts            (int): current time step index (for logging)
            Ts            (int): total number of time steps (for logging)
            batch_size    (int): mini-batch size
            k             (int): current simulation index (for logging)
            K             (int): total number of simulations (for logging)
        """
        f.train()
        T.train()

        optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate/1)   # transport map optimizer
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate/10)  # potential optimizer (slower LR)

        scheduler_f = CosineAnnealingWarmRestarts(optimizer_f, T_0=iterations, T_mult=2, eta_min=learning_rate * 1e-4)
        scheduler_T = CosineAnnealingWarmRestarts(optimizer_T, T_0=iterations, T_mult=2, eta_min=learning_rate * 1e-3)

        inner_iterations = 10  # T update steps per outer f iteration
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)  # shuffled Y for unpaired samples

        for i in range(iterations):
            idx  = torch.randperm(X_Train.shape[0])[:batch_size]  # random particle indices for first batch
            idx2 = torch.randperm(X_Train.shape[0])[:batch_size]  # independent batch for monotonicity reg

            X_train    = X_Train[idx].clone().detach()
            Y_train    = Y_Train[idx].clone().detach()
            X_train2   = X_Train[idx2].clone().detach()
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)  # shuffled observations (breaks coupling)

            for j in range(inner_iterations):
                map_T      = T.forward(X_train, Y_shuffled)
                f_of_map_T = f.forward(map_T, Y_shuffled)

                reg = 0
                if delta_T != 0:
                    map_T2 = T(X_train2, Y_shuffled)
                    # Monotonicity regularization: penalize violations of (T(x2)-T(x))*(x2-x) >= 0
                    reg = nn.functional.elu(
                        ((map_T2 - map_T) * (-X_train2 + X_train)).sum(axis=1), alpha=0.01
                    ).mean()

                loss_T = -f_of_map_T.mean() + 0.5*((X_train - map_T)*(X_train - map_T)).sum(axis=1).mean() + delta_T * reg
                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()

            f_of_xy    = f.forward(X_train, Y_train)
            map_T      = T.forward(X_train, Y_shuffled)
            f_of_map_T = f.forward(map_T, Y_shuffled)

            # Optional Hessian regularization to promote c-concavity of f
            reg2 = 0
            if delta_f != 0:
                K_hessian = batch_size  # number of particles used to estimate the Hessian

                def f_scalar(x_flat, y_flat):
                    return f(x_flat.unsqueeze(0), y_flat.unsqueeze(0)).squeeze()

                H_fn = jacrev(jacrev(f_scalar, argnums=0), argnums=0)

                def hess_diag_norm(x_k, y_k):
                    return torch.norm(H_fn(x_k, y_k).diag())

                x_batch   = X_train[:K_hessian]
                y_batch   = Y_train[:K_hessian]
                laplacian = vmap(hess_diag_norm)(x_batch, y_batch).sum()  # sum of Hessian diagonal norms
                reg2      = nn.functional.elu(laplacian, alpha=0.01) / K_hessian

            loss_f = -f_of_xy.mean() + f_of_map_T.mean() + delta_f * reg2
            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()

            scheduler_f.step()
            scheduler_T.step()

            if (i+1) == iterations:
                with torch.no_grad():
                    f_of_xy    = f.forward(X_Train, Y_Train)
                    map_T      = T.forward(X_Train, Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T, Y_Train_shuffled)
                    loss_f     = f_of_xy.mean() - f_of_map_T.mean()  # f-only dual gap
                    loss       = f_of_xy.mean() - f_of_map_T.mean() + 0.5*((X_Train - map_T)*(X_Train - map_T)).sum(axis=1).mean()  # full OT loss
                    print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" % (k+1, K, ts, Ts-1, i+1, iterations, loss.item()))


    # --- Normalization Utilities ---

    def Normalization(X, Type='None'):
        """Normalize data using the specified method. Returns (shift, scale, normalized_data)."""
        if Type == 'None':
            return 0, 0, X
        elif Type == 'Mean':
            Mean_X_training_data = torch.mean(X)  # mean of training data
            Std_X_training_data  = torch.std(X)   # std of training data
            return Mean_X_training_data, Std_X_training_data, (X - Mean_X_training_data) / Std_X_training_data
        elif Type == 'MinMax':
            Min = torch.min(X)  # minimum of training data
            Max = torch.max(X)  # maximum of training data
            return Min, Max, (X - Min) / (Max - Min)

    def Transfer(M, S, X, Type='None'):
        """Apply stored normalization parameters to new data."""
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return (X - M) / S
        elif Type == 'MinMax':
            return (X - M) / (S - M)

    def deTransfer(M, S, X, Type='None'):
        """Invert normalization to recover original-scale data."""
        if Type == 'None':
            return X
        elif Type == 'Mean':
            return X * S + M
        elif Type == 'MinMax':
            return X * (S - M) + M


    # --- Main Filter Loop ---

    start_time    = time.time()
    SAVE_all_X_OT = np.zeros((AVG_SIM, N, J, L))  # output array storing all filtered particles

    for k in range(AVG_SIM):
        y = Y[k,]  # observation sequence for this simulation run

        ITERS = ITERATION    # current iteration budget (decays over time steps)
        LR    = LearningRate # learning rate passed to the train function

        convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)    # potential network f
        MAP_T    = T_NeuralNet(INPUT_DIM, NUM_NEURON)  # transport map network T

        convex_f.apply(init_weights)
        MAP_T.apply(init_weights)
        convex_f.to(device)
        MAP_T.to(device)

        X0   = X0_C[k,].T        # initial particles, shape (J, L)
        X1   = np.zeros((J, L))  # propagated particles for the current step
        Y1   = np.zeros((J, dy)) # predicted observations for the current step
        x_OT = np.zeros((N, L))  # running ensemble mean trajectory (not returned)
        x_OT[0, :] = X0.mean(axis=0)
        SAVE_all_X_OT[k, 0, :, :] = X0

        for i in range(N-1):

            # --- Propagate particles ---
            sai_train = np.random.multivariate_normal(np.zeros(L), sigmma*sigmma * np.eye(L), J)  # (J, L) process noise
            if Odeint:
                sai_train = sai_train.T
                X1 = ((odeint(A, (X0.T).reshape(-1), t[i:i+2])[1,]).reshape(L, J) + sai_train).T
            else:
                X1 = X0 + (((A(X0.T, t[i]).T) * tau) + sai_train)

            eta_train = np.random.multivariate_normal(np.zeros(dy), gamma*gamma * np.eye(dy), J)  # (J, dy) observation noise
            Y1        = np.array(h(X1.T).T + eta_train)                                           # (J, dy) predicted observations

            # Convert training data to float32 tensors on device
            X1_train = torch.from_numpy(X1).to(torch.float32).to(device)
            Y1_train = torch.from_numpy(Y1).to(torch.float32).to(device)

            train(convex_f, MAP_T, X1_train, Y1_train, ITERS, LR, i+1, N, BATCH_SIZE, k, AVG_SIM)

            # Decay iteration budget toward the minimum floor
            if ITERS > final_iter and i % 1 == 0 and i >= 5:
                ITERS = int(ITERS / 2)
                if ITERS < final_iter:
                    ITERS = final_iter

            # --- Update particles with the true observation ---
            Y1_true = torch.from_numpy(y[i+1, :]).to(torch.float32).to(device)  # true observation at t+1
            X1_test = torch.from_numpy(X1).to(torch.float32).to(device)         # propagated particles as test input

            map_T = MAP_T.forward(X1_test, Y1_true * torch.ones((X1_test.shape[0], dy), device=device))

            X0                           = map_T.cpu().detach().numpy()  # updated particles for next step
            x_OT[i+1, :]                = torch.mean(map_T, dim=0).cpu().detach().numpy()
            SAVE_all_X_OT[k, i+1, :, :] = map_T.cpu().detach().numpy()

    print("--- OT time : %s seconds ---" % (time.time() - start_time))
    return SAVE_all_X_OT.transpose((0, 1, 3, 2))

"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time


def rk4_step(f, t, x, tau):
    k1 = f(t,         x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau,   x + tau   * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)


def SIR(Y, X0, A, h, t, tau, Noise, rk4):
    """
    Run the Sequential Importance Resampling (SIR) particle filter.

    Args:
        Y     (ndarray): observations, shape (AVG_SIM, N, dy)
        X0    (ndarray): initial particle ensembles, shape (AVG_SIM, L, J)
        A     (callable): state transition function A(t, x) returning dx/dt
        h     (callable): observation function mapping state to observation space
        t     (ndarray): time grid of length N
        tau   (float): time step size
        Noise (list): [noise, sigmma, sigmma0, gamma, x0_amp]
        rk4   (bool): if True use RK4 integration, else use Euler

    Returns:
        ndarray: filtered particle ensemble, shape (AVG_SIM, N, L, J)
    """
    AVG_SIM = X0.shape[0]  # number of simulation runs
    L       = X0.shape[1]  # state dimension
    J       = X0.shape[2]  # ensemble size

    N  = Y.shape[1]  # number of time steps
    dy = Y.shape[2]  # observation dimension

    sigmma = Noise[0]  # process noise std for the hidden state
    gamma  = Noise[1]  # observation noise std

    start_time = time.time()
    x_SIR      = np.zeros((AVG_SIM, N, L, J))  # output particle array

    rng = np.random.default_rng()  # random number generator for resampling
    for k in range(AVG_SIM):
        x_SIR[k, 0,] = X0[k,]
        y = Y[k,]  # observation sequence for this simulation run

        for i in range(N-1):
            # --- Propagate particles ---
            sai_SIR       = np.random.multivariate_normal(np.zeros(L), sigmma*sigmma * np.eye(L), J).transpose()  # (L, J) process noise
            if rk4:
                x_SIR[k, i+1,] = rk4_step(A, t[i], x_SIR[k, i,], tau) + sai_SIR
            else:
                x_SIR[k, i+1,] = x_SIR[k, i,] + A(t[i], x_SIR[k, i,])*tau + sai_SIR

            # --- Compute and normalize importance weights ---
            W = np.sum((y[i+1,] - h(x_SIR[k, i+1,]).T) * (y[i+1] - h(x_SIR[k, i+1,]).T), axis=1) / (2*gamma*gamma)
            W = W - np.min(W)   # shift for numerical stability before exp
            W = np.exp(-W).T
            W = W / np.sum(W)   # normalize to a probability distribution

            # --- Resample particles ---
            index         = rng.choice(np.arange(J), J, p=W)  # indices sampled according to W
            x_SIR[k, i+1,] = x_SIR[k, i+1, :, index].T

    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    return x_SIR

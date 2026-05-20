"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time

from scipy.integrate import RK45


def SIR(Y, X0, A, h, t, tau, Noise, rk45):
    """
    Sequential Importance Resampling (SIR) particle filter for state estimation.

    Parameters
    ----------
    Y      : ndarray, shape (AVG_SIM, N, dy)  — observations across all simulations
    X0     : ndarray, shape (AVG_SIM, L, J)   — initial ensemble states
    A      : callable                         — dynamics function A(t, x)
    h      : callable                         — observation operator h(x)
    t      : ndarray                          — time grid
    tau    : float                            — time step size
    Noise  : tuple                            — noise parameters (noise, sigma, sigma0, gamma, x0_amp)
    rk45   : bool                             — use RK45 integrator if True, else forward Euler

    Returns
    -------
    ndarray, shape (AVG_SIM, N, L, J) — filtered particle trajectories
    """

    # --- Dimensions ---
    AVG_SIM = X0.shape[0]  # number of independent simulation runs
    L       = X0.shape[1]  # state space dimension
    J       = X0.shape[2]  # number of particles

    N  = Y.shape[1]         # number of observation time steps
    dy = Y.shape[2]         # observation space dimension

    # --- Noise Parameters ---
    sigmma = Noise[0]  # std of process noise in the hidden state
    gamma  = Noise[1]  # std of observation noise

    T = tau * N             # total integration time horizon

    start_time = time.time()

    x_SIR = np.zeros((AVG_SIM, N, L, J))  # storage for particle state trajectories

    rng = np.random.default_rng()          # seeded RNG used for systematic resampling

    # --- Main Loop Over Simulations ---
    for k in range(AVG_SIM):

        x_SIR[k, 0,] = X0[k,]  # initialize particles from prior, shape (L, J)
        y = Y[k,]               # observations for simulation k, shape (N, dy)

        # --- SIR Filter Loop ---
        for i in range(N - 1):

            # Process noise drawn for particle perturbation, transposed to (L, J)
            sai_SIR = np.random.multivariate_normal(np.zeros(L), sigmma * sigmma * np.eye(L), J).transpose()

            # Forecast step: propagate particles forward by one time step
            if rk45:
                solver           = RK45(A, t[i], x_SIR[k, i,].reshape(-1), T, first_step=tau)
                solver.step()
                x_SIR[k, i + 1,] = solver.y.reshape(L, J) + sai_SIR        # RK45 forecast with noise
            else:
                x_SIR[k, i + 1,] = x_SIR[k, i,] + A(t[i], x_SIR[k, i,]).reshape(L, J) * tau + sai_SIR  # Euler forecast with noise

            # --- Weight Computation ---
            # Log-likelihood weights based on squared observation residuals
            W = np.sum((y[i + 1,] - h(x_SIR[k, i + 1,]).T) * (y[i + 1] - h(x_SIR[k, i + 1,]).T), axis=1) / (2 * gamma * gamma)

            W = W - np.min(W)   # shift for numerical stability before exponentiation
            W = np.exp(-W).T    # convert to unnormalized likelihood weights
            W = W / np.sum(W)   # normalize to form a probability distribution

            # --- Resampling Step ---
            index            = rng.choice(np.arange(J), J, p=W)    # resample particle indices according to weights
            x_SIR[k, i + 1,] = x_SIR[k, i + 1, :, index].T        # replace particles with resampled set

    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    return x_SIR

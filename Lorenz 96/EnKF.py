"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time

from scipy.integrate import RK45


def EnKF(Y, X0, A, h, t, tau, Noise, rk45):
    """
    Ensemble Kalman Filter (EnKF) for sequential state estimation.

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
    ndarray, shape (AVG_SIM, N, L, J) — filtered ensemble trajectories
    """

    # --- Dimensions ---
    AVG_SIM = X0.shape[0]  # number of independent simulation runs
    L       = X0.shape[1]  # state space dimension
    J       = X0.shape[2]  # ensemble size

    N  = Y.shape[1]         # number of observation time steps
    dy = Y.shape[2]         # observation space dimension

    # --- Noise Parameters ---
    sigmma = Noise[0]  # std of process noise in the hidden state
    gamma  = Noise[1]  # std of observation noise

    T = tau * N             # total integration time horizon

    start_time = time.time()

    SAVE_X_EnKF = np.zeros((AVG_SIM, N, J, L))  # storage for filtered ensemble states

    # --- Main Loop Over Simulations ---
    for k in range(AVG_SIM):

        y = Y[k,]  # observations for simulation k, shape (N, dy)

        x_EnKF     = np.zeros((N, J, L))  # ensemble state trajectory
        x_EnKF[0,] = X0[k,].T            # initialize ensemble from prior, transposed to (J, L)

        SAVE_X_EnKF[k, 0, :, :] = x_EnKF[0,]

        # --- EnKF Assimilation Loop ---
        for i in range(N - 1):

            # Process noise drawn for ensemble perturbation
            sai_EnKF = np.random.multivariate_normal(np.zeros(L), sigmma * sigmma * np.eye(L), J)

            # Forecast step: propagate ensemble forward by one time step
            if rk45:
                solver    = RK45(A, t[i], (x_EnKF[i,].T).reshape(-1), T, first_step=tau)
                solver.step()
                x_hatEnKF = (solver.y.reshape(L, J)).T + sai_EnKF        # RK45 forecast with noise, shape (J, L)
            else:
                x_hatEnKF = x_EnKF[i,] + tau * (A(t[i], x_EnKF[i,].T).reshape(L, J)).T + sai_EnKF  # Euler forecast with noise

            # Perturbed observations for ensemble members
            eta_EnKF  = np.random.multivariate_normal(np.zeros(dy), gamma * gamma * np.eye(dy), J)  # obs noise, shape (J, dy)
            y_hatEnKF = h(x_hatEnKF.T).T + eta_EnKF                                                 # predicted observations, shape (J, dy)

            # --- Kalman Gain Computation ---
            X_hat = x_hatEnKF.mean(axis=0, keepdims=True)  # ensemble mean state,       shape (1, L)
            Y_hat = y_hatEnKF.mean(axis=0, keepdims=True)  # ensemble mean observation, shape (1, dy)

            a = x_hatEnKF - X_hat  # state anomalies,       shape (J, L)
            b = y_hatEnKF - Y_hat  # observation anomalies, shape (J, dy)

            C_xy = 1 / J * a.T @ b  # state-observation cross-covariance, shape (L, dy)
            C_yy = 1 / J * b.T @ b  # observation error covariance,       shape (dy, dy)

            K = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-6)  # Kalman gain with Tikhonov regularization, shape (L, dy)

            # --- Analysis Step ---
            x_EnKF[i + 1, :, :] = x_hatEnKF + (K @ (y[i + 1, :] - y_hatEnKF).T).T  # ensemble update

            SAVE_X_EnKF[k, i + 1, :, :] = x_EnKF[i + 1,]

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_EnKF.transpose(0, 1, 3, 2)

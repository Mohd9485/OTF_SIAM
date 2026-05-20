"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time


def EnKF(Y, X0, A, h, t, tau, Noise, Odeint):
    """
    Run the Ensemble Kalman Filter (EnKF).

    Args:
        Y      (ndarray): observations, shape (AVG_SIM, N, dy)
        X0     (ndarray): initial particle ensembles, shape (AVG_SIM, L, J)
        A      (callable): state transition function A(x, t) returning dx/dt
        h      (callable): observation function mapping state to observation space
        t      (ndarray): time grid of length N
        tau    (float): time step size
        Noise  (list): [noise, sigmma, sigmma0, gamma, x0_amp]
        Odeint (bool): unused flag for ODE integration method

    Returns:
        ndarray: filtered ensemble, shape (AVG_SIM, N, L, J)
    """
    AVG_SIM = X0.shape[0]  # number of simulation runs
    L       = X0.shape[1]  # state dimension
    J       = X0.shape[2]  # ensemble size

    N  = Y.shape[1]  # number of time steps
    dy = Y.shape[2]  # observation dimension

    sigmma = Noise[0]  # process noise std for the hidden state
    gamma  = Noise[1]  # observation noise std

    start_time   = time.time()
    SAVE_X_EnKF  = np.zeros((AVG_SIM, N, J, L))  # output array storing all filtered states

    for k in range(AVG_SIM):
        y = Y[k,]  # observation sequence for this simulation run

        x_EnKF     = np.zeros((N, J, L))  # per-simulation filtered ensemble
        x_EnKF[0,] = X0[k,].T

        SAVE_X_EnKF[k, 0, :, :] = x_EnKF[0,]

        for i in range(N-1):

            # --- Forecast step ---
            sai_EnKF  = np.random.multivariate_normal(np.zeros(L), sigmma*sigmma * np.eye(L), J)    # (J, L) process noise
            x_hatEnKF = x_EnKF[i,] + tau*A(x_EnKF[i,].T, t[i]).T + sai_EnKF                        # (J, L) propagated ensemble

            eta_EnKF  = np.random.multivariate_normal(np.zeros(dy), gamma*gamma * np.eye(dy), J)    # (J, dy) observation noise
            y_hatEnKF = h(x_hatEnKF.T).T + eta_EnKF                                                 # (J, dy) predicted observations

            # --- Compute Kalman gain ---
            X_hat = x_hatEnKF.mean(axis=0, keepdims=True)  # (1, L) ensemble mean of states
            Y_hat = y_hatEnKF.mean(axis=0, keepdims=True)  # (1, dy) ensemble mean of predicted obs

            a   = x_hatEnKF - X_hat  # (J, L) state anomalies
            b   = y_hatEnKF - Y_hat  # (J, dy) observation anomalies

            C_xy = 1/J * a.T @ b  # (L, dy) state-observation cross-covariance
            C_yy = 1/J * b.T @ b  # (dy, dy) observation-observation covariance

            K = C_xy @ np.linalg.inv(C_yy + np.eye(dy)*1e-6)  # (L, dy) Kalman gain

            # --- Analysis step ---
            x_EnKF[i+1, :, :] = x_hatEnKF + (K @ (y[i+1, :] - y_hatEnKF).T).T

            SAVE_X_EnKF[k, i+1, :, :] = x_EnKF[i+1,]

    print("--- EnKF time : %s seconds ---" % (time.time() - start_time))
    return SAVE_X_EnKF.transpose(0, 1, 3, 2)

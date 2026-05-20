"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

from scipy.integrate import RK45
from EnKF import EnKF
from SIR import SIR
from OTF import OTF
from param import get_params


# --- Random Seed ---
randint = np.random.randint(0, 100000)  # random seed for reproducibility
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)


# --- Device Setup ---

def get_free_gpu(n=1):
    """
    Query nvidia-smi and return the IDs of the n least-loaded GPUs by memory usage.

    Parameters
    ----------
    n : int — number of GPU IDs to return

    Returns
    -------
    list of int — GPU device IDs sorted by ascending memory usage
    """
    result     = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    memory_used = [int(x) for x in result.stdout.strip().split("\n")]
    sorted_ids  = sorted(range(len(memory_used)), key=lambda i: memory_used[i])
    return sorted_ids[:n]

if torch.cuda.is_available():
    gpu_ids = get_free_gpu(n=2)
    device1 = torch.device(f"cuda:{gpu_ids[0]}")
    device2 = torch.device(f"cuda:{gpu_ids[1]}") if len(gpu_ids) > 1 else device1
    torch.cuda.manual_seed(randint)
    # Warm up cuSOLVER (used by MultivariateNormal) on both devices before any filter call
    for _dev in [device1, device2]:
        _t = torch.eye(2, device=_dev)
        torch.linalg.cholesky(_t)
        del _t
else:
    device1 = torch.device("cpu")
    device2 = torch.device("cpu")
print(f"Using device: {device1}, {device2}")

plt.close('all')


# --- Problem Setup ---
L               = 9             # state space dimension
tau             = 0.01          # time step size
T               = 5             # total simulation time in seconds
F               = 10            # forcing constant for the L96 model
N               = int(T / tau)  # number of time steps
dy              = L // 3        # number of observed state components
observation_idx = list(range(0, L, L // dy))[:dy]  # indices of observed state components

rk45 = True  # use RK45 integrator for data generation and filtering

noise   = 0.1            # std of process and observation noise
sigmma  = noise          # std of process noise in the hidden state
sigmma0 = np.sqrt(1e1)   # std of noise in the initial state distribution
gamma   = noise          # std of observation noise
x0_amp  = 1              # amplitude scaling applied to the initial state
Noise   = [sigmma, gamma]

J       = 250 * 4   # ensemble size for all filters
AVG_SIM = 10        # number of independent simulation runs to average over
delta   = [1e-1, 1e-1]  # regularization weights (delta_T, delta_f) for OT_reg

parameters = get_params(L, dy)  # network and training hyperparameters from param.py

t = np.arange(0.0, tau * N, tau)  # time grid, shape (N,)


# --- Model and Observation Functions ---

def h(x):
    """Observation operator: extract observed components from state x."""
    return x[observation_idx,]


def L96(t, x):
    """
    Lorenz 96 vector field for a single state vector.

    Parameters
    ----------
    t : float           — current time (unused, required by RK45 interface)
    x : ndarray, shape (L,) — current state vector

    Returns
    -------
    ndarray, shape (L,) — time derivative of x
    """
    d = np.zeros_like(x)
    for i in range(L):
        d[i] = (x[(i + 1) % L] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


def ML96(t, x):
    """
    Lorenz 96 vector field for a stacked ensemble matrix (vectorized over particles).

    Parameters
    ----------
    t : float               — current time (unused, required by RK45 interface)
    x : ndarray, shape (L*J,) — flattened ensemble matrix

    Returns
    -------
    ndarray, shape (L*J,) — flattened time derivatives for all particles
    """
    x = x.reshape(L, -1)
    d = np.zeros_like(x)
    for i in range(L):
        d[i, :] = (x[(i + 1) % L, :] - x[i - 2, :]) * x[i - 1, :] - x[i, :] + F
    return d.reshape(-1)


def Gen_Data(L, dy, N, x0_amp, sigmma0, sigmma, gamma, tau, rk45):
    """
    Generate one realization of the true trajectory and noisy observations.

    Parameters
    ----------
    L      : int   — state space dimension
    dy     : int   — observation space dimension
    N      : int   — number of time steps
    x0_amp : float — amplitude scaling for the initial state
    sigmma0: float — std of initial state distribution
    sigmma : float — std of process noise
    gamma  : float — std of observation noise
    tau    : float — time step size
    rk45   : bool  — use RK45 integrator if True, else forward Euler

    Returns
    -------
    x : ndarray, shape (N, L)  — true hidden state trajectory
    y : ndarray, shape (N, dy) — noisy observations
    """
    sai = np.random.multivariate_normal(np.zeros(L),  sigmma * sigmma * np.eye(L),  N)  # process noise samples
    eta = np.random.multivariate_normal(np.zeros(dy), gamma * gamma  * np.eye(dy), N)  # observation noise samples

    x      = np.zeros((N, L))   # true state trajectory
    y      = np.zeros((N, dy))  # noisy observation sequence
    x0     = 10 + x0_amp * np.random.multivariate_normal(np.zeros(L), sigmma0 * sigmma0 * np.eye(L), 1)
    x[0,]  = x0

    for i in range(N - 1):
        if rk45:
            solver   = RK45(L96, t[i], x[i,], T, first_step=tau)
            solver.step()
            x[i + 1,] = solver.y + sai[i,]
        else:
            x[i + 1, :] = x[i, :] + L96(t[i], x[i, :]) * tau + sai[i, :]
        y[i + 1,] = h(x[i + 1,]) + eta[i + 1,]
    return x, y


def mse(x, x_true):
    """
    Compute per-time-step MSE between the ensemble mean and the true trajectory.

    Parameters
    ----------
    x      : ndarray, shape (AVG_SIM, N, L, J) — filter ensemble output
    x_true : ndarray, shape (AVG_SIM, N, L)    — true state trajectory

    Returns
    -------
    ndarray, shape (N,) — MSE at each time step, averaged over simulations and state components
    """
    x_mean = (x - x_true.reshape(AVG_SIM, N, L, 1)).mean(axis=3)  # residual from ensemble mean
    return ((x_mean * x_mean).sum(axis=2)).mean(axis=0)


# --- Generate Data ---
X_True = np.zeros((AVG_SIM, N, L))   # true state trajectories across all simulations
Y_True = np.zeros((AVG_SIM, N, dy))  # noisy observations across all simulations
X0     = np.zeros((AVG_SIM, L, J))   # initial ensembles across all simulations

for k in range(AVG_SIM):
    x, y      = Gen_Data(L, dy, N, x0_amp, sigmma0, sigmma, gamma, tau, rk45)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,]     = 10 + x0_amp * np.transpose(
        np.random.multivariate_normal(np.zeros(L), sigmma0 * sigmma0 * np.eye(L), J)
    )


# --- Run Filters ---
total_start = time.time()
X_EnKF = EnKF(Y_True, X0, ML96, h, t, tau, Noise, rk45)
X_SIR  = SIR(Y_True, X0, ML96, h, t, tau, Noise, rk45)

# Run OT and OT_reg concurrently on separate devices
with ThreadPoolExecutor(max_workers=3) as executor:
    future_reg = executor.submit(OTF, Y_True, X0, parameters, ML96, h, t, tau, Noise, rk45, delta,   device1)
    future_ot  = executor.submit(OTF, Y_True, X0, parameters, ML96, h, t, tau, Noise, rk45, [0, 0], device2)
    X_OT_reg   = future_reg.result()
    X_OT       = future_ot.result()
print(f"--- Total time (all methods): {time.time() - total_start:.2f} seconds ---")


# --- Compute MSE ---
MSE_EnKF   = mse(X_EnKF,   X_True)
MSE_SIR    = mse(X_SIR,    X_True)
MSE_OT     = mse(X_OT,     X_True)
MSE_OT_reg = mse(X_OT_reg, X_True)
print(f"MSE EnKF:   mean={MSE_EnKF.mean():.4f}")
print(f"MSE SIR:    mean={MSE_SIR.mean():.4f}")
print(f"MSE OT:     mean={MSE_OT.mean():.4f}")
print(f"MSE OT_reg: mean={MSE_OT_reg.mean():.4f}")



# --- Plotting ---
j             = 0     # simulation index to visualize
x_lim         = 25    # y-axis limit for state trajectory plots

for s in range(3):
    plt.figure(figsize=(8, 9))

    plt.subplot(4, 1, 1)
    plt.plot(t, X_EnKF[j, :, s, :], 'C4', alpha=0.1)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2, label='True state')
    plt.ylabel('EnKF')
    plt.legend(loc=4)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 2)
    plt.plot(t, X_SIR[j, :, s, :], 'C5', alpha=0.1)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel('SIR')
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 3)
    plt.plot(t, X_OT[j, :, s, :], 'b', alpha=0.1)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT~(\lambda=0)$')
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 4)
    plt.plot(t, X_OT_reg[j, :, s, :], 'g', alpha=0.1)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT~(\lambda=0.1)$')
    plt.xlabel('time')
    plt.ylim(-x_lim, x_lim)

# MSE comparison
plt.figure(figsize=(8, 9))
plt.semilogy(t, MSE_OT,     'b--', label=r"$OT_{(\lambda=0)}$",    lw=2)
plt.semilogy(t, MSE_OT_reg, 'g-.', label=r"$OT_{(\lambda=0.1)}$", lw=2)
plt.semilogy(t, MSE_EnKF,   ':',   label=r"$EnKF$", color='C4',    lw=2.5)
plt.semilogy(t, MSE_SIR,    ':',   label=r"$SIR$",  color='C5',    lw=2.5)
plt.xlabel('time')
plt.ylabel('MSE')
plt.legend(loc=0)


# --- Save Results ---
np.savez(
    'DATA_file_L96.npz',
    randint    = randint,
    time       = t,
    Y_true     = Y_True,
    X_true     = X_True,
    Noise      = Noise,
    X_EnKF     = X_EnKF,
    X_OT       = X_OT,
    X_SIR      = X_SIR,
    MSE_EnKF   = MSE_EnKF,
    MSE_OT     = MSE_OT,
    MSE_SIR    = MSE_SIR,
    X_OT_reg   = X_OT_reg,
    MSE_OT_reg = MSE_OT_reg,
    delta      = delta,
)

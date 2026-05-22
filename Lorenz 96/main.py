"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

from EnKF import EnKF
from SIR import SIR
from OTF import OTF
from OTF_EnKF import OTF_EnKF
from param import get_params, get_params_enkf


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
    result      = subprocess.run(
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
    # Warm up cuSOLVER on both devices before OTF_EnKF (which uses MultivariateNormal)
    for _dev in [device1, device2]:
        _t = torch.eye(2, device=_dev)
        torch.linalg.cholesky(_t)
        del _t
else:
    device1 = torch.device("cpu")
    device2 = torch.device("cpu")
print(f"Using device: {device1}, {device2}")

plt.close('all')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42


# --- Problem Setup ---

L               = 9             # state space dimension
tau             = 0.01          # time step size
T               = 5             # total simulation time in seconds
F               = 10            # forcing constant for the L96 model
N               = int(T / tau)  # number of time steps
dy              = L // 3        # number of observed state components
observation_idx = list(range(0, L, L // dy))[:dy]  # indices of observed state components

rk4     = True   # use RK4 integrator for data generation and filtering
noise   = 0.1    # std of process and observation noise
sigmma  = noise  # std of process noise in the hidden state
sigmma0 = np.sqrt(1e1)  # std of noise in the initial state distribution
gamma   = noise  # std of observation noise
x0_amp  = 1      # amplitude scaling applied to the initial state
Noise   = [sigmma, gamma]

J       = 250 * 4       # ensemble size for all filters
AVG_SIM = 10            # number of independent simulation runs to average over
delta   = [1e-1, 1e-1]  # regularization weights (delta_T, delta_f) for OT_reg

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
    t : float           — current time (unused, required by RK4 interface)
    x : ndarray, shape (L,) — current state vector

    Returns
    -------
    ndarray, shape (L,) — time derivative of x
    """
    d = np.zeros_like(x)
    for i in range(L):
        d[i] = (x[(i + 1) % L] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


def rk4_step(f, t, x, tau):
    k1 = f(t,         x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau,   x + tau   * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)


def ML96(t, x):
    """
    Lorenz 96 vector field for a stacked ensemble matrix (vectorized over particles).

    Parameters
    ----------
    t : float               — current time (unused, required by RK4 interface)
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


def Gen_Data(L, dy, N, x0_amp, sigmma0, sigmma, gamma, tau, rk4):
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
    rk4    : bool  — use RK4 integrator if True, else forward Euler

    Returns
    -------
    x : ndarray, shape (N, L)  — true hidden state trajectory
    y : ndarray, shape (N, dy) — noisy observations
    """
    sai = np.random.multivariate_normal(np.zeros(L),  sigmma * sigmma * np.eye(L),  N)  # process noise
    eta = np.random.multivariate_normal(np.zeros(dy), gamma  * gamma  * np.eye(dy), N)  # observation noise

    x     = np.zeros((N, L))
    y     = np.zeros((N, dy))
    x[0,] = 10 + x0_amp * np.random.multivariate_normal(np.zeros(L), sigmma0 * sigmma0 * np.eye(L), 1)

    for i in range(N - 1):
        if rk4:
            x[i + 1, :] = rk4_step(L96, t[i], x[i, :], tau) + sai[i, :]
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
    x_mean = (x - x_true.reshape(AVG_SIM, N, L, 1)).mean(axis=3)
    return ((x_mean * x_mean).sum(axis=2)).mean(axis=0)


# --- Generate Data ---

X_True = np.zeros((AVG_SIM, N, L))   # true state trajectories across all simulations
Y_True = np.zeros((AVG_SIM, N, dy))  # noisy observations across all simulations
X0     = np.zeros((AVG_SIM, L, J))   # initial ensembles across all simulations

for k in range(AVG_SIM):
    x, y       = Gen_Data(L, dy, N, x0_amp, sigmma0, sigmma, gamma, tau, rk4)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,]     = 10 + x0_amp * np.transpose(
        np.random.multivariate_normal(np.zeros(L), sigmma0 * sigmma0 * np.eye(L), J)
    )


# --- Run Filters ---

def _timed(func, *args):
    t0 = time.time()
    return func(*args), time.time() - t0

total_start = time.time()

t0 = time.time(); X_EnKF = EnKF(Y_True, X0, ML96, h, t, tau, Noise, rk4); time_EnKF = time.time() - t0
t0 = time.time(); X_SIR  = SIR(Y_True, X0, ML96, h, t, tau, Noise, rk4); time_SIR  = time.time() - t0

parameters = get_params(L, dy)  # network and training hyperparameters from param.py
with ThreadPoolExecutor(max_workers=2) as executor:
    future_ot     = executor.submit(_timed, OTF, Y_True, X0, parameters, ML96, h, t, tau, Noise, rk4, [0, 0], device1)
    future_ot_reg = executor.submit(_timed, OTF, Y_True, X0, parameters, ML96, h, t, tau, Noise, rk4, delta,   device2)
    X_OT,     time_OT     = future_ot.result()
    X_OT_reg, time_OT_reg = future_ot_reg.result()

parameters = get_params_enkf(L, dy)  # network and training hyperparameters from param.py
with ThreadPoolExecutor(max_workers=2) as executor:
    future_enkf     = executor.submit(_timed, OTF_EnKF, Y_True, X0, parameters, ML96, h, t, tau, Noise, rk4, [0, 0], device1)
    future_enkf_reg = executor.submit(_timed, OTF_EnKF, Y_True, X0, parameters, ML96, h, t, tau, Noise, rk4, delta,   device2)
    X_OT_EnKF,     time_OT_EnKF     = future_enkf.result()
    X_OT_EnKF_reg, time_OT_EnKF_reg = future_enkf_reg.result()

print(f"--- Total time: {time.time() - total_start:.2f}s ---")
print(f"    EnKF:        {time_EnKF:.2f}s")
print(f"    SIR:         {time_SIR:.2f}s")
print(f"    OT:          {time_OT:.2f}s")
print(f"    OT_reg:      {time_OT_reg:.2f}s")
print(f"    OT_EnKF:     {time_OT_EnKF:.2f}s")
print(f"    OT_EnKF_reg: {time_OT_EnKF_reg:.2f}s")


# --- Compute MSE ---

MSE_EnKF        = mse(X_EnKF,        X_True)
MSE_SIR         = mse(X_SIR,         X_True)
MSE_OT          = mse(X_OT,          X_True)
MSE_OT_reg      = mse(X_OT_reg,      X_True)
MSE_OT_EnKF     = mse(X_OT_EnKF,     X_True)
MSE_OT_EnKF_reg = mse(X_OT_EnKF_reg, X_True)

print(f"MSE EnKF:        mean={MSE_EnKF.mean():.4f}")
print(f"MSE SIR:         mean={MSE_SIR.mean():.4f}")
print(f"MSE OT:          mean={MSE_OT.mean():.4f}")
print(f"MSE OT_reg:      mean={MSE_OT_reg.mean():.4f}")
print(f"MSE OT_EnKF:     mean={MSE_OT_EnKF.mean():.4f}")
print(f"MSE OT_EnKF_reg: mean={MSE_OT_EnKF_reg.mean():.4f}")


# --- Plotting ---

fontsize = 19
j        = 0   # simulation index to visualize
x_lim    = 25  # y-axis limit for state trajectory plots

for s in range(3):
    plt.figure(figsize=(6, 18))

    plt.subplot(6, 1, 1)
    plt.plot(t, X_EnKF[j, :, s, :], 'C1', alpha=0.1, rasterized=True)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2, label='True state')
    plt.ylabel('EnKF', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.legend(loc=4, fontsize=fontsize)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 2)
    plt.plot(t, X_SIR[j, :, s, :], 'C2', alpha=0.1, rasterized=True)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel('SIR', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 3)
    plt.plot(t, X_OT[j, :, s, :], 'C3', alpha=0.1, rasterized=True)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 4)
    plt.plot(t, X_OT_reg[j, :, s, :], 'C4', alpha=0.1, rasterized=True)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT~(\lambda=0.1)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 5)
    plt.plot(t, X_OT_EnKF[j, :, s, :], 'C5', alpha=0.1, rasterized=True)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT_{EnKF}~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 6)
    plt.plot(t, X_OT_EnKF_reg[j, :, s, :], 'C6', alpha=0.1, rasterized=True)
    plt.plot(t, X_True[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT_{EnKF}~(\lambda=0.1)$', fontsize=fontsize)
    plt.xlabel('time', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)

plt.figure(figsize=(8, 6))
plt.semilogy(t, MSE_EnKF,        ':',  color='C1', label=r"$EnKF$",                    lw=2.5)
plt.semilogy(t, MSE_SIR,         ':',  color='C2', label=r"$SIR$",                     lw=2.5)
plt.semilogy(t, MSE_OT,          '--', color='C3', label=r"$OT~(\lambda=0)$",           lw=2.5)
plt.semilogy(t, MSE_OT_reg,      '-.', color='C4', label=r"$OT~(\lambda=0.1)$",         lw=2.5)
plt.semilogy(t, MSE_OT_EnKF,     '--', color='C5', label=r"$OT_{EnKF}~(\lambda=0)$",   lw=2.5)
plt.semilogy(t, MSE_OT_EnKF_reg, '-.', color='C6', label=r"$OT_{EnKF}~(\lambda=0.1)$", lw=2.5)
plt.xlabel('time', fontsize=fontsize)
plt.ylabel('MSE',  fontsize=fontsize)
plt.legend(loc=0,  fontsize=fontsize)
plt.savefig(f"MSE_L96_{time.strftime('%Y%m%d_%H%M%S')}.pdf", bbox_inches='tight')


# --- Save Results ---

np.savez(
    'DATA_file_L96.npz',
    randint          = randint,
    time             = t,
    Y_true           = Y_True,
    X_true           = X_True,
    Noise            = Noise,
    delta            = delta,
    X_EnKF           = X_EnKF,
    X_SIR            = X_SIR,
    X_OT             = X_OT,
    X_OT_reg         = X_OT_reg,
    X_OT_EnKF        = X_OT_EnKF,
    X_OT_EnKF_reg    = X_OT_EnKF_reg,
    MSE_EnKF         = MSE_EnKF,
    MSE_SIR          = MSE_SIR,
    MSE_OT           = MSE_OT,
    MSE_OT_reg       = MSE_OT_reg,
    MSE_OT_EnKF      = MSE_OT_EnKF,
    MSE_OT_EnKF_reg  = MSE_OT_EnKF_reg,
    time_EnKF        = time_EnKF,
    time_SIR         = time_SIR,
    time_OT          = time_OT,
    time_OT_reg      = time_OT_reg,
    time_OT_EnKF     = time_OT_EnKF,
    time_OT_EnKF_reg = time_OT_EnKF_reg,
)

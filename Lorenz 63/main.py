"""
@author: Mohammad Al-Jarrah
"""

# --- Imports ---

import numpy as np
import matplotlib.pyplot as plt
import torch, time
import ot
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from EnKF import EnKF
from SIR import SIR
from OTF import OTF
from param import get_params


# --- Reproducibility ---

randint = np.random.randint(0, 1000)  # random seed drawn at runtime
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)


# --- GPU Selection ---

def get_free_gpu(n=1):
    """Return the indices of the n GPUs with the least memory currently in use."""
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
    # Force full CUDA init (including cuSOLVER used by MultivariateNormal) on both devices
    for _dev in [device1, device2]:
        _t = torch.eye(2, device=_dev)
        torch.linalg.cholesky(_t)
        del _t
else:
    device1 = torch.device("cpu")
    device2 = torch.device("cpu")
print(f"Using device: {device1}, {device2}")

plt.close('all')


# --- Model Definition ---

def h(x):
    """Map the full state vector to the observed component."""
    return x[2,].reshape(dy, -1)

def L63(x, t):
    """Evaluate the Lorenz 63 vector field at state x and time t."""
    d     = np.zeros_like(x)  # output derivative vector
    sigma = 10                 # standard L63 coefficient
    r     = 28                 # standard L63 coefficient
    b     = 8/3                # standard L63 coefficient

    d[0] = sigma*(x[1] - x[0])
    d[1] = x[0]*(r - x[2]) - x[1]
    d[2] = x[0]*x[1] - b*x[2]
    return d


# --- Data Generation ---

def Gen_True_Data(L, dy, N, x0_amp, sigmma0, sigmma, gamma, tau):
    """
    Simulate a noisy Lorenz 63 trajectory and its partial observations.

    Returns:
        x (ndarray): state trajectory of shape (N, L)
        y (ndarray): observation trajectory of shape (N, dy)
    """
    eta = np.random.multivariate_normal(np.zeros(dy), gamma*gamma * np.eye(dy), N)  # observation noise samples

    x   = np.zeros((N, L))                                                           # state trajectory
    y   = np.zeros((N, dy))                                                          # observation trajectory
    x0  = 5 + np.random.multivariate_normal(np.zeros(L), np.eye(L), 1)              # perturbed initial condition

    x[0,] = x0

    for i in range(N-1):
        x[i+1,:] = x[i,:] + L63(x[i,:], t[i])*tau
        y[i+1,]  = h(x[i+1,]) + eta[i+1,]

    return x, y


# --- Metrics ---

def mse(x, x_true):
    """Compute per-timestep MSE between the ensemble mean and the true state."""
    x_mean = (x - x_true.reshape(AVG_SIM, N, L, 1)).mean(axis=3)
    return ((x_mean*x_mean).sum(axis=2)).mean(axis=0)

def w2_distance(x, y):
    """Compute the W2 distance between two empirical distributions in R^L."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.ones(x.shape[0]) / x.shape[0]    # uniform weights over x samples
    b = np.ones(y.shape[0]) / y.shape[0]    # uniform weights over y samples
    M = ot.dist(x, y, metric='sqeuclidean')  # squared-Euclidean cost matrix
    return np.sqrt(ot.emd2(a, b, M))

def w2_vs_sir(X_filt, X_sir_ref, p_true=1000):
    """
    Compute the time-averaged W2 distance between a filter and a large-SIR reference.

    Args:
        X_filt    (ndarray): filtered particles, shape (AVG_SIM, N, L, J)
        X_sir_ref (ndarray): reference SIR particles, shape (AVG_SIM, N, L, J_sir)
        p_true    (int):     number of reference particles to subsample for W2

    Returns:
        float: time-averaged W2 distance
    """
    total_w2 = 0.0
    for n in range(1, N):
        filt_pts  = X_filt[0, n].T                  # (J, L)
        sir_pts   = X_sir_ref[0, n, :, :p_true].T   # (p_true, L)
        total_w2 += w2_distance(filt_pts, sir_pts)
    return total_w2 / (N - 1)


# --- Simulation Parameters ---

L   = 3           # state dimension
tau = 1e-2        # time step size
T   = 5           # final simulation time (seconds)
N   = int(T/tau)  # total number of time steps

dy = 1                 # number of observed states
H  = np.zeros((dy, L)) # observation matrix (unused; h() defines the observation rule instead)

noise   = np.sqrt(1e1)  # noise level standard deviation
sigmma  = noise/10      # process noise std for the hidden state
sigmma0 = noise**2      # variance of the initial state distribution
gamma   = noise/1       # observation noise std
x0_amp  = 1             # initial state amplitude scaling factor
Noise   = [sigmma, gamma]  # packed noise vector passed to filters
Odeint  = False         # use Euler integration (False) rather than odeint

delta   = [0.1, 0.1]   # OTF regularization weights [lambda_T, lambda_f]
J       = int(1000/4)  # EnKF ensemble size
J_sir   = 100_000      # large-SIR reference particle count
p_true  = 1000         # particles subsampled from SIR for W2 computation
AVG_SIM = 10           # number of independent simulation runs to average over

parameters = get_params(L, dy)  # OTF network hyperparameters

# --- Generate Trajectories ---

t      = np.arange(0.0, tau*N, tau)    # time grid
X_True = np.zeros((AVG_SIM, N, L))    # true state trajectories
Y_True = np.zeros((AVG_SIM, N, dy))   # observation trajectories
X0     = np.zeros((AVG_SIM, L, J))    # EnKF initial particle ensembles
X0_sir = np.zeros((AVG_SIM, L, J_sir))# SIR initial particle ensembles

for k in range(AVG_SIM):
    x, y       = Gen_True_Data(L, dy, N, x0_amp, sigmma0, sigmma, gamma, tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,]     = np.transpose(np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0 * np.eye(L), J))
    X0_sir[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0 * np.eye(L), J_sir))


# --- Run Filters ---

# data shape: AVG_SIM x N x L x J
total_start = time.time()
X_EnKF      = EnKF(Y_True, X0, L63, h, t, tau, Noise, Odeint)
X_SIR       = SIR(Y_True, X0, L63, h, t, tau, Noise, Odeint)
with ThreadPoolExecutor(max_workers=3) as executor:
    future_reg = executor.submit(OTF, Y_True, X0, parameters, L63, h, t, tau, Noise, Odeint, delta,  device1)
    future_ot  = executor.submit(OTF, Y_True, X0, parameters, L63, h, t, tau, Noise, Odeint, [0, 0], device2)
    X_OT_reg   = future_reg.result()
    X_OT       = future_ot.result()
print(f"--- Total time (all methods): {time.time() - total_start:.2f} seconds ---")


# # --- Plot Results ---

# p              = 100  # number of particles to plot per method
# num_plot_state = 1    # state index to visualize
# l              = 0    # simulation index to visualize

# plt.figure(figsize=(15, 10))

# plt.subplot(5, 1, 1)
# plt.plot(t, X_EnKF[l,:,num_plot_state,:p], 'g', ls='none', marker='o', ms=4, alpha=0.1)
# plt.plot(t, X_True[l,:,num_plot_state], 'k--', label='True state')
# plt.ylabel('EnKF')
# plt.legend()

# plt.subplot(5, 1, 2)
# plt.plot(t, X_SIR[l,:,num_plot_state,:p], 'b', ls='none', marker='o', ms=4, alpha=0.1)
# plt.plot(t, X_True[l,:,num_plot_state], 'k--')
# plt.ylabel('SIR')

# plt.subplot(5, 1, 3)
# plt.plot(t, X_OT[l,:,num_plot_state,:p], 'r', ls='none', marker='o', ms=4, alpha=0.1)
# plt.plot(t, X_True[l,:,num_plot_state], 'k--')
# plt.ylabel('OT')

# plt.subplot(5, 1, 4)
# plt.plot(t, X_OT_reg[l,:,num_plot_state,:p], 'm', ls='none', marker='o', ms=4, alpha=0.1)
# plt.plot(t, X_True[l,:,num_plot_state], 'k--')
# plt.ylabel('OT_reg')
# plt.xlabel('time')

# plt.savefig(f'L63_EnKF_SIR_OT_OTreg_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)


# --- Save Data ---

np.savez('DATA_file_L63.npz',
    randint=randint, t=t, Noise=Noise, tau=tau, Odeint=Odeint, sigmma0=sigmma0,
    X0=X0, Y_true=Y_True, X_true=X_True, X_OT=X_OT, X_OT_reg=X_OT_reg)

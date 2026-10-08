"""
@author: Mohammad Al-Jarrah
"""

# --- Imports ---

import numpy as np
import torch
import time
import ot
import subprocess
from concurrent.futures import ThreadPoolExecutor

from EnKF import EnKF
from SIR import SIR
from OTF import OTF
from param import get_params


# --- Reproducibility ---

randint = np.random.randint(0, 1000)  # random seed drawn at runtime; printed and saved so a run can be reproduced
print(randint)
np.random.seed(randint)
torch.manual_seed(randint)


# --- GPU Selection ---

def get_free_gpu(n=1):
    """Return the indices of the n GPUs with the least memory currently in use."""
    result      = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True, text=True
    )
    memory_used = [int(x) for x in result.stdout.strip().split("\n")]
    sorted_ids  = sorted(range(len(memory_used)), key=lambda i: memory_used[i])
    return sorted_ids[:n]

if torch.cuda.is_available():
    gpu_ids = get_free_gpu(n=2)  # the two OTF runs (λ=0 and λ=0.1) go on separate GPUs
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


# --- Model Definition ---

def h(x):
    """Map the full state vector to the observed component."""
    return x[2,].reshape(dy, -1)  # only the third state x3 is observed, (dy, particles)

def L63(t, x):
    """Evaluate the Lorenz 63 vector field at state x and time t."""
    d     = np.zeros_like(x)  # output derivative vector
    sigma = 10                 # standard L63 coefficient
    r     = 28                 # standard L63 coefficient
    b     = 8/3                # standard L63 coefficient

    d[0] = sigma*(x[1] - x[0])
    d[1] = x[0]*(r - x[2]) - x[1]
    d[2] = x[0]*x[1] - b*x[2]
    return d


def rk4_step(f, t, x, tau):
    k1 = f(t,         x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau,   x + tau   * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)


def _timed(func, *args):
    t0 = time.time()
    return func(*args), time.time() - t0


# --- Data Generation ---

def Gen_True_Data(L, dy, N, gamma, tau):
    """
    Simulate a noisy Lorenz 63 trajectory and its partial observations.

    Returns:
        x (ndarray): state trajectory of shape (N, L)
        y (ndarray): observation trajectory of shape (N, dy)
    """
    eta = np.random.multivariate_normal(np.zeros(dy), gamma*gamma * np.eye(dy), N)  # observation noise samples

    x  = np.zeros((N, L))                                                # state trajectory
    y  = np.zeros((N, dy))                                               # observation trajectory
    x0 = 5 + np.random.multivariate_normal(np.zeros(L), np.eye(L), 1)  # perturbed initial condition

    x[0,] = x0

    # The true state evolves without process noise; rk4 and t are read from the module scope
    for i in range(N-1):
        if rk4:
            x[i+1,:] = rk4_step(L63, t[i], x[i,:], tau)
        else:
            x[i+1,:] = x[i,:] + L63(t[i], x[i,:]) * tau
        y[i+1,] = h(x[i+1,]) + eta[i+1,]  # y[0] stays 0: there is no observation at t = 0

    return x, y


# --- Metrics ---

def w2_distance(x, y):
    """Compute the W2 distance between two empirical distributions in R^L."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.ones(x.shape[0]) / x.shape[0]    # uniform weights over x samples
    b = np.ones(y.shape[0]) / y.shape[0]    # uniform weights over y samples
    M = ot.dist(x, y, metric='sqeuclidean')  # squared-Euclidean cost matrix
    # The default iteration cap (1e5) can stop the network simplex before the optimum
    return np.sqrt(ot.emd2(a, b, M, numItermax=int(1e7)))


# --- Simulation Parameters ---

L   = 3           # state dimension
tau = 1e-2        # time step size
T   = 5           # final simulation time (seconds)
N   = int(T/tau)  # total number of time steps

dy = 1  # number of observed states

noise   = np.sqrt(1e1)     # noise level standard deviation
sigmma  = noise/10         # process noise std for the hidden state
sigmma0 = noise**2         # standard deviation of the initial state distribution (the covariance is sigmma0^2 I)
gamma   = noise/1          # observation noise std
Noise   = [sigmma, gamma]  # packed noise vector passed to filters
rk4     = True*0           # integration scheme: 0 -> forward Euler, 1 -> RK4 fixed step

delta   = [0.1, 0.1]   # OTF regularization weights [lambda_T, lambda_f]
J       = int(1000/1)  # ensemble size, shared by all filters
AVG_SIM = 10           # number of independent simulation runs to average over

p_true        = int(1e3)  # reference particles used to compute W2
true_particle = int(1e5)  # large-SIR ensemble size for the true distribution
ref_seed      = 0         # seed of the reference SIR (initial ensemble, process noise, resampling)
sim           = 0         # simulation index whose reference is saved for the density heatmaps

parameters = get_params(L, dy)  # OTF network hyperparameters

# --- Generate Trajectories ---

t      = np.arange(0.0, tau*N, tau)    # time grid
X_True = np.zeros((AVG_SIM, N, L))    # true state trajectories
Y_True = np.zeros((AVG_SIM, N, dy))   # observation trajectories
X0     = np.zeros((AVG_SIM, L, J))    # initial particle ensembles, shared by all filters

for k in range(AVG_SIM):
    x, y       = Gen_True_Data(L, dy, N, gamma, tau)
    X_True[k,] = x
    Y_True[k,] = y
    X0[k,]     = np.transpose(np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0 * np.eye(L), J))  # prior N(0, sigmma0^2 I), (L, J)


# --- Run Filters ---

# data shape: AVG_SIM x N x L x J
X_EnKF, time_EnKF = _timed(EnKF, Y_True, X0, L63, h, t, tau, Noise, rk4)
X_SIR,  time_SIR  = _timed(SIR,  Y_True, X0, L63, h, t, tau, Noise, rk4)
# The two OTF runs are independent, so they train concurrently on the two devices
with ThreadPoolExecutor(max_workers=2) as executor:
    future_otf_reg = executor.submit(_timed, OTF, Y_True, X0, parameters, L63, h, t, tau, Noise, rk4, delta,  device1)  # OTF (λ=0.1)
    future_otf     = executor.submit(_timed, OTF, Y_True, X0, parameters, L63, h, t, tau, Noise, rk4, [0, 0], device2)  # OTF (λ=0)
    X_OTF_reg, time_OTF_reg = future_otf_reg.result()
    X_OTF,     time_OTF     = future_otf.result()

print(f"EnKF        time: {time_EnKF:.2f}s")
print(f"SIR         time: {time_SIR:.2f}s")
print(f"OTF (λ=0)   time: {time_OTF:.2f}s")
print(f"OTF (λ=0.1) time: {time_OTF_reg:.2f}s")


# --- W2 against a Large-SIR Reference ---

np.random.seed(ref_seed)                   # SIR draws its process noise from np.random
rng_ref = np.random.default_rng(ref_seed)  # large-SIR initial ensembles

methods = {            # filter name -> particles, shape (AVG_SIM, N, L, J)
    'EnKF':        X_EnKF,
    'SIR':         X_SIR,
    'OTF (λ=0)':   X_OTF,
    'OTF (λ=0.1)': X_OTF_reg,
}
distance_W2 = {name: np.zeros((N, AVG_SIM)) for name in methods}  # W2 array: (time steps, simulations)

for k in range(AVG_SIM):
    # The reference starts from the same prior N(0, sigmma0^2 I) as the filters
    X0_true     = rng_ref.multivariate_normal(np.zeros(L), sigmma0*sigmma0 * np.eye(L), true_particle).T  # (L, true_particle)
    X_true_dist = SIR(Y_True[k,].reshape(1, N, dy), X0_true.reshape(1, L, true_particle),
                      L63, h, t, tau, Noise, rk4, seed=ref_seed + k)                                   # (1, N, L, true_particle)
    if k == sim:
        X_true_dist_sim = X_true_dist  # kept for the heatmaps, so they match the W2 values
    print(f"Computing W2 — simulation {k+1}/{AVG_SIM}", flush=True)

    for i in range(N):
        # Particles are stored as (state, particle); the W2 takes (particle, state) clouds
        ref_T = X_true_dist[0, i, :, :p_true].T  # (p_true, L) reference particles
        for name, X in methods.items():
            distance_W2[name][i, k] = w2_distance(ref_T, X[k, i].T)


# --- Save Data ---

np.savez_compressed('DATA_file_L63.npz',
    randint=randint, t=t, Noise=Noise, tau=tau, rk4=rk4, sigmma0=sigmma0,
    X0=X0, Y_true=Y_True, X_true=X_True,
    X_EnKF=X_EnKF, time_EnKF=time_EnKF,
    X_SIR=X_SIR,   time_SIR=time_SIR,
    X_OTF=X_OTF,         time_OTF=time_OTF,
    X_OTF_reg=X_OTF_reg, time_OTF_reg=time_OTF_reg,
    w2_EnKF=distance_W2['EnKF'],     w2_SIR=distance_W2['SIR'],              # W2 per time step and simulation, (N, AVG_SIM)
    w2_OTF=distance_W2['OTF (λ=0)'], w2_OTF_reg=distance_W2['OTF (λ=0.1)'],
    X_true_dist=X_true_dist_sim[0],                                          # large-SIR reference of simulation `sim`, (N, L, true_particle)
    sim=sim, p_true=p_true, true_particle=true_particle, ref_seed=ref_seed)

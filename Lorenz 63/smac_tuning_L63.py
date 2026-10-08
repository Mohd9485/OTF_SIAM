"""
@author: Mohammad Al-Jarrah

SMAC hyperparameter tuning for the OTF filter on the Lorenz 63 model.

Metric: W2 distance between OTF particles and a large-particle SIR reference,
        averaged over time steps and AVG_SIM independent runs per config.

The tuning runs OTF without regularization (delta = [0, 0], i.e. OTF (λ=0)).
Run directly with `python smac_tuning_L63.py`; results are written to smac3_output/.
"""

# --- Imports ---

import os
import time
import numpy as np
import torch
import ot
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from smac import Scenario, HyperparameterOptimizationFacade as HPO

from OTF import OTF
from SIR import SIR


# --- Device Setup ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Parallel Workers ---

WORKERS_PER_GPU = 3                                                     # SMAC workers sharing each GPU
N_WORKERS       = WORKERS_PER_GPU * max(torch.cuda.device_count(), 1)  # total workers (3 on a CPU-only machine)
N_THREADS       = max(1, (os.cpu_count() or 1) // N_WORKERS)            # CPU threads per worker, avoids oversubscription


# --- Problem Setup ---

L_state = 3                # state dimension
tau     = 1e-2             # time step size
T_end   = 1                # final simulation time (seconds)
N       = int(T_end / tau)  # total number of time steps
dy      = 1                # number of observed states (only x3 is observed, see h)
J       = int(1000 / 4)    # OTF ensemble size (smaller than in main.py to keep each trial cheap)
J_sir   = 100_000          # large-SIR reference particle count
p_true  = 1000             # SIR reference particles used to compute W2 (subset of the J_sir)
AVG_SIM = 2                # independent runs averaged per SMAC trial

delta   = [0, 0]           # OTF regularization weights [lambda_T, lambda_f]; zero, so this tunes OTF (λ=0)

noise   = np.sqrt(1e1)     # noise level standard deviation
sigmma  = noise / 10       # process noise std for the hidden state
sigmma0 = noise ** 2       # std of the initial state distribution (its variance is sigmma0**2)
gamma   = noise / 1        # observation noise std
x0_amp  = 1                # initial state amplitude scaling factor
Noise   = [sigmma, gamma]  # packed noise vector passed to filters
rk4     = False            # False: forward Euler steps; True: RK4 fixed-step integration

t = np.arange(0.0, tau * N, tau)  # time grid, shape (N,)


# --- Model Definition ---

def h(x):
    """Map the full state vector to the observed component."""
    return x[2, ].reshape(dy, -1)  # observe the third state x3, shape (dy, particles)


def L63(t, x):
    """Evaluate the Lorenz 63 vector field at state x and time t."""
    d           = np.zeros_like(x)  # output derivative vector
    sigma, r, b = 10, 28, 8 / 3     # standard L63 coefficients
    d[0] = sigma * (x[1] - x[0])
    d[1] = x[0] * (r - x[2]) - x[1]
    d[2] = x[0] * x[1] - b * x[2]
    return d


def rk4_step(f, t, x, tau):
    k1 = f(t,         x)
    k2 = f(t + tau/2, x + tau/2 * k1)
    k3 = f(t + tau/2, x + tau/2 * k2)
    k4 = f(t + tau,   x + tau   * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)


# --- Data Generation ---

def gen_data():
    """
    Generate a true trajectory, observations, and initial ensembles for OTF and SIR.

    Returns:
        Y_True  (ndarray): observations, shape (1, N, dy)
        X0_ot   (ndarray): OTF initial particles, shape (1, L, J)
        X0_sir  (ndarray): SIR initial particles, shape (1, L, J_sir)
    """
    eta  = np.random.multivariate_normal(np.zeros(dy), gamma ** 2 * np.eye(dy), N)  # observation noise samples
    x    = np.zeros((N, L_state))                                                    # state trajectory
    y    = np.zeros((N, dy))                                                         # observation trajectory
    x[0] = 5 + np.random.multivariate_normal(np.zeros(L_state), np.eye(L_state), 1)  # perturbed initial condition

    for i in range(N - 1):
        if rk4:
            x[i + 1] = rk4_step(L63, t[i], x[i], tau)
        else:
            x[i + 1] = x[i] + L63(t[i], x[i]) * tau
        y[i + 1] = h(x[i + 1]) + eta[i + 1]  # y[0] stays zero: no observation at t = 0

    Y_True = y[np.newaxis]  # (1, N, dy)

    X0_ot = np.transpose(
        np.random.multivariate_normal(
            np.zeros(L_state), sigmma0 ** 2 * np.eye(L_state), J
        )
    )[np.newaxis]  # (1, L, J); same prior N(0, sigmma0^2 I) as the SIR reference

    X0_sir = np.transpose(
        np.random.multivariate_normal(
            np.zeros(L_state), sigmma0 ** 2 * np.eye(L_state), J_sir
        )
    )[np.newaxis]  # (1, L, J_sir)

    return Y_True, X0_ot, X0_sir


# --- Shared Data ---

# The settings are part of the file name, so changing J, p_true or AVG_SIM builds a new cache
_SHARED_DATA_FILE = f"smac_shared_data_L63_J{J}_p{p_true}_avg{AVG_SIM}.npz"

# Lazy cache: each worker process loads the shared data once on first call
_cache = {}


def build_shared_data():
    """
    Generate the AVG_SIM datasets and their large-SIR references once and save them.

    Every SMAC trial then filters the same observations from the same initial
    ensembles, so configs are compared on identical data (a paired comparison), and
    the expensive J_sir-particle SIR is run AVG_SIM times in total instead of in every
    trial. Only the first p_true reference particles are stored, since only those
    enter the W2 computation.
    """
    np.random.seed(0)  # fixed, so the shared datasets are reproducible regardless of the SMAC seed
    Y_all, X0_all, ref_all = [], [], []
    for sim_idx in range(AVG_SIM):
        sim_start = time.time()
        Y_True, X0_ot, X0_sir = gen_data()
        X_sir_ref = SIR(Y_True, X0_sir, L63, h, t, tau, Noise, rk4)
        Y_all.append(Y_True)
        X0_all.append(X0_ot)
        ref_all.append(X_sir_ref[:, :, :, :p_true])  # keep only the p_true particles used by W2
        print(f"  reference {sim_idx + 1}/{AVG_SIM} built in {time.time() - sim_start:.1f}s", flush=True)

    np.savez(_SHARED_DATA_FILE,
             Y_True    = np.stack(Y_all),    # (AVG_SIM, 1, N, dy)
             X0_ot     = np.stack(X0_all),   # (AVG_SIM, 1, L, J)
             X_sir_ref = np.stack(ref_all))  # (AVG_SIM, 1, N, L, p_true)


def _load_data():
    """Load the shared datasets and SIR references from disk, caching after first read."""
    if not _cache:
        _cache.update(dict(np.load(_SHARED_DATA_FILE)))
    return _cache


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


def w2_vs_sir(X_filt, X_sir_ref):
    """
    Compute the time-averaged W2 distance between OTF particles and a large-SIR reference.

    Uses the first p_true SIR particles as the reference at each time step.

    Args:
        X_filt    (ndarray): OTF filtered particles, shape (1, N, L, J)
        X_sir_ref (ndarray): SIR reference particles, shape (1, N, L, p_true) or (1, N, L, J_sir)

    Returns:
        float: time-averaged W2 distance
    """
    total_w2 = 0.0
    for n in range(1, N):                          # skip t=0: every filter still holds the prior there
        # Particles are stored as (state, particle); the W2 takes (particle, state) clouds
        ot_pts    = X_filt[0, n].T                 # (J, L)
        sir_pts   = X_sir_ref[0, n, :, :p_true].T  # (p_true, L)
        total_w2 += w2_distance(ot_pts, sir_pts)
    return total_w2 / (N - 1)


# --- SMAC Configuration Space ---

cs = ConfigurationSpace(seed=42)
cs.add([
    Float(      "lr",         (5e-5, 5e-3), log=True, default=5e-4),  # learning rate search range
    Integer(    "num_neuron", (1, 8),        default=1),               # x32  → hidden width 32–256
    Integer(    "batch_size", (1, 4),        default=2),               # x32  → batch size 32–128
    Categorical("iteration",  [1, 3],        default=3),               # x512 → training iterations
])


# --- SMAC Target Function ---

def target_fun(config, seed: int = 0) -> float:
    """
    Evaluate a hyperparameter configuration and return the average W2 loss.

    Runs AVG_SIM independent simulations per config, comparing OTF filtered
    particles against a large-SIR reference using W2 distance.

    Args:
        config: SMAC configuration object with hyperparameter values
        seed (int): random seed for reproducibility across trials

    Returns:
        float: mean W2 distance (lower is better); returns 1e6 on failure or divergence
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Assign each Dask worker to its own GPU slot to avoid memory contention
    local_device = device
    if device.type == 'cuda':
        try:
            from dask.distributed import get_worker
            worker    = get_worker()
            name      = str(worker.name)                                                          # worker name as string
            worker_id = int(name) if name.isdigit() else hash(name) % torch.cuda.device_count()  # map worker name to GPU index
        except Exception:
            worker_id = 0  # not running under a Dask worker (e.g. the final validation call)
        local_device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")
        # Make it the current device, so implicit allocations never land on cuda:0
        torch.cuda.set_device(local_device)
    torch.set_num_threads(N_THREADS)

    # Load shared data (each worker process reads once, then caches in memory)
    shared = _load_data()

    parameters = {
        'normalization':          'None',                           # no input normalization applied
        'INPUT_DIM':              [L_state, dy],                    # network input dimensions: [state dim, observation dim]
        'NUM_NEURON':             int(config["num_neuron"]) * 32,   # hidden layer width
        'BATCH_SIZE':             int(config["batch_size"]) * 32,   # mini-batch size
        'LearningRate':           float(config["lr"]),              # optimizer step size
        'ITERATION':              int(config["iteration"]) * 512,   # total training iterations
        'Final_Number_ITERATION': 64,                               # iterations for the final refinement stage
    }

    print(
        f"\n[SMAC] seed={seed}  {time.strftime('%H:%M:%S')}\n"
        f"  neurons={parameters['NUM_NEURON']}, batch={parameters['BATCH_SIZE']}, "
        f"lr={parameters['LearningRate']:.2e}, iters={parameters['ITERATION']}, "
        f"final_iter={parameters['Final_Number_ITERATION']}, "
        f"delta=[{delta[0]:.3f},{delta[1]:.3f}]",
        flush=True,
    )

    trial_start = time.time()  # wall-clock start for the full trial
    try:
        total_w2 = 0.0  # accumulated W2 across simulations
        for sim_idx in range(AVG_SIM):
            sim_start = time.time()  # wall-clock start for this simulation

            Y_True    = shared['Y_True'][sim_idx]     # (1, N, dy)
            X0_ot     = shared['X0_ot'][sim_idx]      # (1, L, J)
            X_sir_ref = shared['X_sir_ref'][sim_idx]  # (1, N, L, p_true)

            X_filt    = OTF(Y_True, X0_ot, parameters, L63, h,
                            t, tau, Noise, rk4, delta, local_device)  # (1, N, L, J)
            # ot.emd2 does not propagate NaN/inf -- it warns and returns a finite, wrong
            # value -- so a diverged filter must be caught on the particles themselves
            if not np.isfinite(X_filt).all():
                raise FloatingPointError("OTF returned non-finite particles (diverged)")
            sim_w2    = w2_vs_sir(X_filt, X_sir_ref)  # W2 for this simulation run
            total_w2 += sim_w2
            print(
                f"[SMAC] seed={seed}  sim {sim_idx+1}/{AVG_SIM}  "
                f"W2={sim_w2:.4f}  time={time.time()-sim_start:.1f}s",
                flush=True,
            )

        loss = total_w2 / AVG_SIM  # mean W2 across AVG_SIM runs

        # A diverged filter returns NaN/inf particles without raising, so catch it here
        if not np.isfinite(loss):
            print(f"[SMAC] Trial diverged: W2={loss}", flush=True)
            loss = 1e6  # penalise diverged runs

    except Exception as e:
        print(f"[SMAC] Trial failed: {e}", flush=True)
        loss = 1e6  # penalise failed runs, so SMAC steers away from them

    finally:
        if local_device.type == "cuda":
            torch.cuda.empty_cache()  # free cached GPU memory for the next trial on this worker

    print(f"[SMAC] config={dict(config)}  W2={loss:.6f}  "
          f"total={time.time()-trial_start:.1f}s", flush=True)
    return loss


# --- Run SMAC Optimisation ---

if __name__ == "__main__":
    # Build and save shared data once — worker processes load from disk
    if not os.path.exists(_SHARED_DATA_FILE):
        print(f"Building {AVG_SIM} datasets and SIR references ({J_sir} particles)...", flush=True)
        build_shared_data()
        print(f"Saved shared data to {_SHARED_DATA_FILE}")
    else:
        print(f"Shared data found at {_SHARED_DATA_FILE}, skipping rebuild.")

    scenario = Scenario(
        configspace   = cs,
        name          = f"OT_L63_W2_J{J}_p{p_true}",  # SMAC output folder; keep fixed so runs resume (renamed from OT_L63 when the metric became W2)
        deterministic = False,                         # OTF training is stochastic, so SMAC may re-evaluate a config with new seeds
        n_trials      = 200,                           # total config evaluations
        n_workers     = N_WORKERS,
        seed          = 42,
    )

    smac = HPO(
        scenario        = scenario,
        target_function = target_fun,
        overwrite       = False,  # resume an existing run with the same name instead of starting over
    )

    print(f"\n=== Starting SMAC tuning for OTF on Lorenz 63 ===")
    print(f"Using device: {device}")
    print(f"Using n_workers={scenario.n_workers} for parallel evaluation.\n")

    try:
        incumbent       = smac.optimize()                 # best config found by SMAC
        validation_loss = target_fun(incumbent, seed=42)  # re-evaluated W2 for the best config
    finally:
        try:
            smac._runner.close()  # shut down the Dask workers even if the optimization failed
        except Exception:
            pass

    print("\n=== Best configuration found ===")
    print(incumbent)
    print(f"Validation W2: {validation_loss:.6f}")

"""
@author: Mohammad Al-Jarrah

SMAC hyperparameter tuning for OT_reg filter on the Lorenz 63 model.

Metric: W2 distance between OT_reg particles and a large-particle SIR reference,
        averaged over time steps and AVG_SIM independent runs per config.
"""

# --- Imports ---

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


# --- Problem Setup ---

L_state = 3                # state dimension
tau     = 1e-2             # time step size
T_end   = 1                # final simulation time (seconds)
N       = int(T_end / tau) # total number of time steps
dy      = 1                # number of observed states
J       = int(1000 / 4)    # OT_reg ensemble size
J_sir   = 100_000          # large-SIR reference particle count
p_true  = 1000             # SIR particles used to compute W2
AVG_SIM = 2                # independent runs averaged per SMAC trial

delta   = [0, 0]           # OTF regularization weights [lambda_T, lambda_f]

noise   = np.sqrt(1e1)     # noise level standard deviation
sigmma  = noise / 10       # process noise std for the hidden state
sigmma0 = noise ** 2       # variance of the initial state distribution
gamma   = noise / 1        # observation noise std
x0_amp  = 1                # initial state amplitude scaling factor
Noise   = [sigmma, gamma]  # packed noise vector passed to filters
rk4     = False            # use RK4 fixed-step integration

t = np.arange(0.0, tau * N, tau)  # time grid


# --- Model Definition ---

def h(x):
    """Map the full state vector to the observed component."""
    return x[2, ].reshape(dy, -1)


def L63(t, x):
    """Evaluate the Lorenz 63 vector field at state x and time t."""
    d            = np.zeros_like(x)  # output derivative vector
    sigma, r, b  = 10, 28, 8 / 3    # standard L63 coefficients
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
    Generate a true trajectory, observations, and initial ensembles for OT_reg and SIR.

    Returns:
        Y_True  (ndarray): observations, shape (1, N, dy)
        X0_ot   (ndarray): OT_reg initial particles, shape (1, L, J)
        X0_sir  (ndarray): SIR initial particles, shape (1, L, J_sir)
    """
    eta  = np.random.multivariate_normal(np.zeros(dy), gamma ** 2 * np.eye(dy), N)  # observation noise samples
    x    = np.zeros((N, L_state))                                                    # state trajectory
    y    = np.zeros((N, dy))                                                         # observation trajectory
    x[0] = 5 + np.random.multivariate_normal(np.zeros(L_state), np.eye(L_state), 1) # perturbed initial condition

    for i in range(N - 1):
        if rk4:
            x[i + 1] = rk4_step(L63, t[i], x[i], tau)
        else:
            x[i + 1] = x[i] + L63(t[i], x[i]) * tau
        y[i + 1]  = h(x[i + 1]) + eta[i + 1]

    Y_True = y[np.newaxis]  # (1, N, dy)

    X0_ot = np.transpose(
        np.random.multivariate_normal(
            np.zeros(L_state), sigmma0 ** 2 * np.eye(L_state), J
        )
    )[np.newaxis]  # (1, L, J)

    X0_sir = np.transpose(
        np.random.multivariate_normal(
            np.zeros(L_state), sigmma0 ** 2 * np.eye(L_state), J_sir
        )
    )[np.newaxis]  # (1, L, J_sir)

    return Y_True, X0_ot, X0_sir


# --- Metrics ---

def w2_distance(x, y):
    """Compute the W2 distance between two empirical distributions in R^L."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.ones(x.shape[0]) / x.shape[0]    # uniform weights over x samples
    b = np.ones(y.shape[0]) / y.shape[0]    # uniform weights over y samples
    M = ot.dist(x, y, metric='sqeuclidean')  # squared-Euclidean cost matrix
    return np.sqrt(ot.emd2(a, b, M))


def w2_vs_sir(X_filt, X_sir_ref):
    """
    Compute the time-averaged W2 distance between OT_reg particles and a large-SIR reference.

    Uses the first p_true SIR particles as the reference at each time step.

    Args:
        X_filt    (ndarray): OT_reg filtered particles, shape (1, N, L, J)
        X_sir_ref (ndarray): SIR reference particles, shape (1, N, L, J_sir)

    Returns:
        float: time-averaged W2 distance
    """
    total_w2 = 0.0
    for n in range(1, N):                          # skip t=0 (prior, identical)
        ot_pts  = X_filt[0, n].T                   # (J, L)
        sir_pts = X_sir_ref[0, n, :, :p_true].T    # (p_true, L)
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

    Runs AVG_SIM independent simulations per config, comparing OT_reg filtered
    particles against a large-SIR reference using W2 distance.

    Args:
        config: SMAC configuration object with hyperparameter values
        seed (int): random seed for reproducibility across trials

    Returns:
        float: mean W2 distance (lower is better); returns 1e6 on failure
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Assign each Dask worker to its own GPU slot to avoid memory contention
    local_device = device
    if device.type == 'cuda':
        try:
            from dask.distributed import get_worker
            worker    = get_worker()
            name      = str(worker.name)                                                            # worker name as string
            worker_id = int(name) if name.isdigit() else hash(name) % torch.cuda.device_count()    # map worker name to GPU index
        except Exception:
            worker_id = 0
        local_device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")

    parameters = {
        'normalization':          'None',                           # no input normalization applied
        'INPUT_DIM':              [L_state, dy],                    # model input shape: [lag, obs_dim]
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
        total_w2 = 0.0         # accumulated W2 across simulations
        for sim_idx in range(AVG_SIM):
            sim_start = time.time()  # wall-clock start for this simulation

            Y_True, X0_ot, X0_sir = gen_data()

            X_sir_ref = SIR(Y_True, X0_sir, L63, h, t, tau, Noise, rk4)
            X_filt    = OTF(Y_True, X0_ot, parameters, L63, h,
                            t, tau, Noise, rk4, delta, local_device)
            sim_w2    = w2_vs_sir(X_filt, X_sir_ref)  # W2 for this simulation run
            total_w2 += sim_w2
            print(
                f"[SMAC] seed={seed}  sim {sim_idx+1}/{AVG_SIM}  "
                f"W2={sim_w2:.4f}  time={time.time()-sim_start:.1f}s",
                flush=True,
            )

        loss = total_w2 / AVG_SIM  # mean W2 across AVG_SIM runs

    except Exception as e:
        print(f"[SMAC] Trial failed: {e}", flush=True)
        loss = 1e6

    finally:
        if local_device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"[SMAC] config={dict(config)}  W2={loss:.6f}  "
          f"total={time.time()-trial_start:.1f}s", flush=True)
    return loss


# --- Run SMAC Optimisation ---

if __name__ == "__main__":
    scenario = Scenario(
        configspace   = cs,
        name          = "OT_L63",
        deterministic = True,
        n_trials      = 200,
        n_workers     = 3,    # keep at 1 when using GPU to avoid memory contention
        seed          = 42,
    )

    smac = HPO(
        scenario        = scenario,
        target_function = target_fun,
        overwrite       = False,
    )

    print(f"\n=== Starting SMAC tuning for OT_reg on Lorenz 63 ===")
    print(f"Using device: {device}\n")

    try:
        incumbent       = smac.optimize()                 # best config found by SMAC
        validation_loss = target_fun(incumbent, seed=42)  # re-evaluated W2 for the best config
    finally:
        try:
            smac._runner.close()
        except Exception:
            pass

    print("\n=== Best configuration found ===")
    print(incumbent)
    print(f"Validation W2: {validation_loss:.6f}")

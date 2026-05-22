"""
@author: Mohammad Al-Jarrah

SMAC hyperparameter tuning for OT filter on the Lorenz 96 model.

Metric: MSE between OT particle mean and true trajectory,
        averaged over time steps and AVG_SIM independent runs per config.
"""

import time
import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from smac import Scenario, HyperparameterOptimizationFacade as HPO

from OTF import OTF
# from OTF_EnKF import OTF_EnKF as OTF


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Problem Setup ---
L               = 9               # state space dimension
F               = 10              # forcing constant for the L96 model
tau             = 0.01            # time step size
T_end           = 1.0             # total simulation time
N               = int(T_end/tau)  # number of time steps
dy              = 3               # observation space dimension
observation_idx = [0, 3, 6]       # indices of observed state components
J               = 250 * 4         # ensemble size for OT_reg filter
AVG_SIM         = 2               # number of independent runs averaged per SMAC trial

delta = [0, 0]  # regularization weights (delta_T, delta_f)

noise   = 0.1            # std of process and observation noise
sigmma  = noise          # std of process noise in the hidden state
sigmma0 = np.sqrt(1e1)   # std of noise in the initial state distribution
gamma   = noise          # std of observation noise
x0_amp  = 1              # amplitude scaling applied to the initial state
Noise   = [sigmma, gamma]

rk4 = True  # use RK4 integrator for data generation and filtering

t = np.arange(0.0, tau * N, tau)  # time grid, shape (N,)


def rk4_step(f, t, x, tau):
    k1 = f(t,          x)
    k2 = f(t + tau/2,  x + tau/2 * k1)
    k3 = f(t + tau/2,  x + tau/2 * k2)
    k4 = f(t + tau,    x + tau   * k3)
    return x + tau/6 * (k1 + 2*k2 + 2*k3 + k4)


# --- Model and Observation Functions ---

def h(x):
    """Observation operator: extract observed components from state x."""
    return x[observation_idx,]


def L96(t, x):
    """
    Lorenz 96 vector field for a single state vector.

    Parameters
    ----------
    t : float    — current time (unused, required by RK4 interface)
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
    t : float              — current time (unused, required by RK4 interface)
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


# --- Data Generation ---

def gen_data():
    """
    Generate one realization of the true trajectory, observations, and initial ensemble.

    Returns
    -------
    Y_True : ndarray, shape (1, N, dy) — noisy observations
    X0_ot  : ndarray, shape (1, L, J)  — initial ensemble drawn from prior
    X_True : ndarray, shape (1, N, L)  — true hidden state trajectory
    """
    sai = np.random.multivariate_normal(np.zeros(L), sigmma**2 * np.eye(L), N)   # process noise samples
    eta = np.random.multivariate_normal(np.zeros(dy), gamma**2 * np.eye(dy), N)  # observation noise samples

    x    = np.zeros((N, L))   # true state trajectory
    y    = np.zeros((N, dy))  # noisy observations
    x[0] = 10 + x0_amp * np.random.multivariate_normal(np.zeros(L), sigmma0**2 * np.eye(L), 1)

    for i in range(N - 1):
        x[i + 1] = rk4_step(L96, t[i], x[i], tau) + sai[i]
        y[i + 1] = h(x[i + 1]) + eta[i + 1]

    Y_True = y[np.newaxis]  # add simulation axis, shape (1, N, dy)
    X_True = x[np.newaxis]  # add simulation axis, shape (1, N, L)

    X0_ot = 10 + x0_amp * np.transpose(
        np.random.multivariate_normal(np.zeros(L), sigmma0**2 * np.eye(L), J)
    )[np.newaxis]            # initial ensemble from prior, shape (1, L, J)

    return Y_True, X0_ot, X_True


# --- Evaluation Metric ---

def mse_score(X_filt, X_true):
    """
    Compute scalar MSE between OT_reg particle mean and the true trajectory.

    Parameters
    ----------
    X_filt : ndarray, shape (AVG_SIM, N, L, J) — filtered ensemble
    X_true : ndarray, shape (AVG_SIM, N, L)    — true state trajectory

    Returns
    -------
    float — mean squared error averaged over time steps and simulations
    """
    x_mean = X_filt.mean(axis=3)                   # ensemble mean, shape (AVG_SIM, N, L)
    sq_err = ((x_mean - X_true) ** 2).sum(axis=2)  # squared error per step, shape (AVG_SIM, N)
    return float(sq_err.mean())                     # scalar average


# --- SMAC Configuration Space ---

cs = ConfigurationSpace(seed=42)
cs.add([
    Float(  "lr_f",             (5e-5, 5e-3), log=True, default=1e-3),
    Float(  "lr_T",             (5e-5, 5e-3), log=True, default=1e-3),
    Integer("num_neuron_f",     (1, 12),               default=8),   # x32 → 32–384 neurons
    Integer("num_neuron_T",     (1, 12),               default=5),   # x32 → 32–384 neurons
    Integer("batch_size",       (1, 8),                default=4),   # x32 → 32–256 samples
    Categorical("iteration",    [1, 3, 7],                default=3),   # x512 → 512–1536 iterations
    Integer("inner_iterations", (1, 4),                default=2),
])


# --- SMAC Target Function ---

def target_fun(config, seed: int = 0) -> float:
    """
    Evaluate a hyperparameter configuration by running the OTF filter and returning the MSE.

    Supports multi-GPU environments via Dask worker assignment.

    Parameters
    ----------
    config : Configuration — SMAC hyperparameter configuration to evaluate
    seed   : int           — random seed for reproducibility

    Returns
    -------
    float — average MSE across AVG_SIM runs (returns 1e6 on failure)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    local_device = device
    if device.type == "cuda":
        try:
            from dask.distributed import get_worker
            worker    = get_worker()
            name      = str(worker.name)
            worker_id = int(name) if name.isdigit() else hash(name) % torch.cuda.device_count()
        except Exception:
            worker_id = 0
        local_device = torch.device(f"cuda:{worker_id % torch.cuda.device_count()}")

    parameters = {
        "normalization":          "None",
        "INPUT_DIM":              [L, dy],
        "NUM_NEURON":             [int(config["num_neuron_f"]) * 32,
                                   int(config["num_neuron_T"]) * 32],
        "BATCH_SIZE":             int(config["batch_size"]) * 32,
        "LearningRate":           [float(config["lr_f"]), float(config["lr_T"])],
        "ITERATION":              int(config["iteration"]) * 512,
        "Final_Number_ITERATION": 64,
        "inner_iterations":       int(config["inner_iterations"] * 5),
    }

    print(
        f"\n[SMAC] seed={seed}  {time.strftime('%H:%M:%S')}\n"
        f"  neurons_f={parameters['NUM_NEURON'][0]}, neurons_T={parameters['NUM_NEURON'][1]}, "
        f"batch={parameters['BATCH_SIZE']}, "
        f"lr_f={parameters['LearningRate'][0]:.2e}, lr_T={parameters['LearningRate'][1]:.2e}, "
        f"iters={parameters['ITERATION']}, inner_iters={parameters['inner_iterations']}, "
        f"final_iter={parameters['Final_Number_ITERATION']}, "
        f"delta=[{delta[0]:.3f},{delta[1]:.3f}]",
        flush=True,
    )

    trial_start = time.time()
    all_X_filt  = []  # filtered ensemble per simulation run
    all_X_true  = []  # true trajectory per simulation run

    try:
        for sim_idx in range(AVG_SIM):
            sim_start = time.time()

            Y_True, X0_ot, X_True = gen_data()

            X_filt = OTF(Y_True, X0_ot, parameters, ML96, h,
                         t, tau, Noise, rk4, delta, local_device)

            all_X_filt.append(X_filt)  # shape (1, N, L, J)
            all_X_true.append(X_True)  # shape (1, N, L)

            sim_mse = mse_score(X_filt, X_True)
            print(
                f"[SMAC] seed={seed}  sim {sim_idx + 1}/{AVG_SIM}  "
                f"MSE={sim_mse:.4f}  time={time.time() - sim_start:.1f}s",
                flush=True,
            )

        X_filt_all = np.concatenate(all_X_filt, axis=0)  # shape (AVG_SIM, N, L, J)
        X_true_all = np.concatenate(all_X_true, axis=0)  # shape (AVG_SIM, N, L)
        loss = mse_score(X_filt_all, X_true_all)

    except Exception as e:
        print(f"[SMAC] Trial failed: {e}", flush=True)
        loss = 1e6

    finally:
        if local_device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"[SMAC] config={dict(config)}  MSE={loss:.6f}  "
        f"total={time.time() - trial_start:.1f}s",
        flush=True,
    )
    return loss


# --- Run SMAC Optimisation ---

if __name__ == "__main__":
    scenario = Scenario(
        configspace   = cs,
        name          = "OTF_L96",
        deterministic = True,
        n_trials      = 200,
        n_workers     = 3,
        seed          = 42,
    )

    smac = HPO(
        scenario        = scenario,
        target_function = target_fun,
        overwrite       = False,
    )

    print(f"\n=== Starting SMAC tuning for OT_reg on Lorenz 96 ===")
    print(f"Using device: {device}\n")

    try:
        incumbent       = smac.optimize()
        validation_loss = target_fun(incumbent, seed=42)
    finally:
        try:
            smac._runner.close()
        except Exception:
            pass

    print("\n=== Best configuration found ===")
    print(incumbent)
    print(f"Validation MSE: {validation_loss:.6f}")

"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.close('all')
fontsize = 19  # global font size for all plot labels and legends
plt.rc('font', size=fontsize)
matplotlib.rcParams['pdf.fonttype'] = 42  # ensure editable fonts in PDF output
matplotlib.rcParams['ps.fonttype']  = 42  # same for PostScript output

# --- Helper Functions ---

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


# --- Load Data ---

load = np.load('DATA_file_L96.npz')
data = {}
for key in load:
    print(key)
    data[key] = load[key]

t        = data['time']    # time grid, shape (N,)
X_true   = data['X_true']  # true state trajectories, shape (AVG_SIM, N, L)
Y_true   = data['Y_true']  # noisy observations,      shape (AVG_SIM, N, dy)
X_EnKF   = data['X_EnKF']  # EnKF ensemble output,    shape (AVG_SIM, N, L, J)
X_SIR    = data['X_SIR']   # SIR ensemble output,     shape (AVG_SIM, N, L, J)
X_OT     = data['X_OT']    # OT filter output,        shape (AVG_SIM, N, L, J)
X_OT_reg = data['X_OT_reg']# OT_reg filter output,   shape (AVG_SIM, N, L, J)
delta    = data['delta']    # regularization weights used during the run


# --- Derived Dimensions ---

AVG_SIM     = X_OT.shape[0]   # number of independent simulation runs
J           = X_EnKF.shape[3] # EnKF ensemble size
SAMPLE_SIZE = X_OT.shape[3]   # OT particle count
L           = X_true.shape[2] # state space dimension
N           = len(t)           # number of time steps

# --- Compute MSE ---

MSE_EnKF   = mse(X_EnKF,   X_true)
MSE_SIR    = mse(X_SIR,    X_true)
MSE_OT     = mse(X_OT,     X_true)
MSE_OT_reg = mse(X_OT_reg, X_true)


# --- State Trajectory Plots ---

j     = 0   # simulation index to visualize
x_lim = 25  # y-axis limit for state trajectory plots

for s in range(3):
    plt.figure(figsize=(6, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, X_EnKF[j, :, s, :], 'C4', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', label='True state', lw=2)
    plt.ylabel('EnKF', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.legend(loc=4, fontsize=fontsize)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 2)
    plt.plot(t, X_SIR[j, :, s, :], 'C5', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    plt.ylabel('SIR', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 3)
    plt.plot(t, X_OT[j, :, s, :], 'b', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 4)
    plt.plot(t, X_OT_reg[j, :, s, :], 'r', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    plt.ylabel(r'$OT~(\lambda=0.1)$', fontsize=fontsize)
    plt.xlabel('time', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.savefig(f'L96_x{s + 1}_reg.pdf', bbox_inches='tight', dpi=200)


# --- MSE Comparison Plot ---

plt.figure(figsize=(6, 10))
plt.semilogy(t, MSE_OT,     'b--', label=r"$OT_{(\lambda=0)}$",   lw=2)
plt.semilogy(t, MSE_OT_reg, 'r-.', label=r"$OT_{(\lambda=0.1)}$", lw=2)
plt.semilogy(t, MSE_EnKF,   ':',   label=r"$EnKF$", color='C4',   lw=2.5)
plt.semilogy(t, MSE_SIR,    ':',   label=r"$SIR$",  color='C5',   lw=2.5)
plt.xlabel('time', fontsize=fontsize)
plt.ylabel('MSE',  fontsize=fontsize)
plt.legend(loc=0,  fontsize=fontsize)
plt.savefig('L96_mse_reg.pdf', bbox_inches='tight')


# --- Print Summary ---

print(f"MSE EnKF:   mean={MSE_EnKF.mean():.4f}")
print(f"MSE SIR:    mean={MSE_SIR.mean():.4f}")
print(f"MSE OT:     mean={MSE_OT.mean():.4f}")
print(f"MSE OT_reg: mean={MSE_OT_reg.mean():.4f}")

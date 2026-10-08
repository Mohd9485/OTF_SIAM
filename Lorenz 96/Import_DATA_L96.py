"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.close('all')
fontsize = 19
plt.rc('font', size=fontsize)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['savefig.transparent'] = False   # True would also blank the axes patch
matplotlib.rcParams['savefig.facecolor']   = 'none'  # transparent outside the axes
matplotlib.rcParams['savefig.edgecolor']   = 'none'
matplotlib.rcParams['axes.facecolor']      = 'white' # white inside the axes
matplotlib.rcParams['figure.facecolor']    = 'none'


# --- Helper Functions ---

def mse(x, x_true):
    """
    Compute the per-time-step MSE of the ensemble mean, averaged over simulations.

    Args:
        x      (ndarray): filter ensemble, shape (AVG_SIM, N, L, J)
        x_true (ndarray): true state trajectories, shape (AVG_SIM, N, L)

    Returns:
        ndarray: MSE at each time step, shape (N,)
    """
    x_mean = (x - x_true.reshape(AVG_SIM, N, L, 1)).mean(axis=3)  # ensemble-mean error, (AVG_SIM, N, L)
    return ((x_mean * x_mean).sum(axis=2)).mean(axis=0)            # squared norm over states, then mean over simulations


# --- Load Data ---

data = dict(np.load('DATA_file_L96.npz'))  # written by main.py

# The stored keys keep the original 'OT' names; the local variables use the OTF naming
t                 = data['time']              # time grid,                   shape (N,)
X_true            = data['X_true']            # true state trajectories,     shape (AVG_SIM, N, L)
X_EnKF            = data['X_EnKF']            # EnKF ensemble,               shape (AVG_SIM, N, L, J)
X_SIR             = data['X_SIR']             # SIR ensemble,                shape (AVG_SIM, N, L, J)
X_OTF             = data['X_OT']              # OTF (λ=0) ensemble,          shape (AVG_SIM, N, L, J)
X_OTF_reg         = data['X_OT_reg']          # OTF (λ=0.1) ensemble,        shape (AVG_SIM, N, L, J)
X_OT_EnKF         = data['X_OT_EnKF']         # OTF_EnKF (λ=0) ensemble,     shape (AVG_SIM, N, L, J)
X_OT_EnKF_reg     = data['X_OT_EnKF_reg']     # OTF_EnKF (λ=0.1) ensemble,   shape (AVG_SIM, N, L, J)
time_EnKF         = data['time_EnKF']         # wall time (s), all simulations — EnKF
time_SIR          = data['time_SIR']          # wall time (s), all simulations — SIR
time_OTF          = data['time_OT']           # wall time (s), all simulations — OTF (λ=0)
time_OTF_reg      = data['time_OT_reg']       # wall time (s), all simulations — OTF (λ=0.1)
time_OT_EnKF      = data['time_OT_EnKF']      # wall time (s), all simulations — OTF_EnKF (λ=0)
time_OT_EnKF_reg  = data['time_OT_EnKF_reg']  # wall time (s), all simulations — OTF_EnKF (λ=0.1)


# --- Derived Dimensions ---

AVG_SIM     = X_OTF.shape[0]   # number of independent simulation runs
L           = X_true.shape[2]  # state space dimension
N           = len(t)           # number of time steps


# --- Compute MSE ---

MSE_EnKF        = mse(X_EnKF,        X_true)
MSE_SIR         = mse(X_SIR,         X_true)
MSE_OTF         = mse(X_OTF,         X_true)
MSE_OTF_reg     = mse(X_OTF_reg,     X_true)
MSE_OT_EnKF     = mse(X_OT_EnKF,     X_true)
MSE_OT_EnKF_reg = mse(X_OT_EnKF_reg, X_true)


# --- State Trajectory Plots ---

labeling = True   # set False to hide all axis labels
j        = 0      # simulation index to visualize
x_lim    = 20     # y-axis limit

# One figure per state component x1, x2, x3: rows EnKF, SIR, OTF (λ=0), OTF_EnKF (λ=0)
for s in range(3):
    plt.figure(figsize=(6, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, X_EnKF[j, :, s, :], color='C1', alpha=0.1, rasterized=True)  # one faint line per particle
    plt.plot(t, X_true[j, :, s], color='k', linestyle='--', lw=2, label='True state')
    if labeling: plt.ylabel('EnKF', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    if s == 0:
        plt.legend(loc=4, fontsize=fontsize)  # legend only on the x1 figure
    plt.gca().get_xaxis().set_visible(False)  # time axis shown on the bottom row only

    plt.subplot(4, 1, 2)
    plt.plot(t, X_SIR[j, :, s, :], color='C2', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], color='k', linestyle='--', lw=2)
    if labeling: plt.ylabel('SIR', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 3)
    plt.plot(t, X_OTF[j, :, s, :], color='C3', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], color='k', linestyle='--', lw=2)
    if labeling: plt.ylabel(r'$OTF~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(4, 1, 4)
    plt.plot(t, X_OT_EnKF[j, :, s, :], color='C5', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], color='k', linestyle='--', lw=2)
    if labeling: plt.ylabel(r'$OTF_{EnKF}~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    if labeling: plt.xlabel('time', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'L96_x{s + 1}.pdf', bbox_inches='tight', dpi=200)  # dpi sets the resolution of the rasterized particles


# --- MSE Comparison Plot ---

plt.figure(figsize=(6, 10))
plt.semilogy(t, MSE_EnKF,        linestyle=':',  color='C1', label=r"$EnKF$",                    lw=2.5)
plt.semilogy(t, MSE_SIR,         linestyle=':',  color='C2', label=r"$SIR$",                     lw=2.5)
plt.semilogy(t, MSE_OTF,         linestyle='--', color='C3', label=r"$OTF~(\lambda=0)$",         lw=2.5)
plt.semilogy(t, MSE_OTF_reg,     linestyle='-.', color='C4', label=r"$OTF~(\lambda=0.1)$",       lw=2.5)
plt.semilogy(t, MSE_OT_EnKF,     linestyle='--', color='C5', label=r"$OTF_{EnKF}~(\lambda=0)$",   lw=2.5)
plt.semilogy(t, MSE_OT_EnKF_reg, linestyle='-.', color='C6', label=r"$OTF_{EnKF}~(\lambda=0.1)$", lw=2.5)
if labeling: plt.xlabel('time', fontsize=fontsize)
if labeling: plt.ylabel('MSE',  fontsize=fontsize)
plt.legend(loc=0, fontsize=fontsize)  # loc=0: matplotlib picks the best position
plt.tight_layout()
plt.savefig('L96_mse.pdf', bbox_inches='tight')


# --- Print Summary ---

print(f"MSE EnKF:             mean={MSE_EnKF.mean():.4f}")
print(f"MSE SIR:              mean={MSE_SIR.mean():.4f}")
print(f"MSE OTF (λ=0):        mean={MSE_OTF.mean():.4f}")
print(f"MSE OTF (λ=0.1):      mean={MSE_OTF_reg.mean():.4f}")
print(f"MSE OTF_EnKF (λ=0):   mean={MSE_OT_EnKF.mean():.4f}")
print(f"MSE OTF_EnKF (λ=0.1): mean={MSE_OT_EnKF_reg.mean():.4f}")

print(f"\nComputational time per simulation (total / {AVG_SIM} runs):")
print(f"    EnKF:             {time_EnKF        / AVG_SIM:.2f}s")
print(f"    SIR:              {time_SIR         / AVG_SIM:.2f}s")
print(f"    OTF (λ=0):        {time_OTF         / AVG_SIM:.2f}s")
print(f"    OTF (λ=0.1):      {time_OTF_reg     / AVG_SIM:.2f}s")
print(f"    OTF_EnKF (λ=0):   {time_OT_EnKF     / AVG_SIM:.2f}s")
print(f"    OTF_EnKF (λ=0.1): {time_OT_EnKF_reg / AVG_SIM:.2f}s")

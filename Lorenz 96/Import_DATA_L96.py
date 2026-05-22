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


# --- Helper Functions ---

def mse(x, x_true):
    x_mean = (x - x_true.reshape(AVG_SIM, N, L, 1)).mean(axis=3)
    return ((x_mean * x_mean).sum(axis=2)).mean(axis=0)


# --- Load Data ---

data = dict(np.load('DATA_file_L96.npz'))

t             = data['time']           # time grid,                    shape (N,)
X_true        = data['X_true']         # true state trajectories,      shape (AVG_SIM, N, L)
Y_true        = data['Y_true']         # noisy observations,           shape (AVG_SIM, N, dy)
X_EnKF        = data['X_EnKF']         # EnKF ensemble,                shape (AVG_SIM, N, L, J)
X_SIR         = data['X_SIR']          # SIR ensemble,                 shape (AVG_SIM, N, L, J)
X_OT          = data['X_OT']           # OT (λ=0) ensemble,            shape (AVG_SIM, N, L, J)
X_OT_reg      = data['X_OT_reg']       # OT (λ=0.1) ensemble,          shape (AVG_SIM, N, L, J)
X_OT_EnKF     = data['X_OT_EnKF']      # OT-EnKF (λ=0) ensemble,       shape (AVG_SIM, N, L, J)
X_OT_EnKF_reg = data['X_OT_EnKF_reg']  # OT-EnKF (λ=0.1) ensemble,    shape (AVG_SIM, N, L, J)
delta         = data['delta']           # regularization weights
time_EnKF        = data['time_EnKF']        # wall time — EnKF
time_SIR         = data['time_SIR']         # wall time — SIR
time_OT          = data['time_OT']          # wall time — OT (λ=0)
time_OT_reg      = data['time_OT_reg']      # wall time — OT (λ=0.1)
time_OT_EnKF     = data['time_OT_EnKF']     # wall time — OT-EnKF (λ=0)
time_OT_EnKF_reg = data['time_OT_EnKF_reg'] # wall time — OT-EnKF (λ=0.1)


# --- Derived Dimensions ---

AVG_SIM     = X_OT.shape[0]    # number of independent simulation runs
J           = X_EnKF.shape[3]  # EnKF ensemble size
SAMPLE_SIZE = X_OT.shape[3]    # OT particle count
L           = X_true.shape[2]  # state space dimension
N           = len(t)            # number of time steps


# --- Compute MSE ---

MSE_EnKF        = mse(X_EnKF,        X_true)
MSE_SIR         = mse(X_SIR,         X_true)
MSE_OT          = mse(X_OT,          X_true)
MSE_OT_reg      = mse(X_OT_reg,      X_true)
MSE_OT_EnKF     = mse(X_OT_EnKF,     X_true)
MSE_OT_EnKF_reg = mse(X_OT_EnKF_reg, X_true)


# --- State Trajectory Plots ---

labeling = True*0  # show labels and legends in all plots
j        = 0     # simulation index to visualize
x_lim    = 25    # y-axis limit

for s in range(3):
    plt.figure(figsize=(6, 18))

    plt.subplot(6, 1, 1)
    plt.plot(t, X_EnKF[j, :, s, :], 'C1', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2, label='True state')
    if labeling: plt.ylabel('EnKF', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.legend(loc=4, fontsize=fontsize)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 2)
    plt.plot(t, X_SIR[j, :, s, :], 'C2', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    if labeling: plt.ylabel('SIR', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 3)
    plt.plot(t, X_OT[j, :, s, :], 'C3', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    if labeling: plt.ylabel(r'$OTF~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 4)
    plt.plot(t, X_OT_reg[j, :, s, :], 'C4', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    if labeling: plt.ylabel(r'$OTF~(\lambda=0.1)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 5)
    plt.plot(t, X_OT_EnKF[j, :, s, :], 'C5', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    if labeling: plt.ylabel(r'$OTF_{EnKF}~(\lambda=0)$', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.gca().get_xaxis().set_visible(False)

    plt.subplot(6, 1, 6)
    plt.plot(t, X_OT_EnKF_reg[j, :, s, :], 'C6', alpha=0.1, rasterized=True)
    plt.plot(t, X_true[j, :, s], 'k--', lw=2)
    if labeling: plt.ylabel(r'$OTF_{EnKF}~(\lambda=0.1)$', fontsize=fontsize)
    if labeling: plt.xlabel('time', fontsize=fontsize)
    plt.ylim(-x_lim, x_lim)
    plt.savefig(f'L96_x{s + 1}.pdf', bbox_inches='tight', dpi=200)


# --- MSE Comparison Plot ---

plt.figure(figsize=(6, 18))
plt.semilogy(t, MSE_EnKF,        ':',  color='C1', label=r"$EnKF$",                    lw=2.5)
plt.semilogy(t, MSE_SIR,         ':',  color='C2', label=r"$SIR$",                     lw=2.5)
plt.semilogy(t, MSE_OT,          '--', color='C3', label=r"$OTF~(\lambda=0)$",           lw=2.5)
plt.semilogy(t, MSE_OT_reg,      '-.', color='C4', label=r"$OTF~(\lambda=0.1)$",         lw=2.5)
plt.semilogy(t, MSE_OT_EnKF,     '--', color='C5', label=r"$OTF_{EnKF}~(\lambda=0)$",   lw=2.5)
plt.semilogy(t, MSE_OT_EnKF_reg, '-.', color='C6', label=r"$OTF_{EnKF}~(\lambda=0.1)$", lw=2.5)
if labeling: plt.xlabel('time', fontsize=fontsize)
if labeling: plt.ylabel('MSE',  fontsize=fontsize)
plt.legend(loc=0,  fontsize=fontsize)
plt.savefig('L96_mse.pdf', bbox_inches='tight')


# --- Print Summary ---

print(f"MSE EnKF:        mean={MSE_EnKF.mean():.4f}")
print(f"MSE SIR:         mean={MSE_SIR.mean():.4f}")
print(f"MSE OTF:          mean={MSE_OT.mean():.4f}")
print(f"MSE OTF_reg:      mean={MSE_OT_reg.mean():.4f}")
print(f"MSE OTF_EnKF:     mean={MSE_OT_EnKF.mean():.4f}")
print(f"MSE OTF_EnKF_reg: mean={MSE_OT_EnKF_reg.mean():.4f}")

print(f"\nComputational time per simulation (total / {AVG_SIM} runs):")
print(f"    EnKF:        {time_EnKF        / AVG_SIM:.2f}s")
print(f"    SIR:         {time_SIR         / AVG_SIM:.2f}s")
print(f"    OTF:          {time_OT          / AVG_SIM:.2f}s")
print(f"    OTF_reg:      {time_OT_reg      / AVG_SIM:.2f}s")
print(f"    OTF_EnKF:     {time_OT_EnKF     / AVG_SIM:.2f}s")
print(f"    OTF_EnKF_reg: {time_OT_EnKF_reg / AVG_SIM:.2f}s")

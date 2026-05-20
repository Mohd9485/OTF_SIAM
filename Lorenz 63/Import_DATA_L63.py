"""
@author: Mohammad Al-Jarrah
"""

# --- Imports ---

import ot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from EnKF import EnKF
from SIR import SIR


# --- Plot Style ---

plt.close('all')
plt.rc('font', size=19)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontsize = 19  # font size for legend labels

# --- Helper Functions ---

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


# --- Load Data ---

data = np.load('DATA_file_L63.npz')

for key in data:
    print(key)

t        = data['t']         # time grid
Noise    = data['Noise']     # [sigmma, gamma] noise vector
tau      = data['tau']       # time step size
Odeint   = data['Odeint']    # integration method flag
sigmma0  = data['sigmma0']   # variance of the initial state distribution
X0       = data['X0']        # initial particle ensembles
Y_true   = data['Y_true']    # observation trajectories
X_true   = data['X_true']    # true state trajectories
X_OT     = data['X_OT']      # OTF filtered particles (no regularization)
X_OT_reg = data['X_OT_reg']  # OTF filtered particles (with regularization)


# --- Simulation Dimensions ---

AVG_SIM = X_OT.shape[0]   # number of simulation runs
N       = X_OT.shape[1]   # number of time steps
L       = X_OT.shape[2]   # state dimension
J       = X_OT.shape[3]   # ensemble size
dy      = Y_true.shape[2]  # observation dimension


# --- Run Filters ---

X_SIR  = SIR(Y_true,  X0, L63, h, t, tau, Noise, Odeint)
X_EnKF = EnKF(Y_true, X0, L63, h, t, tau, Noise, Odeint)


# --- W2 Computation ---

p_true        = int(1e3)  # reference particles used to compute W2
true_particle = int(1e5)  # large-SIR ensemble size for the true distribution

distance_EnKF   = []  # placeholder; replaced by ndarray after loop
distance_SIR    = []
distance_OT     = []
distance_OT_reg = []

X0_true = np.zeros((AVG_SIM, L, true_particle))  # large-SIR initial ensembles
for k in range(AVG_SIM):
    X0_true[k,] = np.transpose(np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0 * np.eye(L), true_particle))

distance_enkf   = {}  # per-timestep W2 values accumulated across simulations
distance_sir    = {}
distance_ot     = {}
distance_ot_reg = {}

for i in range(N):
    distance_enkf[str(i)]   = []
    distance_sir[str(i)]    = []
    distance_ot[str(i)]     = []
    distance_ot_reg[str(i)] = []

for k in range(AVG_SIM):
    X_true_dist = SIR(Y_true[k,].reshape(1, N, dy), X0_true[k,].reshape(1, L, true_particle), L63, h, t, tau, Noise, Odeint)
    print("Sim: ", k)

    for i in range(N):
        M_enkf   = ot.dist(X_true_dist[0, i, :, :p_true].T, X_EnKF[k, i,].T)    # cost matrix: reference vs EnKF
        M_sir    = ot.dist(X_true_dist[0, i, :, :p_true].T, X_SIR[k, i,].T)     # cost matrix: reference vs SIR
        M_ot     = ot.dist(X_true_dist[0, i, :, :p_true].T, X_OT[k, i,].T)      # cost matrix: reference vs OT
        M_ot_reg = ot.dist(X_true_dist[0, i, :, :p_true].T, X_OT_reg[k, i,].T)  # cost matrix: reference vs OT_reg

        a = np.ones(p_true) / p_true  # uniform weights for reference distribution
        b = np.ones(J) / J            # uniform weights for filter distribution

        distance_enkf[str(i)].append(np.sqrt(ot.emd2(a, b, M_enkf)))
        distance_sir[str(i)].append(np.sqrt(ot.emd2(a, b, M_sir)))
        distance_ot[str(i)].append(np.sqrt(ot.emd2(a, b, M_ot)))
        distance_ot_reg[str(i)].append(np.sqrt(ot.emd2(a, b, M_ot_reg)))

distance_EnKF   = np.zeros((N, AVG_SIM))  # W2 array: (time steps, simulations)
distance_SIR    = np.zeros_like(distance_EnKF)
distance_OT     = np.zeros_like(distance_EnKF)
distance_OT_reg = np.zeros_like(distance_EnKF)

for i in range(N):
    distance_EnKF[i,]   = np.array(distance_enkf[str(i)])
    distance_SIR[i,]    = np.array(distance_sir[str(i)])
    distance_OT[i,]     = np.array(distance_ot[str(i)])
    distance_OT_reg[i,] = np.array(distance_ot_reg[str(i)])


# --- W2 Plot ---

plt.figure(figsize=(6, 10))
plt.semilogy(t, distance_OT.mean(axis=1),     'b--', label=r"$OT_{(\lambda=0)}$",   lw=2)
plt.semilogy(t, distance_OT_reg.mean(axis=1), 'r-.', label=r"$OT_{(\lambda=0.1)}$", lw=2)
plt.semilogy(t, distance_EnKF.mean(axis=1),   ':',   label=r"$EnKF$", color='C4',   lw=2.5)
plt.semilogy(t, distance_SIR.mean(axis=1),    ':',   label=r"$SIR$",  color='C5',   lw=2.5)
plt.legend(fontsize=fontsize)
plt.xlabel(r'$time$',fontsize=fontsize)
plt.ylabel(r'$W_2$',fontsize=fontsize)
plt.savefig('L63_w2_vs_time.pdf', bbox_inches='tight', dpi=300)


# --- Density Heatmap Plots ---

sim = 0  # simulation index to visualize

true_particle = int(1e6)  # large-SIR ensemble size for per-state density reference
X0_true       = np.zeros((1, L, true_particle))
X0_true[0,]   = 1 * np.transpose(np.random.multivariate_normal(np.zeros(L), sigmma0*sigmma0 * np.eye(L), true_particle))
X_true_dist   = SIR(Y_true[sim,].reshape(1, N, dy), X0_true, L63, h, t, tau, Noise, Odeint)

for num_plot_state in range(L):
    n_bins   = 50   # number of histogram bins for the density estimate
    fontsize = 16   # font size for axis tick labels

    if num_plot_state == 0:
        y_lim    = [-30, 30]  # y-axis range for state x1
        n_ylabel = 3
    elif num_plot_state == 1:
        y_lim    = [-35, 35]  # y-axis range for state x2
        n_ylabel = 3
    elif num_plot_state == 2:
        y_lim    = [-20, 60]  # y-axis range for state x3
        n_ylabel = 3

    position_bins = np.linspace(y_lim[0], y_lim[1], n_bins)  # histogram bin edges

    plt.figure(figsize=(6, 10))

    # --- SIR Reference ---
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_true_dist[0, :, num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density

    plt.subplot(5, 1, 1)
    sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel(r'$True$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # --- EnKF ---
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_EnKF[sim, :, num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density

    plt.subplot(5, 1, 2)
    sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel(r'$EnKF$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # --- SIR ---
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_SIR[sim, :, num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density

    plt.subplot(5, 1, 3)
    sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel(r'$SIR$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # --- OT (no regularization) ---
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_OT[sim, :, num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density

    plt.subplot(5, 1, 4)
    sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.ylabel(r'$OT~(\lambda=0)$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # --- OT (with regularization) ---
    density_matrix = np.zeros((N, len(position_bins) - 1))
    for n in range(N):
        density, _ = np.histogram(X_OT_reg[sim, :, num_plot_state,][n], bins=position_bins, density=True)
        density_matrix[n, :] = density

    plt.subplot(5, 1, 5)
    sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
    plt.xticks(ticks=np.linspace(0, N, 11), labels=np.round(np.linspace(0, N*tau, 11), 1))
    plt.ylabel(r'$OT~(\lambda=0.1)$')
    plt.xlabel(r'$time$')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    plt.savefig(f'L63_X{num_plot_state+1}_vs_time.pdf', bbox_inches='tight')

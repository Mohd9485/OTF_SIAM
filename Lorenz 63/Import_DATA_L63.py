"""
@author: Mohammad Al-Jarrah

Load the L63 filtering results and plot W2 vs time and per-state densities.

The W2 distances to the large-particle SIR reference (exact W2, the same metric used
to tune OTF in smac_tuning_L63.py) and the reference particles are computed in
main.py and stored in DATA_file_L63.npz, so this script only plots.
"""

# --- Imports ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# --- Plot Style ---

plt.close('all')
plt.rc('font', size=19)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['savefig.transparent'] = False   # True would also blank the axes patch
matplotlib.rcParams['savefig.facecolor']   = 'none'  # transparent outside the axes
matplotlib.rcParams['savefig.edgecolor']   = 'none'
matplotlib.rcParams['axes.facecolor']      = 'white' # white inside the axes
matplotlib.rcParams['figure.facecolor']    = 'none'
fontsize         = 19    # font size for axis labels and the legend
fontsize_heatmap = 16    # font size for the heatmap row names and axis labels
fontsize_poinc   = 25    # font size for the Poincaré plot axis labels and legend
labeling         = True  # set False to hide all axis labels


# --- Load Data ---

data = dict(np.load('DATA_file_L63.npz'))  # written by main.py

for key in data:
    print(key)  # list the stored arrays, as a quick check of the file contents

t            = data['t']            # time grid
tau          = data['tau']          # time step size
X_EnKF       = data['X_EnKF']       # EnKF filtered particles
X_SIR        = data['X_SIR']        # SIR filtered particles
X_OTF        = data['X_OTF']        # OTF filtered particles (no regularization)
X_OTF_reg    = data['X_OTF_reg']    # OTF filtered particles (with regularization)
time_EnKF    = data['time_EnKF']    # EnKF computational time (seconds)
time_SIR     = data['time_SIR']     # SIR computational time (seconds)
time_OTF     = data['time_OTF']     # OTF computational time (seconds, λ=0)
time_OTF_reg = data['time_OTF_reg'] # OTF computational time (seconds, λ=0.1)
X_true_dist  = data['X_true_dist']  # large-SIR reference particles of simulation `sim`, (N, L, true_particle)
sim          = int(data['sim'])     # simulation index visualized in the density heatmaps
distance_W2  = {                    # filter name -> W2 per time step and simulation, (N, AVG_SIM)
    'EnKF':        data['w2_EnKF'],
    'SIR':         data['w2_SIR'],
    'OTF (λ=0)':   data['w2_OTF'],
    'OTF (λ=0.1)': data['w2_OTF_reg'],
}


# --- Simulation Dimensions ---

AVG_SIM = X_OTF.shape[0]  # number of simulation runs
N       = X_OTF.shape[1]  # number of time steps
L       = X_OTF.shape[2]  # state dimension


# --- W2 Plot ---

line_styles = {        # filter name -> (color, line style, legend label)
    'EnKF':        ('C1', ':',  r"EnKF"),
    'SIR':         ('C2', ':',  r"SIR"),
    'OTF (λ=0)':   ('C3', '--', r"$OTF_{(\lambda=0)}$"),
    'OTF (λ=0.1)': ('C4', '-.', r"$OTF_{(\lambda=0.1)}$"),
}

fig, ax = plt.subplots(figsize=(6, 10))
for name, (color, ls, label) in line_styles.items():
    ax.semilogy(t, distance_W2[name].mean(axis=1), color=color, linestyle=ls, label=label, lw=2.5)
ax.legend(fontsize=fontsize)
if labeling: ax.set_xlabel(r'$time$',         fontsize=fontsize)
if labeling: ax.set_ylabel(r'$\mathrm{W}_2$', fontsize=fontsize)
fig.tight_layout()
fig.savefig('L63_w2_vs_time.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)


# --- Density Heatmap Plots ---

heatmap_rows = [       # row name -> particles of simulation `sim`, shape (N, L, particles)
    (r'True',                X_true_dist),
    (r'EnKF',                X_EnKF[sim]),
    (r'SIR',                 X_SIR[sim]),
    (r'$OTF~(\lambda=0)$',   X_OTF[sim]),
    (r'$OTF~(\lambda=0.1)$', X_OTF_reg[sim]),
]

n_bins   = 50   # number of histogram bins for the density estimate
n_ylabel = 3    # number of y-axis tick labels
y_lims   = {0: [-30, 30], 1: [-35, 35], 2: [-20, 60]}  # y-axis range for states x1, x2, x3

for num_plot_state in range(L):
    y_lim         = y_lims[num_plot_state]
    position_bins = np.linspace(y_lim[0], y_lim[1], n_bins)  # histogram bin edges

    fig = plt.figure(figsize=(6, 10))
    for row, (row_name, particles) in enumerate(heatmap_rows):
        density_matrix = np.zeros((N, len(position_bins) - 1))  # (time steps, bins)
        for n in range(N):
            # Normalized histogram of the ensemble at time step n (integrates to 1 over the bins)
            density, _ = np.histogram(particles[n, num_plot_state], bins=position_bins, density=True)
            density_matrix[n, :] = density

        plt.subplot(len(heatmap_rows), 1, row + 1)
        # Transposed so time runs along the x-axis; robust=True clips the color scale at the 2nd/98th percentiles
        sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True, rasterized=True)
        ax = plt.gca()
        ax.invert_yaxis()  # heatmap rows start at the top; flip so the state value increases upward
        plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
        # The row names identify the filters (they act as the legend), hidden with the axis labels
        if labeling: plt.ylabel(row_name, fontsize=fontsize_heatmap)
        if row < len(heatmap_rows) - 1:
            ax.get_xaxis().set_visible(False)
        else:
            # Ticks are placed in time-step units and labelled in physical time
            plt.xticks(ticks=np.linspace(0, N, 11), labels=np.round(np.linspace(0, N*tau, 11), 1))
            if labeling: plt.xlabel(r'$time$', fontsize=fontsize_heatmap)
        for side in ('top', 'right', 'left', 'bottom'):
            ax.spines[side].set_visible(True)  # seaborn hides the spines; draw a frame around every row

    fig.tight_layout()
    fig.savefig(f'L63_X{num_plot_state+1}.pdf', bbox_inches='tight')
    plt.close(fig)


# --- Poincaré Constant Plot ---
# The estimates are computed in OTF_Poincare.py, which saves the plotted series to poincare_otf_data.npz

poinc = dict(np.load('poincare_otf_data.npz'))

sv        = poinc['t'][1:]  # t = 0 is skipped: no prior is reconstructed there
lb_pr_otf = poinc['LB_prior_otf'][1:]    # covariance lower bound, prior, averaged over simulations
lb_po_otf = poinc['LB_post_otf'][1:]     # covariance lower bound, posterior
rk_pr_otf = poinc['RKHS_prior_otf'][1:]  # RKHS estimate, prior
rk_po_otf = poinc['RKHS_post_otf'][1:]   # RKHS estimate, posterior

def _mask(a, b):
    """Keep time steps where both series are finite and positive, as required by the log-scale axis."""
    return np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)

m_lb_otf = _mask(lb_pr_otf, lb_po_otf)
m_rk_otf = _mask(rk_pr_otf, rk_po_otf)

fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
ax1.semilogy(sv[m_lb_otf], lb_pr_otf[m_lb_otf], color='C3', linestyle='-',  lw=4,   label=r'LB Prior')
ax1.semilogy(sv[m_lb_otf], lb_po_otf[m_lb_otf], color='C4', linestyle='--', lw=4,   label=r'LB Posterior')
ax1.semilogy(sv[m_rk_otf], rk_pr_otf[m_rk_otf], color='C5', linestyle='-.', lw=4,   label=r'RKHS Prior')
ax1.semilogy(sv[m_rk_otf], rk_po_otf[m_rk_otf], color='C6', linestyle=':',  lw=4,   label=r'RKHS Posterior')
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=fontsize_poinc)
if labeling: ax1.set_ylabel(r'$\hat{P}$', fontsize=fontsize_poinc)
if labeling: ax1.set_xlabel(r'$time$',    fontsize=fontsize_poinc)
fig.tight_layout()
fig.savefig('poincare_otf.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)


# --- W2 Distances Averaged over Time ---
# t = 0 is skipped: every filter still holds the prior there (as in smac_tuning_L63.py)

for name in distance_W2:
    print(f"{name:12s} W2 (t > 0): {distance_W2[name][1:].mean():.4f}")


# --- Computational Time per Simulation ---

print(f"EnKF        time per sim: {float(time_EnKF)    / AVG_SIM:.2f}s")
print(f"SIR         time per sim: {float(time_SIR)     / AVG_SIM:.2f}s")
print(f"OTF (λ=0)   time per sim: {float(time_OTF)     / AVG_SIM:.2f}s")
print(f"OTF (λ=0.1) time per sim: {float(time_OTF_reg) / AVG_SIM:.2f}s")

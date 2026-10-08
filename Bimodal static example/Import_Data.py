"""
@author: Mohammad Al-Jarrah

Loads three pkl files and saves four subfigures:
  (a) SW2 vs dimension                 (from error_and_time_vs_dim.pkl)
  (b) Computational time vs dimension  (from error_and_time_vs_dim.pkl)
  (c) SW2 vs # of particles (d=2)      (from error_vs_particles_d_2.pkl)
  (d) SW2 vs # of particles (d=10)     (from error_vs_particles_d_10.pkl)

Legend appears only on subfigure (a).

SW2 slices the full joint clouds along random directions.
"""

import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['savefig.transparent'] = False   # True would also blank the axes patch
matplotlib.rcParams['savefig.facecolor']   = 'none'  # transparent outside the axes
matplotlib.rcParams['savefig.edgecolor']   = 'none'
matplotlib.rcParams['axes.facecolor']      = 'white' # white inside the axes
matplotlib.rcParams['figure.facecolor']    = 'none'
plt.rc('font', size=13)    # base size, used for the tick labels
fontsize = 22              # font size for axis labels and the legend
labeling = True            # set False to hide all axis labels

script_dir = os.path.dirname(os.path.abspath(__file__))  # pkl files are read from (and PDFs saved to) this folder

time_dim_pkl      = "error_and_time_vs_dim.pkl"   # SW2 and computational time vs dimension
particles_pkl_d2  = "error_vs_particles_d_2.pkl"  # SW2 vs # of particles, d = 2
particles_pkl_d10 = "error_vs_particles_d_10.pkl" # SW2 vs # of particles, d = 10

with open(os.path.join(script_dir, time_dim_pkl), 'rb') as f:
    dim_data = pickle.load(f)

with open(os.path.join(script_dir, particles_pkl_d2), 'rb') as f:
    part_data_d2 = pickle.load(f)

with open(os.path.join(script_dir, particles_pkl_d10), 'rb') as f:
    part_data_d10 = pickle.load(f)

# The SW2 distances are written by the same scripts that write the pkl files; a pkl
# produced before SW2 was added will be missing them.
for pkl_name, pkl_contents, producer in ((time_dim_pkl,      dim_data,      "error_and_time_vs_dim.py"),
                                         (particles_pkl_d2,  part_data_d2,  "error_vs_particles.py with d = 2"),
                                         (particles_pkl_d10, part_data_d10, "error_vs_particles.py with d = 10")):
    if 'distance_ot_sw2' not in pkl_contents:
        raise KeyError(
            f"{pkl_name} has no SW2 distances -- regenerate it by rerunning {producer}."
        )

# --- OTF Line Styles ---
# One style per regularization weight λ; (0, (3, 1, 1, 1)) is a dense dash-dot pattern for the middle λ
Lambda        = dim_data['Lambda']  # OTF regularization weights λ
lambda_styles = {
    Lambda[0]: {'color': 'C3', 'ls': '--',              'marker': 'v'},
    Lambda[1]: {'color': 'C9', 'ls': (0, (3, 1, 1, 1)), 'marker': 's'},
    Lambda[2]: {'color': 'C4', 'ls': '-.',              'marker': 'o'},
}

# --- Unpack Data ---

D              = dim_data['D']                    # state dimensions of the sweep
time_otf       = dim_data['time_save_otf']        # OTF computational time, keyed by str(λ)
time_baselines = dim_data['time_save_baselines']  # EnKF / SIR computational time, keyed 'enkf' / 'sir'

NN_d2        = part_data_d2['NN']       # particle counts of the d = 2 sweep
Lambda_p_d2  = part_data_d2['Lambda']   # λ values of the d = 2 sweep

NN_d10       = part_data_d10['NN']      # particle counts of the d = 10 sweep
Lambda_p_d10 = part_data_d10['Lambda']  # λ values of the d = 10 sweep

# SW2 distances; the OTF entries are dicts keyed by str(λ)
sw2_distance_ot         = dim_data['distance_ot_sw2']
sw2_distance_sir        = dim_data['distance_sir_sw2']
sw2_distance_enkf       = dim_data['distance_enkf_sw2']

sw2_p_distance_ot_d2    = part_data_d2['distance_ot_sw2']
sw2_p_distance_sir_d2   = part_data_d2['distance_sir_sw2']
sw2_p_distance_enkf_d2  = part_data_d2['distance_enkf_sw2']

sw2_p_distance_ot_d10   = part_data_d10['distance_ot_sw2']
sw2_p_distance_sir_d10  = part_data_d10['distance_sir_sw2']
sw2_p_distance_enkf_d10 = part_data_d10['distance_enkf_sw2']

# --- Save each subfigure independently ---
# Each spec holds the output filename, the axis labels, and a function that draws the curves on an axis
subfig_specs = [
    {
        'filename': 'bimodal_example_vs_dim.pdf',
        'xlabel':   r'$dim$',
        'ylabel':   r'$\mathrm{SW}_2$',
        'plot': lambda ax: [
            ax.semilogy(D, sw2_distance_enkf, marker='D', linestyle=':', color="C1", label=r"EnKF", lw=2.5),
            ax.semilogy(D, sw2_distance_sir,  marker='^', linestyle=':', color="C2", label=r"SIR",  lw=2.5),
            *[ax.semilogy(D, sw2_distance_ot[str(lamda)], marker=lambda_styles[lamda]['marker'],
                          linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'],
                          label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2.5) for lamda in Lambda],
            ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.52), fontsize=fontsize),  # right side, just above EnKF
        ],
    },
    {
        'filename': 'bimodal_example_vs_time.pdf',
        'xlabel':   r'$dim$',
        'ylabel':   r'computational time',
        'plot': lambda ax: [
            ax.semilogy(D, time_baselines['enkf'], marker='D', linestyle=':', color="C1", lw=2.5),
            ax.semilogy(D, time_baselines['sir'],  marker='^', linestyle=':', color="C2", lw=2.5),
            *[ax.semilogy(D, time_otf[str(lamda)], marker=lambda_styles[lamda]['marker'],
                          linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'], lw=2.5)
              for lamda in Lambda],
        ],
    },
    {
        'filename': 'bimodal_example_vs_particles_2d.pdf',
        'xlabel':   r'# of particles',
        'ylabel':   r'$\mathrm{SW}_2$',
        # The largest particle count of the d = 2 sweep is left out of the plot ([:-1])
        'plot': lambda ax: [
            ax.loglog(NN_d2[:-1], sw2_p_distance_enkf_d2[:-1], marker='D', linestyle=':', color="C1", lw=2.5),
            ax.loglog(NN_d2[:-1], sw2_p_distance_sir_d2[:-1],  marker='^', linestyle=':', color="C2", lw=2.5),
            *[ax.loglog(NN_d2[:-1], sw2_p_distance_ot_d2[str(lamda)][:-1], marker=lambda_styles[lamda]['marker'],
                        linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'], lw=2.5)
              for lamda in Lambda_p_d2],
        ],
    },
    {
        'filename': 'bimodal_example_vs_particles_10d.pdf',
        'xlabel':   r'# of particles',
        'ylabel':   r'$\mathrm{SW}_2$',
        'plot': lambda ax: [
            ax.loglog(NN_d10, sw2_p_distance_enkf_d10, marker='D', linestyle=':', color="C1", lw=2.5),
            ax.loglog(NN_d10, sw2_p_distance_sir_d10,  marker='^', linestyle=':', color="C2", lw=2.5),
            *[ax.loglog(NN_d10, sw2_p_distance_ot_d10[str(lamda)], marker=lambda_styles[lamda]['marker'],
                        linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'], lw=2.5)
              for lamda in Lambda_p_d10],
        ],
    },
]

for spec in subfig_specs:
    fig_s, ax_s = plt.subplots(figsize=(6, 8))
    spec['plot'](ax_s)
    if labeling: ax_s.set_xlabel(spec['xlabel'], fontsize=fontsize)
    if labeling: ax_s.set_ylabel(spec['ylabel'], fontsize=fontsize)
    fig_s.tight_layout()
    p_s = os.path.join(script_dir, spec['filename'])
    fig_s.savefig(p_s)
    print(f"Saved {p_s}")
    plt.close(fig_s)

"""
@author: Mohammad Al-Jarrah

Loads three pkl files and generates a single figure with four subplots:
  (a) W2 vs dimension                (from Figure_2_data_2.pkl)
  (b) Computational time vs dimension  (from Figure_2_data_2.pkl)
  (c) W2 vs # of particles (d=2)     (from Figure_2_particles_data_d_2.pkl)
  (d) W2 vs # of particles (d=10)    (from Figure_2_particles_data_d_10.pkl)

Legend appears only on the leftmost subplot.
"""

import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
plt.rc('font', size=13)
fontsize  = 22
labeling  = True  # set False to hide all axis labels

script_dir = os.path.dirname(os.path.abspath(__file__))

time_dim_pkl           = "error_and_time_vs_dim.pkl"
particles_pkl_d2  = "error_vs_particles_d_2.pkl"
particles_pkl_d10 = "error_vs_particles_d_10.pkl"

with open(os.path.join(script_dir, time_dim_pkl), 'rb') as f:
    dim_data = pickle.load(f)

with open(os.path.join(script_dir, particles_pkl_d2), 'rb') as f:
    part_data_d2 = pickle.load(f)

with open(os.path.join(script_dir, particles_pkl_d10), 'rb') as f:
    part_data_d10 = pickle.load(f)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

Lambda = dim_data['Lambda']
lambda_styles = {
    Lambda[0]: {'color': 'C3', 'ls': '--',  'marker': 'v'},
    Lambda[1]: {'color': 'C9', 'ls': (0, (3, 1, 1, 1)),  'marker': 's'},
    Lambda[2]: {'color': 'C4', 'ls': '-.',  'marker': 'o'},
}

D              = dim_data['D']
distance_ot    = dim_data['distance_ot']
distance_sir   = dim_data['distance_sir']
distance_enkf  = dim_data['distance_enkf']
time_otf       = dim_data['time_save_otf']
time_baselines = dim_data['time_save_baselines']

NN_d2              = part_data_d2['NN']
p_distance_ot_d2   = part_data_d2['distance_ot']
p_distance_sir_d2  = part_data_d2['distance_sir']
p_distance_enkf_d2 = part_data_d2['distance_enkf']
Lambda_p_d2        = part_data_d2['Lambda']

NN_d10              = part_data_d10['NN']
p_distance_ot_d10   = part_data_d10['distance_ot']
p_distance_sir_d10  = part_data_d10['distance_sir']
p_distance_enkf_d10 = part_data_d10['distance_enkf']
Lambda_p_d10        = part_data_d10['Lambda']

# fig, axes = plt.subplots(1, 4, figsize=(24, 8))

# # --- (a) W2 vs dimension ---
# ax = axes[0]
# ax.semilogy(D, distance_enkf, 'D:', color="C1", label=r"EnKF", lw=2.5)
# ax.semilogy(D, distance_sir,  '^:', color="C2", label=r"SIR",  lw=2.5)
# for lamda in Lambda:
#     s = lambda_styles[lamda]
#     ax.semilogy(D, distance_ot[str(lamda)], marker=s['marker'], linestyle=s['ls'],
#                 color=s['color'], label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2.5)
# if labeling: ax.set_xlabel(r'$dim$', fontsize=fontsize)
# if labeling: ax.set_ylabel(r'$\mathrm{AA\text{-}SW}_2$', fontsize=fontsize)
# ax.legend(loc='center', bbox_to_anchor=(0.5, 0.7), fontsize=fontsize)

# # --- (b) Computational time vs dimension ---
# ax = axes[1]
# for lamda in Lambda:
#     s = lambda_styles[lamda]
#     ax.semilogy(D, time_otf[str(lamda)], marker=s['marker'], linestyle=s['ls'],
#                 color=s['color'], lw=2.5)
# if labeling: ax.set_xlabel(r'$dim$', fontsize=fontsize)
# if labeling: ax.set_ylabel(r'computational time', fontsize=fontsize)

# # --- (c) W2 vs number of particles (d=2) ---
# ax = axes[2]
# ax.loglog(NN_d2[:-1], p_distance_enkf_d2[:-1], 'D:', color="C1", lw=2.5)
# ax.loglog(NN_d2[:-1], p_distance_sir_d2[:-1],  '^:', color="C2", lw=2.5)
# for lamda in Lambda_p_d2:
#     s = lambda_styles[lamda]
#     ax.loglog(NN_d2[:-1], p_distance_ot_d2[str(lamda)][:-1], marker=s['marker'], linestyle=s['ls'],
#               color=s['color'], lw=2.5)
# if labeling: ax.set_xlabel(r'# of particles', fontsize=fontsize)
# if labeling: ax.set_ylabel(r'$\mathrm{AA\text{-}SW}_2$', fontsize=fontsize)

# # --- (d) W2 vs number of particles (d=10) ---
# ax = axes[3]
# ax.loglog(NN_d10, p_distance_enkf_d10, 'D:', color="C1", lw=2.5)
# ax.loglog(NN_d10, p_distance_sir_d10,  '^:', color="C2", lw=2.5)
# for lamda in Lambda_p_d10:
#     s = lambda_styles[lamda]
#     ax.loglog(NN_d10, p_distance_ot_d10[str(lamda)], marker=s['marker'], linestyle=s['ls'],
#               color=s['color'], lw=2.5)
# if labeling: ax.set_xlabel(r'# of particles', fontsize=fontsize)
# if labeling: ax.set_ylabel(r'$\mathrm{AA\text{-}SW}_2$', fontsize=fontsize)

# p = os.path.join(script_dir, f'Combined_figures.pdf')
# plt.savefig(p, bbox_inches='tight')
# print(f"Saved {p}")

# plt.show()

# --- Save each subfigure independently ---
subfig_specs = [
    {
        'filename': 'bimodal_example_vs_dim.pdf',
        'xlabel':   r'$dim$',
        'ylabel':   r'$\mathrm{AA\text{-}SW}_2$',
        'plot': lambda ax: [
            ax.semilogy(D, distance_enkf, 'D:', color="C1", label=r"EnKF", lw=2.5),
            ax.semilogy(D, distance_sir,  '^:', color="C2", label=r"SIR",  lw=2.5),
            *[ax.semilogy(D, distance_ot[str(lamda)], marker=lambda_styles[lamda]['marker'],
                          linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'],
                          label=rf'$OTF_{{(\lambda={lamda})}}$', lw=2.5) for lamda in Lambda],
            ax.legend(loc='center', bbox_to_anchor=(0.4, 0.65), fontsize=fontsize),
        ],
    },
    {
        'filename': 'bimodal_example_vs_time.pdf',
        'xlabel':   r'$dim$',
        'ylabel':   r'computational time',
        'plot': lambda ax: [
            ax.semilogy(D, time_baselines['enkf'], 'D:', color="C1", lw=2.5),
            ax.semilogy(D, time_baselines['sir'],  '^:', color="C2", lw=2.5),
            *[ax.semilogy(D, time_otf[str(lamda)], marker=lambda_styles[lamda]['marker'],
                          linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'], lw=2.5)
              for lamda in Lambda],
        ],
    },
    {
        'filename': 'bimodal_example_vs_particles_2d.pdf',
        'xlabel':   r'# of particles',
        'ylabel':   r'$\mathrm{AA\text{-}SW}_2$',
        'plot': lambda ax: [
            ax.loglog(NN_d2[:-1], p_distance_enkf_d2[:-1], 'D:', color="C1", lw=2.5),
            ax.loglog(NN_d2[:-1], p_distance_sir_d2[:-1],  '^:', color="C2", lw=2.5),
            *[ax.loglog(NN_d2[:-1], p_distance_ot_d2[str(lamda)][:-1], marker=lambda_styles[lamda]['marker'],
                        linestyle=lambda_styles[lamda]['ls'], color=lambda_styles[lamda]['color'], lw=2.5)
              for lamda in Lambda_p_d2],
        ],
    },
    {
        'filename': 'bimodal_example_vs_particles_10d.pdf',
        'xlabel':   r'# of particles',
        'ylabel':   r'$\mathrm{AA\text{-}SW}_2$',
        'plot': lambda ax: [
            ax.loglog(NN_d10, p_distance_enkf_d10, 'D:', color="C1", lw=2.5),
            ax.loglog(NN_d10, p_distance_sir_d10,  '^:', color="C2", lw=2.5),
            *[ax.loglog(NN_d10, p_distance_ot_d10[str(lamda)], marker=lambda_styles[lamda]['marker'],
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

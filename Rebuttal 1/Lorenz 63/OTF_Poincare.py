"""
@author: Niyizhen (Jenny) Jin 
@author: Mohammad Al-Jarrah

Empirical Poincaré constant for OTF on Lorenz 63.

Loads saved filter particles from DATA_file_L63.npz, reconstructs
the prior at each time step by propagating the previous posterior forward,
and estimates the empirical Poincaré constant using a covariance lower bound
and an RKHS estimator (GPU-batched), averaged over all simulations.
"""

# --- Imports ---

import numpy as np
import torch
import time
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

plt.rc('font', size=19)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
fontsize = 19

SEED = 42
np.random.seed(SEED)


# --- Load Data ---

data    = dict(np.load('DATA_file_L63.npz'))
t       = data['t']
tau     = float(data['tau'])
rk4     = bool(data['rk4'])
Noise   = data['Noise']
sigmma  = float(Noise[0])  # process noise std

X_OTF     = data['X_OTF']      # (AVG_SIM, N, L, J)
# X_OTF_reg = data['X_OTF_reg']  # (AVG_SIM, N, L, J)

# np.savez_compressed('DATA_file_L63_Poincare.npz', t=t, tau=tau, rk4=rk4, Noise=Noise, sigmma=sigmma, X_OTF=X_OTF)

N = X_OTF.shape[1]
L = X_OTF.shape[2]
J = X_OTF.shape[3]

# sim      = 0     # first simulation
labeling = True*0  # set False to hide all axis labels


# --- GPU Setup ---

BATCH_SIZE_GPU = 25
DTYPE          = torch.float64
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# --- Model Definition (for prior propagation) ---

def L63(t, x):
    """Evaluate the Lorenz 63 vector field at state x and time t."""
    d     = np.zeros_like(x)
    sigma = 10
    r     = 28
    b     = 8 / 3
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


# --- Poincaré Estimators ---

def poincare_LB_batch(particles_BLJ):
    """
    Lower bound on the Poincaré constant via the max eigenvalue of the
    empirical covariance matrix Σ̂.

    Args:
        particles_BLJ (ndarray): shape (B, L, J)

    Returns:
        ndarray: shape (B,) — max eigenvalue of Σ̂ per batch item
    """
    X  = np.transpose(particles_BLJ, (0, 2, 1))   # (B, J, L)
    Xc = X - X.mean(axis=1, keepdims=True)
    S  = np.einsum('bjl,bjm->blm', Xc, Xc) / X.shape[1]
    return np.linalg.eigvalsh(S).max(axis=-1)


_eye_cache = {}

def rkhs_poincare_batch_gpu(particles_BLJ):
    """
    RKHS Poincaré estimator: top eigenvalue of L^{-1} Kc L^{-T} with
    Bmat = LK + λI, using RBF kernel and median bandwidth. GPU-batched.

    Args:
        particles_BLJ (ndarray): shape (B, L, J)

    Returns:
        ndarray: shape (B,) — estimated Poincaré constant per batch item
    """
    B = particles_BLJ.shape[0]
    X = torch.as_tensor(np.transpose(particles_BLJ, (0, 2, 1)), dtype=DTYPE, device=device)
    n = X.shape[1]

    diff   = X.unsqueeze(2) - X.unsqueeze(1)   # (B, n, n, d)
    sqdist = diff.pow(2).sum(-1)                 # (B, n, n)

    if n not in _eye_cache:
        _eye_cache[n] = ~torch.eye(n, dtype=torch.bool, device=device)
    off_diag_mask = _eye_cache[n]

    off_vals = sqdist.masked_select(off_diag_mask).view(B, -1)
    h        = (off_vals.median(dim=1).values / 2.0).sqrt().clamp_min(1e-8)
    h2       = (h ** 2).view(B, 1, 1)

    lam = float(n) ** (-0.25)

    K  = torch.exp(-sqdist / (2 * h2))
    Kc = K - K.mean(dim=2, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean(dim=(1, 2), keepdim=True)

    G  = -(diff / h2.unsqueeze(-1)) * K.unsqueeze(-1)    # (B, n, n, d)
    LK = torch.einsum('bkid,bkjd->bij', G, G) / n

    Bmat   = LK + lam * torch.eye(n, dtype=DTYPE, device=device).unsqueeze(0)
    L_chol = torch.linalg.cholesky(Bmat)
    T1     = torch.linalg.solve_triangular(L_chol, Kc,                   upper=False)
    T2     = torch.linalg.solve_triangular(L_chol, T1.transpose(-1, -2), upper=False)
    A      = T2.transpose(-1, -2)
    A      = 0.5 * (A + A.transpose(-1, -2))

    return torch.linalg.eigvalsh(A)[:, -1].detach().cpu().numpy()


# --- Compute Poincaré Constants ---

AVG_SIM = X_OTF.shape[0]

# --- Step 1: Reconstruct priors ---

X_prior_all = np.zeros((AVG_SIM, N, L, J))

print(f"\nReconstructing priors | AVG_SIM={AVG_SIM}  N={N}  J={J}")
for k in range(AVG_SIM):
    for i in range(1, N):
        X_prev = X_OTF[k, i - 1]  # (L, J)
        proc   = np.random.multivariate_normal(np.zeros(L), sigmma ** 2 * np.eye(L), J)  # (J, L)
        if rk4:
            X_prior_all[k, i] = rk4_step(L63, t[i - 1], X_prev, tau) + proc.T   # (L, J)
        else:
            X_prior_all[k, i] = X_prev + L63(t[i - 1], X_prev) * tau + proc.T   # (L, J)
    print(f"  sim {k + 1}/{AVG_SIM} done")

# --- Step 2: Batch Poincaré estimation ---

def compute_series_gpu(X_prior_all, X_post_all):
    """Run both estimators for all (sim, step) pairs and return time-averaged series."""
    lb_pr   = np.full((AVG_SIM, N), np.nan)
    lb_po   = np.full((AVG_SIM, N), np.nan)
    rkhs_pr = np.full((AVG_SIM, N), np.nan)
    rkhs_po = np.full((AVG_SIM, N), np.nan)

    idx_all   = [(k, i) for k in range(AVG_SIM) for i in range(1, N)]
    n_batches = (len(idx_all) + BATCH_SIZE_GPU - 1) // BATCH_SIZE_GPU
    print(f"\nOTF: {len(idx_all)} (sim, step) pairs  batch_size={BATCH_SIZE_GPU}  n_batches={n_batches}")

    # Lower bound (CPU, vectorized per simulation)
    for k in range(AVG_SIM):
        steps           = np.arange(1, N)
        lb_pr[k, steps] = poincare_LB_batch(X_prior_all[k, steps])
        lb_po[k, steps] = poincare_LB_batch(X_post_all[k,  steps])

    # RKHS estimator (GPU batched)
    t0 = time.time()
    for b in range(n_batches):
        chunk = idx_all[b * BATCH_SIZE_GPU:(b + 1) * BATCH_SIZE_GPU]
        ks    = np.array([c[0] for c in chunk])
        iss   = np.array([c[1] for c in chunk])

        rkhs_pr[ks, iss] = rkhs_poincare_batch_gpu(X_prior_all[ks, iss])
        rkhs_po[ks, iss] = rkhs_poincare_batch_gpu(X_post_all[ks,  iss])

        if (b + 1) % 10 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            rate    = (b + 1) / elapsed
            eta     = (n_batches - b - 1) / rate if rate > 0 else float('nan')
            print(f"  batch {b + 1}/{n_batches}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    return {
        'lb_prior':   np.nanmean(lb_pr,   axis=0),
        'lb_post':    np.nanmean(lb_po,   axis=0),
        'rkhs_prior': np.nanmean(rkhs_pr, axis=0),
        'rkhs_post':  np.nanmean(rkhs_po, axis=0),
    }


otf = compute_series_gpu(X_prior_all, X_OTF)

LB_prior_otf   = otf['lb_prior']
LB_post_otf    = otf['lb_post']
RKHS_prior_otf = otf['rkhs_prior']
RKHS_post_otf  = otf['rkhs_post']

print(f"\n--- λ=0 (avg over {AVG_SIM} sims) ---")
print(f"Mean LB   P_prior: {np.nanmean(LB_prior_otf[1:]):.3f}   P_post: {np.nanmean(LB_post_otf[1:]):.3f}")
print(f"Mean RKHS P_prior: {np.nanmean(RKHS_prior_otf[1:]):.3f}   P_post: {np.nanmean(RKHS_post_otf[1:]):.3f}")
# print(f"\n--- λ=0.1 ---")
# print(f"Mean LB   P_prior: {np.nanmean(LB_prior_otf_reg[1:]):.3f}   P_post: {np.nanmean(LB_post_otf_reg[1:]):.3f}")
# print(f"Mean RKHS P_prior: {np.nanmean(RKHS_prior_otf_reg[1:]):.3f}   P_post: {np.nanmean(RKHS_post_otf_reg[1:]):.3f}")


# --- Plot ---

sv = t[1:]

lb_pr_otf      = LB_prior_otf[1:]
lb_po_otf      = LB_post_otf[1:]
# lb_pr_otf_reg  = LB_prior_otf_reg[1:]
# lb_po_otf_reg  = LB_post_otf_reg[1:]

rk_pr_otf      = RKHS_prior_otf[1:]
rk_po_otf      = RKHS_post_otf[1:]
# rk_pr_otf_reg  = RKHS_prior_otf_reg[1:]
# rk_po_otf_reg  = RKHS_post_otf_reg[1:]

def _mask(a, b):
    return np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)

m_lb_otf      = _mask(lb_pr_otf,     lb_po_otf)
m_rk_otf      = _mask(rk_pr_otf,     rk_po_otf)
# m_lb_otf_reg  = _mask(lb_pr_otf_reg, lb_po_otf_reg)
# m_rk_otf_reg  = _mask(rk_pr_otf_reg, rk_po_otf_reg)

fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))

# --- OTF (λ=0) ---
ax1.semilogy(sv[m_lb_otf], lb_pr_otf[m_lb_otf], color='C3', linestyle='--', lw=2.5, label=r'LB Prior')
ax1.semilogy(sv[m_lb_otf], lb_po_otf[m_lb_otf], color='C4', linestyle='-',  lw=2.5, label=r'LB Posterior')
ax1.semilogy(sv[m_rk_otf], rk_pr_otf[m_rk_otf], color='C5', linestyle='-.', lw=2.5, label=r'RKHS Prior')
ax1.semilogy(sv[m_rk_otf], rk_po_otf[m_rk_otf], color='C6', linestyle=':',  lw=2.5, label=r'RKHS Posterior')
ax1.legend(loc='best', fontsize=fontsize)
if labeling: ax1.set_ylabel(r'$\hat{P}$', fontsize=fontsize)
if labeling: ax1.set_xlabel(r'$time$',                     fontsize=fontsize)

# # --- OTF (λ=0.1) ---
# ax2.semilogy(sv[m_lb_otf_reg], lb_pr_otf_reg[m_lb_otf_reg], color='C4', linestyle='--', lw=2.5, label=r'LB Prior $\hat{\eta}_t$')
# ax2.semilogy(sv[m_lb_otf_reg], lb_po_otf_reg[m_lb_otf_reg], color='C4', linestyle='-',  lw=2.5, label=r'LB Posterior $\hat{\pi}_t$')
# ax2.semilogy(sv[m_rk_otf_reg], rk_pr_otf_reg[m_rk_otf_reg], color='C4', linestyle='-.', lw=2.5, label=r'RKHS Prior $\hat{\eta}_t$')
# ax2.semilogy(sv[m_rk_otf_reg], rk_po_otf_reg[m_rk_otf_reg], color='C4', linestyle=':',  lw=2.5, label=r'RKHS Posterior $\hat{\pi}_t$')
# ax2.legend(loc='upper right', fontsize=fontsize)
# if labeling: ax2.set_ylabel(r'$\hat{P}_\mu~(\lambda=0.1)$', fontsize=fontsize)
# if labeling: ax2.set_xlabel(r'$time$',                       fontsize=fontsize)

plt.savefig('poincare_otf.pdf', bbox_inches='tight', dpi=300)
plt.show()


# # --- Density Heatmaps ---

# n_bins   = 50
# n_ylabel = 3
# fontsize = 16

# y_lims = [[-30, 30], [-35, 35], [-20, 60]]

# for num_plot_state in range(L):
#     y_lim         = y_lims[num_plot_state]
#     position_bins = np.linspace(y_lim[0], y_lim[1], n_bins)

#     plt.figure(figsize=(6, 10))

#     # --- OTF (λ=0) ---
#     density_matrix = np.zeros((N, len(position_bins) - 1))
#     for step in range(N):
#         density, _ = np.histogram(X_OTF[sim, step, num_plot_state, :], bins=position_bins, density=True)
#         density_matrix[step, :] = density

#     plt.subplot(2, 1, 1)
#     sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
#     ax = plt.gca()
#     ax.invert_yaxis()
#     ax.get_xaxis().set_visible(False)
#     plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
#     if labeling: plt.ylabel(r'$OTF~(\lambda=0)$', fontsize=fontsize)
#     ax.spines['top'].set_visible(True)
#     ax.spines['right'].set_visible(True)
#     ax.spines['left'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)

#     # --- OTF (λ=0.1) ---
#     density_matrix = np.zeros((N, len(position_bins) - 1))
#     for step in range(N):
#         density, _ = np.histogram(X_OTF_reg[sim, step, num_plot_state, :], bins=position_bins, density=True)
#         density_matrix[step, :] = density

#     plt.subplot(2, 1, 2)
#     sns.heatmap(density_matrix.T, cmap='Purples', cbar=False, robust=True)
#     ax = plt.gca()
#     ax.invert_yaxis()
#     plt.yticks(ticks=np.linspace(0, n_bins, n_ylabel), labels=np.linspace(y_lim[0], y_lim[1], n_ylabel).astype(int))
#     plt.xticks(ticks=np.linspace(0, N, 11), labels=np.round(np.linspace(0, N*tau, 11), 1))
#     if labeling: plt.ylabel(r'$OTF~(\lambda=0.1)$', fontsize=fontsize)
#     if labeling: plt.xlabel(r'$time$',               fontsize=fontsize)
#     ax.spines['top'].set_visible(True)
#     ax.spines['right'].set_visible(True)
#     ax.spines['left'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)

#     plt.savefig(f'poincare_density_X{num_plot_state + 1}.pdf', bbox_inches='tight')

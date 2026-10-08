"""
@author: Niyizhen (Jenny) Jin 
@author: Mohammad Al-Jarrah

Empirical Poincaré constant for OTF on Lorenz 63.

Loads the OTF (λ=0) particles saved by main.py in DATA_file_L63.npz, reconstructs
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

plt.rc('font', size=19)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['savefig.transparent'] = False   # True would also blank the axes patch
matplotlib.rcParams['savefig.facecolor']   = 'none'  # transparent outside the axes
matplotlib.rcParams['savefig.edgecolor']   = 'none'
matplotlib.rcParams['axes.facecolor']      = 'white' # white inside the axes
matplotlib.rcParams['figure.facecolor']    = 'none'
fontsize = 25  # font size for the axis labels and legend

SEED = 42
np.random.seed(SEED)  # fixes the process noise drawn when reconstructing the priors


# --- Load Data ---

# DATA_file_L63.npz is written by main.py. Only the arrays needed here are read: the file
# also holds the large-SIR reference (~1.2 GB), so it is not loaded in full with dict(np.load(...)).
with np.load('DATA_file_L63.npz') as data:
    t      = data['t']               # time grid, (N,)
    tau    = float(data['tau'])      # time step size
    rk4    = bool(data['rk4'])       # True: RK4 integration, False: forward Euler
    Noise  = data['Noise']           # [sigmma, gamma] noise vector
    X_OTF  = data['X_OTF']           # OTF (λ=0) filtered particles, (AVG_SIM, N, L, J)
sigmma = float(Noise[0])             # process noise std

N = X_OTF.shape[1]  # number of time steps
L = X_OTF.shape[2]  # state dimension
J = X_OTF.shape[3]  # ensemble size

labeling = True  # set False to hide all axis labels


# --- GPU Setup ---

BATCH_SIZE_GPU = 25             # (sim, step) pairs per GPU batch; each holds J x J kernel matrices
DTYPE          = torch.float64  # double precision keeps the Cholesky factorization stable
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# --- Model Definition (for prior propagation) ---

def L63(t, x):
    """Evaluate the Lorenz 63 vector field at state x and time t."""
    d     = np.zeros_like(x)  # output derivative vector
    sigma = 10                 # standard L63 coefficient
    r     = 28                 # standard L63 coefficient
    b     = 8 / 3              # standard L63 coefficient
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
    Xc = X - X.mean(axis=1, keepdims=True)                 # centered particles
    S  = np.einsum('bjl,bjm->blm', Xc, Xc) / X.shape[1]  # empirical covariance, (B, L, L)
    return np.linalg.eigvalsh(S).max(axis=-1)


_eye_cache = {}  # off-diagonal masks keyed by ensemble size, built once per size

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

    off_vals = sqdist.masked_select(off_diag_mask).view(B, -1)                  # pairwise distances, self-pairs excluded
    h        = (off_vals.median(dim=1).values / 2.0).sqrt().clamp_min(1e-8)  # median-heuristic bandwidth
    h2       = (h ** 2).view(B, 1, 1)

    lam = float(n) ** (-0.25)  # ridge regularization, decays with the ensemble size

    K  = torch.exp(-sqdist / (2 * h2))  # RBF Gram matrix, (B, n, n)
    # Doubly centered Gram matrix (feature-space covariance)
    Kc = K - K.mean(dim=2, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean(dim=(1, 2), keepdim=True)

    G  = -(diff / h2.unsqueeze(-1)) * K.unsqueeze(-1)  # kernel gradients, (B, n, n, d)
    LK = torch.einsum('bkid,bkjd->bij', G, G) / n     # empirical gradient (Dirichlet) Gram matrix, (B, n, n)

    Bmat   = LK + lam * torch.eye(n, dtype=DTYPE, device=device).unsqueeze(0)
    L_chol = torch.linalg.cholesky(Bmat)
    T1     = torch.linalg.solve_triangular(L_chol, Kc,                   upper=False)
    T2     = torch.linalg.solve_triangular(L_chol, T1.transpose(-1, -2), upper=False)
    A      = T2.transpose(-1, -2)                 # L^{-1} Kc L^{-T}
    A      = 0.5 * (A + A.transpose(-1, -2))      # symmetrize against round-off before eigvalsh

    return torch.linalg.eigvalsh(A)[:, -1].detach().cpu().numpy()


# --- Compute Poincaré Constants ---

AVG_SIM = X_OTF.shape[0]  # number of simulation runs

# --- Step 1: Reconstruct priors ---

# Prior at step i = previous posterior propagated one step plus process noise; step 0 stays zero
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

    idx_all   = [(k, i) for k in range(AVG_SIM) for i in range(1, N)]  # t = 0 is skipped: no prior there
    n_batches = (len(idx_all) + BATCH_SIZE_GPU - 1) // BATCH_SIZE_GPU   # ceiling division
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
        ks    = np.array([c[0] for c in chunk])  # simulation indices of the batch
        iss   = np.array([c[1] for c in chunk])  # time-step indices of the batch

        rkhs_pr[ks, iss] = rkhs_poincare_batch_gpu(X_prior_all[ks, iss])
        rkhs_po[ks, iss] = rkhs_poincare_batch_gpu(X_post_all[ks,  iss])

        if (b + 1) % 10 == 0 or b == n_batches - 1:
            elapsed = time.time() - t0
            rate    = (b + 1) / elapsed
            eta     = (n_batches - b - 1) / rate if rate > 0 else float('nan')  # estimated seconds remaining
            print(f"  batch {b + 1}/{n_batches}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

    return {
        'lb_prior_sims':   lb_pr,    # per-simulation series, (AVG_SIM, N); t = 0 is NaN
        'lb_post_sims':    lb_po,
        'rkhs_prior_sims': rkhs_pr,
        'rkhs_post_sims':  rkhs_po,
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

# Save the plotted series (and the per-simulation values) so the figure can be redrawn without recomputing
np.savez('poincare_otf_data.npz',
         t               = t,
         LB_prior_otf    = LB_prior_otf,      # averaged over simulations, (N,)
         LB_post_otf     = LB_post_otf,
         RKHS_prior_otf  = RKHS_prior_otf,
         RKHS_post_otf   = RKHS_post_otf,
         LB_prior_sims   = otf['lb_prior_sims'],    # before averaging, (AVG_SIM, N)
         LB_post_sims    = otf['lb_post_sims'],
         RKHS_prior_sims = otf['rkhs_prior_sims'],
         RKHS_post_sims  = otf['rkhs_post_sims'],
         SEED            = SEED)

print(f"\n--- λ=0 (avg over {AVG_SIM} sims) ---")
print(f"Mean LB   P_prior: {np.nanmean(LB_prior_otf[1:]):.3f}   P_post: {np.nanmean(LB_post_otf[1:]):.3f}")
print(f"Mean RKHS P_prior: {np.nanmean(RKHS_prior_otf[1:]):.3f}   P_post: {np.nanmean(RKHS_post_otf[1:]):.3f}")


# --- Plot ---

sv = t[1:]  # t = 0 is skipped: no prior is reconstructed there

lb_pr_otf = LB_prior_otf[1:]
lb_po_otf = LB_post_otf[1:]
rk_pr_otf = RKHS_prior_otf[1:]
rk_po_otf = RKHS_post_otf[1:]

def _mask(a, b):
    """Keep time steps where both series are finite and positive, as required by the log-scale axis."""
    return np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)

m_lb_otf = _mask(lb_pr_otf, lb_po_otf)
m_rk_otf = _mask(rk_pr_otf, rk_po_otf)

fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))

# --- OTF (λ=0) ---
ax1.semilogy(sv[m_lb_otf], lb_pr_otf[m_lb_otf], color='C3', linestyle='-',  lw=4,   label=r'LB Prior')
ax1.semilogy(sv[m_lb_otf], lb_po_otf[m_lb_otf], color='C4', linestyle='--', lw=4,   label=r'LB Posterior')
ax1.semilogy(sv[m_rk_otf], rk_pr_otf[m_rk_otf], color='C5', linestyle='-.', lw=4,   label=r'RKHS Prior')
ax1.semilogy(sv[m_rk_otf], rk_po_otf[m_rk_otf], color='C6', linestyle=':',  lw=4,   label=r'RKHS Posterior')
ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=fontsize)
if labeling: ax1.set_ylabel(r'$\hat{P}$', fontsize=fontsize)
if labeling: ax1.set_xlabel(r'$time$',    fontsize=fontsize)

fig.tight_layout()
fig.savefig('poincare_otf.pdf', bbox_inches='tight', dpi=300)
plt.show()

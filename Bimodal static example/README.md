# Bimodal Static Example

Demonstrates the OTF on a static problem with the quadratic observation model
`y = 0.5 x² + noise`. The prior is standard Gaussian and, since the sign of `x` is
unobservable, the posterior is bimodal, making this a challenging test case for
Gaussian-based methods. `bimodal_static_example.py` solves the 1-D problem; the other
scripts apply the same model to every component of a d-dimensional state.

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned hyperparameters used by the Figure 4 and supplementary marginal scripts |
| `bimodal_static_example.py` | **Figure 3.** Trains OTF and plots transported density, Kantorovich potential, and transport map for four regularization strengths (λ = 0, 0.01, 0.1, 1) |
| `error_and_time_vs_dim.py` | Sweeps state dimension d ∈ {2, 4, 6, 8, 10, 15, 20, 30, 40, 50}; computes SW₂ error and compute time vs. dimension for OTF, EnKF, and SIR and saves the results |
| `error_vs_particles.py` | Fixes d (2 or 10) and sweeps ensemble size N ∈ {100, …, 500 000}; computes SW₂ error vs. number of particles and saves the results |
| `Import_Data.py` | **Figure 4.** Plotting only: loads the saved sweep results and generates the four Figure 4 subfigures |
| `plotting_marginals.py` | **Supplementary material.** Trains OTF for a chosen dimension d and plots per-state marginal histograms comparing the prior, OTF posterior, and SIR reference |
| `smac_tuning_Bimodal.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |

---

## Running Order

`bimodal_static_example.py` (Figure 3) and `plotting_marginals.py` (supplementary marginal
figures) are standalone. The Figure 4 subfigures are produced in two steps: first generate
the data, then plot it.

```bash
python bimodal_static_example.py
python plotting_marginals.py

# Figure 4: generate the data (long GPU runs) ...
python error_and_time_vs_dim.py
python error_vs_particles.py      # with d = 2
python error_vs_particles.py      # with d = 10
# ... then plot it (fast, no computation)
python Import_Data.py
```

To re-run hyperparameter tuning from scratch:

```bash
python smac_tuning_Bimodal.py
# update param.py with the best config printed at the end
```

---

## Expected Outputs

| Script | Output file |
|---|---|
| `bimodal_static_example.py` | `bimodal_static_example.pdf` |
| `error_and_time_vs_dim.py` | `error_and_time_vs_dim.pdf` (quick-look plot) |
| `error_vs_particles.py` | `error_vs_particles_for_dim_<d>.pdf` (quick-look plot) |
| `Import_Data.py` | `bimodal_example_vs_dim.pdf`, `bimodal_example_vs_time.pdf`, `bimodal_example_vs_particles_2d.pdf`, `bimodal_example_vs_particles_10d.pdf` |
| `plotting_marginals.py` | `marginals_d<d>.pdf` |

---

## Notes

- `error_and_time_vs_dim.py` and `error_vs_particles.py` run AVG_SIM = 10 independent trials per configuration and average SW₂. Runtime scales with the number of dimensions / particle counts swept. Both use a fixed seed so the sweeps reproduce.
- `error_vs_particles.py` must be run **twice** — once with `d = 2` and once with `d = 10` (**line 269**) — to produce the two particle-count panels.
- `Import_Data.py` needs the results of `error_and_time_vs_dim.py` and of both `error_vs_particles.py` runs (d = 2 and d = 10).
- The SMAC tuning script builds its reference data once on the first run and caches it, so worker processes do not regenerate data independently. Its run name is fixed, so re-running it resumes an existing tuning run.
- The three regularization strengths λ = 0, 0.01, 0.1 are evaluated in a single run of each sweep script; `bimodal_static_example.py` additionally includes λ = 1.
- To plot marginals for a different state dimension, change `d` on **line 80** of `plotting_marginals.py`.

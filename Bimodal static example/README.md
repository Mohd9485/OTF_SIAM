# Bimodal Static Example

Demonstrates the OTF on a static 1-D problem with the quadratic observation model
`y = 0.5 x² + noise`. The prior is standard Gaussian and the posterior is bimodal,
making this a challenging test case for Gaussian-based methods.

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned hyperparameters used by the Figure 2 scripts |
| `bimodal_static_example.py` | Trains OTF and plots transported density, Kantorovich potential, and transport map for four regularization strengths (λ = 0, 0.01, 0.1, 1) |
| `error_and_time_vs_dim.py` | Sweeps state dimension d ∈ {2, 4, …, 50} and plots AA-SW₂ error and compute time vs. dimension for OTF, EnKF, and SIR |
| `error_vs_particles.py` | Fixes d = 10 and sweeps ensemble size N ∈ {100, …, 500 000}; plots AA-SW₂ error vs. number of particles |
| `smac_tuning_Bimodal.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |
| `Import_Data.py` | Loads saved `.pkl` files and generates the combined 4-panel Figure 2 and individual subfigures |
| `plotting_marginals.py` | Trains OTF for a chosen dimension d and plots per-state marginal histograms comparing the prior, OTF posterior, and SIR reference |

---

## Running Order

The three Figure scripts are independent and can be run in any order:

```bash
python bimodal_static_example.py
python error_and_time_vs_dim.py
python error_vs_particles.py
python plotting_marginals.py
```

To re-run hyperparameter tuning from scratch:

```bash
python smac_tuning_Bimodal.py   # writes results to smac3_output/
# update param.py with the best config printed at the end
```

---

## Expected Outputs

| Script | Output file |
|---|---|
| `bimodal_static_example.py` | `bimodal_static_example.pdf` |
| `error_and_time_vs_dim.py` | `bimodal_example_vs_dim.pdf`, `bimodal_example_vs_time.pdf` |
| `error_vs_particles.py` | `bimodal_example_vs_particles_2d.pdf`, `bimodal_example_vs_particles_10d.pdf` |
| `plotting_marginals.py` | `marginals_d<d>.pdf` |

---

## Notes

- Each Figure script runs AVG_SIM = 10 independent trials per configuration and averages AA-SW₂. Runtime scales with the number of dimensions / particle counts swept.
- `error_vs_particles.py` must be run **twice** — once with `d = 2` and once with `d = 10` (line 226) — to produce the two particle-count panels.
- The SMAC tuning script uses a shared data cache (`smac_shared_data_d<d>.npz`) written on first run so worker processes do not regenerate data independently.
- All three regularization strengths (λ = 0, 0.01, 0.1) are evaluated in a single run.
- To plot marginals for a different state dimension, change `d` on **line 69** of `plotting_marginals.py`.

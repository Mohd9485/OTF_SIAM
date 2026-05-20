# Bimodal Static Example

Demonstrates the OTF on a static 1-D problem with the quadratic observation model
`y = 0.5 x² + noise`. The prior is standard Gaussian and the posterior is bimodal,
making this a challenging test case for Gaussian-based methods.

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned hyperparameters used by the Figure 2 scripts |
| `Figure_1_Bimodal_static_example.py` | Trains OTF and plots transported density, Kantorovich potential, and transport map for three regularization strengths (λ = 0, 0.01, 0.1) |
| `Figure_2_W2_vs_dim_and_time.py` | Sweeps state dimension d ∈ {2, 4, …, 50} and plots W₂ error and compute time vs. dimension for OTF, EnKF, and SIR |
| `Figure_2_W2_vs_particles.py` | Fixes d = 10 and sweeps ensemble size N ∈ {100, …, 500 000}; plots W₂ error vs. number of particles |
| `smac_tuning_Bimodal.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |

---

## Running Order

The three Figure scripts are independent and can be run in any order:

```bash
python Figure_1_Bimodal_static_example.py
python Figure_2_W2_vs_dim_and_time.py
python Figure_2_W2_vs_particles.py
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
| `Figure_1_Bimodal_static_example.py` | `Figure_1_Bimodal_static_example_<timestamp>.pdf` |
| `Figure_2_W2_vs_dim_and_time.py` | `Figure_2_W2_vs_dim.pdf` |
| `Figure_2_W2_vs_particles.py` | `Figure_2_W2_vs_particles_for_dim_10.pdf` |

---

## Notes

- Each Figure 2 script runs AVG_SIM = 10 independent trials per configuration and averages W₂. Runtime scales with the number of dimensions / particle counts swept.
- The SMAC tuning script uses a shared data cache (`smac_shared_data_d<d>.npz`) written on first run so worker processes do not regenerate data independently.
- All three regularization strengths (λ = 0, 0.01, 0.1) are evaluated in a single run.

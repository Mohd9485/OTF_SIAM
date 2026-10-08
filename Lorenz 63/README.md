# Lorenz 63

Sequential filtering experiment on the 3-dimensional Lorenz 63 chaotic system.
The state x = (x₁, x₂, x₃) evolves under the L63 equations and only x₃ is observed.
Four filters are compared: EnKF, SIR, OTF (λ = 0), and OTF (λ = 0.1).

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned OTF hyperparameters |
| `main.py` | Generates AVG_SIM = 10 independent trajectories, runs all four filters, computes the W₂ distance of each filter to a large-particle SIR reference, and saves the results |
| `OTF_Poincare.py` | **Figure 6 (data).** Estimates the empirical Poincaré constant of the OTF (λ = 0) prior and posterior over time (covariance lower bound and RKHS estimator) and saves the results |
| `Import_DATA_L63.py` | **Figures 5 and 6.** Plotting only: loads the saved results and produces all figures |
| `OTF.py` | Optimal Transport Filter implementation |
| `EnKF.py` | Ensemble Kalman Filter implementation |
| `SIR.py` | Sequential Importance Resampling particle filter |
| `smac_tuning_L63.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |

---

## Running Order

**Step 1 — Generate data, run all filters, and compute W₂:**

```bash
python main.py
```

Runtime depends on hardware; the two OTF runs dominate (about 2 hours on two GPUs), followed by the W₂ computation (about 30 minutes).

**Step 2 — Estimate the Poincaré constants:**

```bash
python OTF_Poincare.py
```

This uses the OTF (λ = 0) particles from Step 1 and takes a few minutes on a GPU.

**Step 3 — Plot results:**

```bash
python Import_DATA_L63.py
```

This only loads the saved results and plots them; it does no filtering or W₂ computation, so it runs in seconds.

---

## Expected Outputs

| Paper figure | Panel | Output file |
|---|---|---|
| Figure 5 | W₂ vs. time | `L63_w2_vs_time.pdf` |
| Figure 5 | Density heatmap — state x₁ | `L63_X1.pdf` |
| Figure 5 | Density heatmap — state x₂ | `L63_X2.pdf` |
| Figure 5 | Density heatmap — state x₃ | `L63_X3.pdf` |
| Figure 6 | Empirical Poincaré constant vs. time | `poincare_otf.pdf` |

---

## Re-tuning Hyperparameters

```bash
python smac_tuning_L63.py
# update param.py with the best config printed at the end
```

---

## Notes

- `main.py` runs OTF (λ = 0) and OTF (λ = 0.1) concurrently on two separate GPU devices using `ThreadPoolExecutor`. On a single-GPU or CPU machine both threads share the same device.
- The W₂ reference is an SIR filter with 10⁵ particles, run once per simulation from the same prior as the filters; the exact W₂ is computed against 10³ of its particles at every time step. W₂ is stored per time step and per simulation, and averaged over simulations only when plotting.
- The iteration budget for the OTF network starts at `ITERATION = 512` and is halved at every time step after the first five, down to a floor of 64, reducing cost as the filter warms up.
- The SMAC tuning script builds its datasets and SIR references once on the first run and caches them, so every trial is evaluated on the same data. Its run name is fixed, so re-running it resumes an existing tuning run.

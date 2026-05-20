# Lorenz 63

Sequential filtering experiment on the 3-dimensional Lorenz 63 chaotic system.
The state x = (x₁, x₂, x₃) evolves under the L63 equations and only x₃ is observed.
Four filters are compared: OTF (λ = 0), OTF_reg (λ = 0.1), EnKF, and SIR.

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned OTF hyperparameters |
| `main.py` | Generates AVG_SIM = 10 independent trajectories, runs all four filters, and saves results to `DATA_file_L63.npz` |
| `Import_DATA_L63.py` | Loads `DATA_file_L63.npz`, re-runs EnKF and SIR for W₂ evaluation, and produces all plots |
| `OTF.py` | Optimal Transport Filter implementation |
| `EnKF.py` | Ensemble Kalman Filter implementation |
| `SIR.py` | Sequential Importance Resampling particle filter |
| `smac_tuning_L63.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |

---

## Running Order

**Step 1 — Generate data and run all filters:**

```bash
python main.py
```

This creates `DATA_file_L63.npz` in the same directory. Runtime depends on hardware; expect several minutes with a GPU.

**Step 2 — Plot results:**

```bash
python Import_DATA_L63.py
```

This re-runs EnKF and SIR (fast) for W₂ evaluation against a large-SIR reference, then saves all figures.

---

## Expected Outputs

| Figure | Output file |
|---|---|
| W₂ vs. time | `L63_w2_vs_time.pdf` |
| Density heatmap — state x₁ | `L63_X1_vs_time.pdf` |
| Density heatmap — state x₂ | `L63_X2_vs_time.pdf` |
| Density heatmap — state x₃ | `L63_X3_vs_time.pdf` |

---

## Re-tuning Hyperparameters

```bash
python smac_tuning_L63.py   # writes results to smac3_output/
# update param.py with the best config printed at the end
```

---

## Notes

- `main.py` runs OTF (λ = 0) and OTF_reg (λ = 0.1) concurrently on two separate GPU devices using `ThreadPoolExecutor`. On a single-GPU or CPU machine both threads share the same device.
- The iteration budget for the OTF network is halved at each time step (starting from `ITERATION = 512`) down to a floor of 64, reducing cost as the filter warms up.
- `Import_DATA_L63.py` computes W₂ using a large-SIR reference (10⁶ particles) run separately per simulation, which is the dominant runtime of that script.

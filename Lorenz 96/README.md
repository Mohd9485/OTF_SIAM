# Lorenz 96

Sequential filtering experiment on the 9-dimensional Lorenz 96 chaotic system.
Every third component is observed (indices 0, 3, 6), giving a 3-dimensional
observation space. Six filters are compared: EnKF, SIR, OTF (λ = 0), OTF (λ = 0.1),
OTF_EnKF (λ = 0), and OTF_EnKF (λ = 0.1). RK4 integration is used throughout.

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned hyperparameters for OTF and OTF_EnKF (separate learning rates and neuron counts for f and T) |
| `main.py` | Generates AVG_SIM = 10 independent trajectories, runs all six filters, prints their MSE, and saves the results |
| `Import_DATA_L96.py` | **Figure 7.** Plotting only: loads the saved results, computes the MSE, and produces all figures |
| `OTF.py` | Optimal Transport Filter |
| `OTF_EnKF.py` | OTF_EnKF: OTF whose transport map starts from an EnKF update (EnKF warm start) and learns a correction on top of it |
| `EnKF.py` | Ensemble Kalman Filter |
| `SIR.py` | Sequential Importance Resampling particle filter |
| `smac_tuning_L96.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |

---

## Running Order

**Step 1 — Generate data and run all filters:**

```bash
python main.py
```

The OTF runs dominate the runtime: about 6 hours for the two OTF filters and under 1 hour for the two OTF_EnKF filters, each pair running concurrently on two GPUs.

**Step 2 — Plot results:**

```bash
python Import_DATA_L96.py
```

This only loads the saved results and plots them, so it runs in seconds. It also prints the mean MSE and the computational time per simulation of every filter.

---

## Expected Outputs

| Paper figure | Panel | Output file |
|---|---|---|
| Figure 7 | Ensemble trajectories — state x₁ | `L96_x1.pdf` |
| Figure 7 | Ensemble trajectories — state x₂ | `L96_x2.pdf` |
| Figure 7 | Ensemble trajectories — state x₃ | `L96_x3.pdf` |
| Figure 7 | MSE vs. time | `L96_mse.pdf` |

The trajectory figures show EnKF, SIR, OTF (λ = 0), and OTF_EnKF (λ = 0) against the true state; the MSE figure shows all six filters.

---

## Re-tuning Hyperparameters

```bash
python smac_tuning_L96.py
# update param.py with the best config printed at the end
```

The script tunes OTF (λ = 0); swap the commented import at the top to tune OTF_EnKF instead.

---

## Notes

- Unlike L63, the L96 filters use **separate learning rates** for f and T, and OTF_EnKF adds an **EnKF warm start** inside the transport map network, which helps convergence in higher dimensions.
- `main.py` runs OTF (λ = 0) and OTF (λ = 0.1) concurrently on two GPU devices using `ThreadPoolExecutor`, then OTF_EnKF (λ = 0) and OTF_EnKF (λ = 0.1) the same way. On a single GPU or CPU both threads share the same device.
- The vectorized dynamics function `ML96` propagates the full ensemble in one RK4 call; the scalar `L96` is used for data generation only.
- Ensemble size is J = 1000 for all filters. OTF trains for 512 iterations at every assimilation step; for OTF_EnKF the iteration budget starts at 512 and is halved at every step down to a floor of 64.
- The SMAC tuning script's run name is fixed, so re-running it resumes an existing tuning run.

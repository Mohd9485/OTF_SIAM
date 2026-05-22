# Lorenz 96

Sequential filtering experiment on the 9-dimensional Lorenz 96 chaotic system.
Every third component is observed (indices 0, 3, 6), giving a 3-dimensional
observation space. Four filters are compared: OTF (λ = 0), OTF_reg (λ = 0.1),
EnKF, and SIR. RK4 integration is used throughout for higher accuracy.

---

## Files

| File | Description |
|---|---|
| `param.py` | Best SMAC-tuned OTF hyperparameters (separate learning rates and neuron counts for f and T) |
| `main.py` | Generates AVG_SIM = 10 independent trajectories, runs all four filters, computes MSE, saves results to `DATA_file_L96.npz`, and plots state trajectories and MSE |
| `Import_DATA_L96.py` | Loads `DATA_file_L96.npz` and re-produces the publication-quality figures |
| `OTF.py` | Optimal Transport Filter with EnKF warm-start for the transport map |
| `EnKF.py` | Ensemble Kalman Filter with optional RK4 integration |
| `SIR.py` | Sequential Importance Resampling particle filter with optional RK4 integration |
| `smac_tuning_L96.py` | SMAC hyperparameter search (only needed to re-tune; results are already in `param.py`) |

---

## Running Order

**Step 1 — Generate data and run all filters:**

```bash
python main.py
```

This creates `DATA_file_L96.npz` in the same directory. The L96 experiment is more expensive than L63; expect 30–60 minutes on a GPU depending on ensemble size.

**Step 2 — Plot results:**

```bash
python Import_DATA_L96.py
```

Loads the saved data and writes publication-quality PDF figures.

---

## Expected Outputs

| Figure | Output file |
|---|---|
| State trajectory — component 1 | `L96_x1_reg.pdf` |
| State trajectory — component 2 | `L96_x2_reg.pdf` |
| State trajectory — component 3 | `L96_x3_reg.pdf` |
| MSE vs. time | `L96_mse_reg.pdf` |

MSE statistics are also printed to stdout:

```
MSE EnKF:   mean=...
MSE SIR:    mean=...
MSE OT:     mean=...
MSE OT_reg: mean=...
```

---

## Re-tuning Hyperparameters

```bash
python smac_tuning_L96.py   # writes results to smac3_output/
# update param.py with the best config printed at the end
```

---

## Notes

- Unlike L63, the L96 OTF uses **separate learning rates** for f and T and an **EnKF warm start** inside the transport map network, which helps convergence in higher dimensions.
- `main.py` runs OTF and OTF_reg concurrently on two GPU devices using `ThreadPoolExecutor`. On a single GPU or CPU both threads share the same device.
- The vectorized dynamics function `ML96` propagates the full ensemble matrix in one RK4 call; the scalar `L96` is used for data generation only.
- Ensemble size is J = 1000 (250 × 4) and the iteration budget decays from 512 down to 64 over assimilation steps.

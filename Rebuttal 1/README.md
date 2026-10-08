# Optimal Transport Filter (OTF)

Code accompanying the paper:

> **[Paper title]**
> Mohammad Al-Jarrah et al.
> *[Journal/Conference]*

The OTF is a sequential Bayesian filter that learns the optimal transport map from the forecast ensemble to the posterior at each assimilation step, using a dual-formulation neural network trained on-the-fly. It is compared against the Ensemble Kalman Filter (EnKF) and Sequential Importance Resampling (SIR) on three benchmarks.

---

## Repository Structure

```
SIAM_paper/
├── Bimodal static example/   # Static 1-D OT problem (Figures 1 & 2)
├── Lorenz 63/                # Sequential filtering on the L63 system (3-D chaotic)
└── Lorenz 96/                # Sequential filtering on the L96 system (9-D chaotic)
```

Each folder contains its own `README.md` with experiment-specific instructions.

---

## Dependencies

Install all requirements with:

```bash
pip install torch numpy scipy matplotlib seaborn POT smac ConfigSpace
```

| Package | Purpose |
|---|---|
| `torch` | Neural network training (OTF) |
| `numpy` / `scipy` | Numerics, RK45 integration |
| `matplotlib` / `seaborn` | Plotting |
| `POT` | Wasserstein distance computation |
| `smac` / `ConfigSpace` | Hyperparameter tuning (optional) |

> **Note:** GPU access is recommended for the Bimodal and SMAC experiments. The Lorenz experiments can run on CPU but will be significantly slower.

---

## Quick Start

Hyperparameters have already been tuned and stored in each folder's `param.py`. To reproduce the figures directly:

```bash
# Figure 1 and Figure 2 (Bimodal)
cd "Bimodal static example"
python Figure_1_Bimodal_static_example.py
python Figure_2_W2_vs_dim_and_time.py
python Figure_2_W2_vs_particles.py

# Lorenz 63 results
cd "../Lorenz 63"
python main.py          # generates DATA_file_L63.npz
python Import_DATA_L63.py

# Lorenz 96 results
cd "../Lorenz 96"
python main.py          # generates DATA_file_L96.npz
python Import_DATA_L96.py
```

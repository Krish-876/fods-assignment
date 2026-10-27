# CS F320 Foundations of Data Science – Assignment 1

| Name | ID |
|---|---|
| B.N.R. Krishna | 2024A7PS0175H |
| Sudhanva B H | 20240085H |

Polynomial regression, ridge/lasso regularisation, and Bayesian regression,
implemented from scratch in NumPy (no scikit-learn).

---

## Folder Structure

```
fods-assignment/
  A1-datasets/               # original data (do not modify)
    poly_train.csv           # N=10 training points
    poly_test.csv            # N=100 test points
    p4_train_N5.csv          # Bayesian regression datasets
    p4_train_N10.csv
    p4_train_N20.csv
    p4_train_N50.csv
  common.py                  # shared helpers (paths, loaders, metrics, I/O)
  p1_gd_family.py            # P1: closed-form + GD family
  p2_ridge_map.py            # P2: ridge as MAP
  p3_lasso_subgradient.py    # P3: lasso via subgradient descent
  p4_bayesian_regression.py  # P4: Bayesian linear regression
  requirements.txt
  README.md
  outputs/
    p1/                      # figures + results.txt for P1
    p2/                      # figures + results.txt for P2
    p3/                      # figures + results.txt for P3
    p4/                      # figures + results.txt for P4
```

> **Note:** The assignment PDF may refer to files with spaces (e.g. `poly train.csv`).
> The actual files on disk use **underscores** (`poly_train.csv`). The code uses the
> underscore names.

---

## Setup

A virtual environment already exists at `.venv`. Activate it, then install dependencies:

```powershell
# Activate the existing venv
.\.venv\Scripts\Activate.ps1

# Install exact versions
pip install -r requirements.txt
```

---

## How to Run

Run each problem script from the project root:

```powershell
# P1 – Polynomial regression, closed-form vs GD family
python p1_gd_family.py

# P2 – Ridge regression as MAP
python p2_ridge_map.py

# P3 – Lasso via subgradient descent
python p3_lasso_subgradient.py

# P4 – Bayesian linear regression
python p4_bayesian_regression.py
```

### `--show` flag (interactive display)

Append `--show` to display figures interactively after saving:

```powershell
python p1_gd_family.py --show
```

### Build the report

```powershell
.\.venv\Scripts\python.exe build_report.py   # writes report.html (+ report.pdf if Edge/pandoc is available)
```

### Run all problems

```powershell
python p1_gd_family.py; python p2_ridge_map.py; python p3_lasso_subgradient.py; python p4_bayesian_regression.py
```

---

## Output Files

### P1 (`outputs/p1/`)

| File | Deliverable | Description |
|---|---|---|
| `p1_d1_closed_form_fits.png` | D1 | 3×3 subplot, degree 1–9 fits + sin curve |
| `p1_d3_loss_curves.png` | D3 | Batch GD / SGD / Mini-batch GD loss (log-y, η=0.1) |
| `p1_d4_lr_study.png` | D4 | Learning-rate study (η=0.01, 0.1, 2.1), 100 epochs |
| `results.txt` | all | All printed numbers, condition numbers, weight vectors, discussion |

### P2 (`outputs/p2/`)

| File | Deliverable | Description |
|---|---|---|
| `p2_d2_gd_convergence.png` | D2 | Ridge GD loss vs iteration (log-y, λ=1e-2) |
| `p2_d3_rmse_sweep.png` | D3 | Train/test RMSE vs λ, reference line at 0.1 |
| `p2_d4_fitted_curves.png` | D4 | Three panels: λ=1e-10, λ*, λ=1e-2 |
| `results.txt` | all | Condition numbers, λ*, RMSE, ‖w‖₂ values, discussion |

### P3 (`outputs/p3/`)

| File | Deliverable | Description |
|---|---|---|
| `p3_d1a_convergence.png` | D1(a) | Subgradient loss + best-so-far at λ=0.1 |
| `p3_d2_rmse_sweep.png` | D2 | Lasso train/test RMSE vs λ |
| `p3_d3_coef_path.png` | D3 | Coefficient path vs λ |
| `p3_d6_fitted_curves.png` | D6 | Lasso vs Ridge at best λ each |
| `results.txt` | all | Optimality conditions, zero-coef table, comparison table, discussion |

### P4 (`outputs/p4/`)

| File | Deliverable | Description |
|---|---|---|
| `p4_d1d2_predictive_N10.png` | D1+D2 | Posterior mean, 95% band, w_MAP overlay (N=10) |
| `p4_d3_interval_width.png` | D3 | Interval width 2·1.96·σ(x) for N=5,10,20,50 |
| `results.txt` | all | λ=α/β, ‖w_MAP−m_N‖, mean/max width per N, discussion |

---

## Seeds

| Script | Seed | Usage |
|---|---|---|
| `p1_gd_family.py` | 42 | SGD epoch shuffle; Mini-batch epoch shuffle |

P2, P3, P4 are deterministic (no random sampling).

---

## Numerical Notes

- **QR trick (P1):** Phi = QR (thin), then solve in the α-space where Q^TQ = I
  (all curvatures = 1). Weights recovered via `np.linalg.solve(R, alpha)`.
- **No explicit inverses for weight vectors:** All linear systems solved with
  `np.linalg.lstsq` (P1 closed form) or `np.linalg.solve` (P2, P4). The
  posterior covariance `S_N = inv(S_N_inv)` in P4 is computed via
  `np.linalg.inv` (acceptable for predictive variance).
- **~0 threshold (P3):** A Lasso coefficient w_j is counted as effectively zero
  iff |w_j| < 2·η·λ (twice the subgradient step times regularisation strength).
- **Condition numbers (P1):** cond(Phi) ≈ 1.5×10⁷ at d=9, confirming severe
  ill-conditioning of the raw Vandermonde matrix.
- **cond(Phi^T Phi) (P2):** ≈ 2.3×10¹⁴ without penalty; drops to ≈ 2×10³ at λ=0.01.
- **cond(Phi^T Phi) (P3, Legendre):** ≈ 6.1×10² — much better conditioned than
  the monomial basis.
- **MAP ↔ Ridge (P2, P4):** λ = α/β links the MAP weight to the ridge solution.
  Confirmed numerically: ‖w_MAP − m_N‖₂ ≈ 10⁻¹².

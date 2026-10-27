"""
p2_ridge_map.py
CS F320 Foundations of Data Science – Assignment 1, Problem 2

Ridge regression as MAP estimation.
  Objective: E~(w) = 0.5||Phi w - t||^2 + (lam/2)||w||^2
  RMSE = sqrt(mean squared error)

MAP -> Ridge derivation:
  Bayesian interpretation: Gaussian prior w ~ N(0, alpha^{-1} I), Gaussian noise
  t_n ~ N(phi(x_n)^T w, beta^{-1}).  The MAP estimate maximises the log-posterior:
    log p(w|t) = -0.5 beta ||Phi w - t||^2 - 0.5 alpha ||w||^2 + const
  Maximising this is equivalent to minimising
    0.5||Phi w - t||^2 + (alpha/(2*beta))||w||^2
  Setting lam = alpha/beta recovers the ridge objective.

Run:  python p2_ridge_map.py [--show]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_xy, vandermonde, rmse,
    ensure_outdir, save_fig, init_results, rprint, close_results,
    POLY_TRAIN, POLY_TEST
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "p2")

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

ensure_outdir(OUT_DIR)
init_results(OUT_DIR)

rprint("=" * 60)
rprint("P2: Ridge Regression as MAP")
rprint("=" * 60)

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
x_tr, t_tr = load_xy(POLY_TRAIN)
x_te, t_te = load_xy(POLY_TEST)
N_tr = len(x_tr)
N_te = len(x_te)
rprint(f"Train N={N_tr}, Test N={N_te}")

DEGREE = 9
Phi_tr = vandermonde(x_tr, DEGREE)   # N_tr x 10
Phi_te = vandermonde(x_te, DEGREE)   # N_te x 10

lams = np.logspace(-10, 2, 49)

# --------------------------------------------------------------------------- #
# Deliverable 1 – Closed-form ridge for every lam
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 1: Closed-form ridge (np.linalg.solve) ---")

PhiTt  = Phi_tr.T @ t_tr               # (d+1,)
PhiTPhi = Phi_tr.T @ Phi_tr            # (d+1) x (d+1)

def ridge_closed(lam):
    """Solve (Phi^T Phi + lam I) w = Phi^T t  using np.linalg.solve (no inverse)."""
    A = PhiTPhi + lam * np.eye(DEGREE + 1)
    return np.linalg.solve(A, PhiTt)

w_ridge = {lam: ridge_closed(lam) for lam in lams}

# --------------------------------------------------------------------------- #
# Deliverable 2 – GD agreement at lam=1e-2
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 2: GD agreement at lam=1e-2 ---")

LAM_GD = 1e-2
ETA_GD = 0.04
TOL    = 1e-6
MAX_ITER = 5_000_000

w_cf_gd = ridge_closed(LAM_GD)   # reference

# Precompute for efficiency
A_gd = PhiTPhi + LAM_GD * np.eye(DEGREE + 1)    # (d+1) x (d+1)
b_gd = PhiTt.copy()                               # (d+1,)

# Condition numbers
cond_PtP  = np.linalg.cond(PhiTPhi)
cond_PtPl = np.linalg.cond(A_gd)
rprint(f"  cond(Phi^T Phi)           = {cond_PtP:.4e}")
rprint(f"  cond(Phi^T Phi + lam I)   = {cond_PtPl:.4e}  (lam={LAM_GD})")
rprint("  The ridge penalty lam*I adds to every diagonal entry, raising the smallest "
       "singular value and thus shrinking the condition number.")

def ridge_obj(w, lam, Phi, t):
    r = Phi @ w - t
    return 0.5 * np.dot(r, r) + (lam / 2) * np.dot(w, w)

w_gd = np.zeros(DEGREE + 1)
loss_history_gd = [ridge_obj(w_gd, LAM_GD, Phi_tr, t_tr)]
norm_cf = np.linalg.norm(w_cf_gd)
converged = False
record_every = 100   # record every 100 iters to keep memory reasonable

for it in range(1, MAX_ITER + 1):
    grad = A_gd @ w_gd - b_gd          # efficient matvec
    w_gd = w_gd - ETA_GD * grad
    if it % record_every == 0:
        loss_history_gd.append(ridge_obj(w_gd, LAM_GD, Phi_tr, t_tr))
    rel_err = np.linalg.norm(w_gd - w_cf_gd) / norm_cf
    if rel_err < TOL:
        converged = True
        rprint(f"  GD converged at iteration {it}, rel_err={rel_err:.4e}")
        break

if not converged:
    rprint(f"  WARNING: GD did not converge within {MAX_ITER} iterations (rel_err={rel_err:.4e})")

rprint(f"  ||w_GD - w_closed||_2 / ||w_closed||_2 = {np.linalg.norm(w_gd - w_cf_gd)/norm_cf:.4e}")

# Plot convergence
fig, ax = plt.subplots(figsize=(8, 4))
iters_recorded = [i * record_every for i in range(len(loss_history_gd))]
iters_recorded[0] = 0   # first point is iter 0
ax.semilogy(iters_recorded, np.maximum(loss_history_gd, 1e-32), "b-", lw=1)
ax.set_xlabel("Iteration")
ax.set_ylabel("E~(w)  [log scale]")
ax.set_title(f"P2 D2: Ridge GD convergence  (lam={LAM_GD}, eta={ETA_GD})")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p2_d2_gd_convergence.png"), show=args.show)
rprint("  Saved: p2_d2_gd_convergence.png")

# --------------------------------------------------------------------------- #
# Deliverable 3 – RMSE sweep
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 3: RMSE sweep vs lam ---")

train_rmses = []
test_rmses  = []
for lam in lams:
    w = w_ridge[lam]
    train_rmses.append(rmse(t_tr, Phi_tr @ w))
    test_rmses.append(rmse(t_te, Phi_te @ w))

train_rmses = np.array(train_rmses)
test_rmses  = np.array(test_rmses)

best_idx  = np.argmin(test_rmses)
lam_star  = lams[best_idx]
best_test = test_rmses[best_idx]
thresh    = 1.05 * best_test
good_mask = test_rmses <= thresh
lam_lo    = lams[good_mask][0]
lam_hi    = lams[good_mask][-1]

rprint(f"  lam* = {lam_star:.4e}  ->  test RMSE = {best_test:.6f}")
rprint(f"  Interval where test RMSE <= 1.05 * min: [{lam_lo:.4e}, {lam_hi:.4e}]")
rprint("  The minimum is fairly flat (broad plateau), meaning a range of lam values "
       "give near-optimal test RMSE. Below lam_lo the model overfits (train RMSE falls "
       "but test RMSE rises sharply); above lam_hi the model underfits "
       "(both train and test RMSE increase as weights are shrunk too aggressively).")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(lams, train_rmses, "b-o", ms=3, lw=1, label="Train RMSE")
ax.semilogx(lams, test_rmses,  "r-o", ms=3, lw=1, label="Test RMSE")
ax.axhline(0.1, color="k", ls="--", lw=1, label="noise level 0.1")
ax.axvline(lam_star, color="g", ls=":", lw=1.5, label=f"lam*={lam_star:.2e}")
ax.set_xlabel("lam (log scale)")
ax.set_ylabel("RMSE")
ax.set_title("P2 D3: RMSE sweep vs regularisation lam")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p2_d3_rmse_sweep.png"), show=args.show)
rprint("  Saved: p2_d3_rmse_sweep.png")

# --------------------------------------------------------------------------- #
# Deliverable 4 – Fitted curves at three lam values
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 4: Fitted curves at lam=1e-10, lam*, lam=1e-2 ---")

x_grid  = np.linspace(0, 1, 300)
t_sin   = np.sin(2 * np.pi * x_grid)
Phi_grid = vandermonde(x_grid, DEGREE)

lam_triplet = [1e-10, lam_star, 1e-2]
titles = ["lam=1e-10 (overfit)", f"lam={lam_star:.2e} (best)", "lam=1e-2 (underfit)"]

# Find the closest lam on the grid for 1e-10 and 1e-2
def closest_lam(target):
    idx = np.argmin(np.abs(lams - target))
    return lams[idx]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("P2 D4: Ridge fits at three lam values", fontsize=12)

for ax, lam_t, title in zip(axes, lam_triplet, titles):
    cl = closest_lam(lam_t)
    w  = w_ridge[cl]
    y_fit = Phi_grid @ w
    norm_w = np.linalg.norm(w)
    rprint(f"  {title}: actual lam={cl:.4e}, ||w||_2 = {norm_w:.4f}")

    ax.plot(x_grid, t_sin,  "k--", lw=1,   label="sin(2pi x)")
    ax.plot(x_grid, y_fit,  "b-",  lw=1.5, label="ridge fit")
    ax.scatter(x_tr, t_tr, color="red", zorder=5, s=20, label="train")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"{title}\n||w||={norm_w:.2f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p2_d4_fitted_curves.png"), show=args.show)
rprint("  Saved: p2_d4_fitted_curves.png")

close_results()
rprint("\n[P2] Done. All figures saved in outputs/p2/")

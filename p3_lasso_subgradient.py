"""
p3_lasso_subgradient.py
CS F320 Foundations of Data Science – Assignment 1, Problem 3

Lasso via subgradient descent in the Legendre polynomial basis.
  Basis:    Phi = np.polynomial.legendre.legvander(2*x - 1, 9)
  Objective: E(w) = 0.5||Phi w - t||^2 + lam*||w||_1
  Update:   w <- w - eta*(Phi^T(Phi w - t) + lam*sign(w))
  eta=0.05, w0=0, 20000 iterations per lam.
  General P3 approximate-zero threshold: |w_j| < 2*eta*lam (evaluated across P3).
  P3 Deliverable 5 Lasso-Ridge comparison threshold: |w_j| < 0.01 (fixed comparison).

Run:  python p3_lasso_subgradient.py [--show]
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
    load_xy, rmse,
    ensure_outdir, save_fig, init_results, rprint, close_results,
    POLY_TRAIN, POLY_TEST
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "p3")

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

ensure_outdir(OUT_DIR)
init_results(OUT_DIR)

rprint("=" * 60)
rprint("P3: Lasso via Subgradient Descent (Legendre basis)")
rprint("=" * 60)

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
x_tr, t_tr = load_xy(POLY_TRAIN)
x_te, t_te = load_xy(POLY_TEST)
N_tr = len(x_tr)
rprint(f"Train N={N_tr}, Test N={len(x_te)}")

DEGREE = 9
ETA    = 0.05
N_ITER = 20000

def legendre_basis(x, degree):
    """Legendre Vandermonde: Phi[n,j] = P_j(2*x_n - 1)."""
    return np.polynomial.legendre.legvander(2 * x - 1, degree)

Phi_tr = legendre_basis(x_tr, DEGREE)   # N_tr x 10
Phi_te = legendre_basis(x_te, DEGREE)   # N_te x 10

PhiTPhi_tr = Phi_tr.T @ Phi_tr
cond_PtP = np.linalg.cond(PhiTPhi_tr)
rprint(f"cond(Phi^T Phi) = {cond_PtP:.4e}  (expect ~6e2)")

lams = np.logspace(-4, 0, 21)

def zero_thresh(lam):
    # General P3 approximate-zero threshold: |w_j| < 2*eta*lam (equals 0.01 at lam=0.1 since eta=0.05)
    return 2 * ETA * lam

def lasso_obj(w, Phi, t, lam):
    r = Phi @ w - t
    return 0.5 * np.dot(r, r) + lam * np.sum(np.abs(w))

# --------------------------------------------------------------------------- #
# Run subgradient descent for every lam, store final weights and best E
# --------------------------------------------------------------------------- #
all_w_final = {}    # lam -> final w
all_best_E  = {}    # lam -> best E seen

for lam in lams:
    w = np.zeros(DEGREE + 1)
    best_E = lasso_obj(w, Phi_tr, t_tr, lam)
    PhiTt  = Phi_tr.T @ t_tr   # precompute

    for _ in range(N_ITER):
        r    = Phi_tr @ w - t_tr
        grad = Phi_tr.T @ r + lam * np.sign(w)
        w    = w - ETA * grad
        E    = lasso_obj(w, Phi_tr, t_tr, lam)
        if E < best_E:
            best_E = E

    all_w_final[lam] = w.copy()
    all_best_E[lam]  = best_E

# --------------------------------------------------------------------------- #
# Deliverable 1 – Detailed analysis at lam=0.1
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 1: Detailed analysis at lam=0.1 ---")

LAM1 = 0.1
# Run again to record history
w = np.zeros(DEGREE + 1)
E_history  = []
best_history = []
best_E_cur = lasso_obj(w, Phi_tr, t_tr, LAM1)
PhiTt = Phi_tr.T @ t_tr

for it in range(N_ITER):
    E_cur = lasso_obj(w, Phi_tr, t_tr, LAM1)
    if E_cur < best_E_cur:
        best_E_cur = E_cur
    E_history.append(E_cur)
    best_history.append(best_E_cur)
    r    = Phi_tr @ w - t_tr
    grad = Phi_tr.T @ r + LAM1 * np.sign(w)
    w    = w - ETA * grad

E_final_1  = lasso_obj(w, Phi_tr, t_tr, LAM1)
best_E_1   = best_E_cur if E_final_1 >= best_E_cur else E_final_1

# Reference from assignment
E_STAR  = 0.2625
W_STAR  = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0], dtype=float)

rprint(f"  Final E  = {E_final_1:.6f}")
rprint(f"  Best E   = {best_E_1:.6f}")
rprint(f"  E - E*   = {E_final_1 - E_STAR:.6f}   (E* = {E_STAR})")
rprint(f"  ||w - w*||_2 = {np.linalg.norm(w - W_STAR):.6f}")
rprint("  Note: constant-step subgradient descent produces iterates that chatter")
rprint(f"  around the optimum with amplitude ~eta*lam = {ETA*LAM1:.4f}; this explains")
rprint(f"  the non-zero gap |E_final - E*| = {abs(E_final_1-E_STAR):.4f} and ||w-w*||.")
rprint("  w* is the 3-decimal reference from the assignment spec.")

# (a) convergence plot
fig, ax = plt.subplots(figsize=(8, 5))
iters = np.arange(1, N_ITER + 1)
ax.semilogy(iters, np.maximum(E_history,    1e-32), "b-",  lw=1,   label="E (current)")
ax.semilogy(iters, np.maximum(best_history, 1e-32), "r--", lw=1.5, label="best-so-far E")
ax.set_xlabel("Iteration")
ax.set_ylabel("E(w)  [log scale]")
ax.set_title(f"P3 D1a: Lasso subgradient convergence  lam={LAM1}")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p3_d1a_convergence.png"), show=args.show)
rprint("  Saved: p3_d1a_convergence.png")

# (b) optimality conditions
r_final = t_tr - Phi_tr @ w
thr1 = zero_thresh(LAM1)
nonzero_mask = np.abs(w) >= thr1
zero_mask    = ~nonzero_mask

# For non-zero coefficients: |Phi_j^T r - lam*sign(w_j)|  (should be ~0)
PhiT_r = Phi_tr.T @ r_final    # (d+1,)
if np.any(nonzero_mask):
    opt_nz = np.max(np.abs(PhiT_r[nonzero_mask] - LAM1 * np.sign(w[nonzero_mask])))
    rprint(f"  Optimality (non-zero): max |Phi_j^T r - lam*sign(w_j)| = {opt_nz:.6e}  (should be ~0)")
else:
    rprint("  No non-zero coefficients (threshold may be too tight).")

if np.any(zero_mask):
    opt_z = np.max(np.abs(PhiT_r[zero_mask]) / LAM1)
    rprint(f"  Optimality (zero):     max |Phi_j^T r|/lam            = {opt_z:.6f}  (should be <=1)")
else:
    rprint("  All coefficients are non-zero (no zero check needed).")

rprint(f"  Non-zero coefficients (|w_j| >= {thr1:.4f}): indices {np.where(nonzero_mask)[0].tolist()}")
rprint(f"  Zero     coefficients (|w_j| <  {thr1:.4f}): indices {np.where(zero_mask)[0].tolist()}")

# --------------------------------------------------------------------------- #
# Deliverable 2 – RMSE sweep
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 2: RMSE sweep vs lam ---")

train_rmses = []
test_rmses  = []
for lam in lams:
    w_f = all_w_final[lam]
    train_rmses.append(rmse(t_tr, Phi_tr @ w_f))
    test_rmses.append(rmse(t_te, Phi_te @ w_f))

train_rmses = np.array(train_rmses)
test_rmses  = np.array(test_rmses)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(lams, train_rmses, "b-o", ms=4, lw=1, label="Train RMSE")
ax.semilogx(lams, test_rmses,  "r-o", ms=4, lw=1, label="Test RMSE")
ax.axhline(0.1, color="k", ls="--", lw=1, label="noise level 0.1")
ax.set_xlabel("lam (log scale)")
ax.set_ylabel("RMSE")
ax.set_title("P3 D2: Lasso RMSE sweep vs lam")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p3_d2_rmse_sweep.png"), show=args.show)
rprint("  Saved: p3_d2_rmse_sweep.png")

# --------------------------------------------------------------------------- #
# Deliverable 3 – Coefficient path
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 3: Coefficient path ---")

W_path = np.array([all_w_final[lam] for lam in lams])   # (21, 10)

# D3, D4 and D1 all use the same rule: a coefficient is ~0 iff |w_j| < 2*eta*lam
# (zero_thresh function). This is the subgradient step size times the regulariser.
rprint(f"\n  {'lam':<12}  {'# ~zero coefs (|w_j| < 2*eta*lam)'}")
rprint(f"  {'-'*50}")
for i, lam in enumerate(lams):
    thr = zero_thresh(lam)
    n_zero = int(np.sum(np.abs(W_path[i]) < thr))
    rprint(f"  {lam:<12.4e}  {n_zero}")

fig, ax = plt.subplots(figsize=(9, 5))
for j in range(DEGREE + 1):
    ax.semilogx(lams, W_path[:, j], "-o", ms=3, lw=1, label=f"w_{j}")
ax.set_xlabel("lam (log scale)")
ax.set_ylabel("Coefficient value")
ax.set_title("P3 D3: Lasso coefficient path vs lam")
ax.axhline(0, color="k", lw=0.5)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p3_d3_coef_path.png"), show=args.show)
rprint("  Saved: p3_d3_coef_path.png")

# --------------------------------------------------------------------------- #
# Deliverable 4 – lam* = argmin test RMSE
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 4: Best lam (lam*) ---")

best_idx_lasso = int(np.argmin(test_rmses))
lam_star_lasso = lams[best_idx_lasso]
best_test_lasso = test_rmses[best_idx_lasso]
w_star_lasso   = all_w_final[lam_star_lasso]

thr_star = zero_thresh(lam_star_lasso)
zero_indices_lasso = np.where(np.abs(w_star_lasso) < thr_star)[0].tolist()
n_zero_lasso       = len(zero_indices_lasso)

rprint(f"  lam* = {lam_star_lasso:.4e}")
rprint(f"  Test RMSE at lam* = {best_test_lasso:.6f}")
rprint(f"  Number of ~zero coefficients: {n_zero_lasso}")
rprint(f"  Indices of ~zero coefficients: {zero_indices_lasso}")

# --------------------------------------------------------------------------- #
# Deliverable 5 – Lasso vs Ridge comparison in Legendre basis
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 5: Lasso vs Ridge comparison (Legendre basis) ---")

# Ridge in Legendre basis via np.linalg.solve
PhiTPhi_leg = Phi_tr.T @ Phi_tr
PhiTt_leg   = Phi_tr.T @ t_tr

def ridge_legendre(lam):
    A = PhiTPhi_leg + lam * np.eye(DEGREE + 1)
    return np.linalg.solve(A, PhiTt_leg)

ridge_test_rmses = np.array([rmse(t_te, Phi_te @ ridge_legendre(lam)) for lam in lams])
best_ridge_idx   = int(np.argmin(ridge_test_rmses))
lam_star_ridge   = lams[best_ridge_idx]
w_star_ridge     = ridge_legendre(lam_star_ridge)
best_test_ridge  = ridge_test_rmses[best_ridge_idx]

# P3 Deliverable 5 Lasso–Ridge comparison threshold: fixed at |w_j| < 0.01
SMALL = 0.01
n_small_lasso = int(np.sum(np.abs(w_star_lasso) < SMALL))
n_small_ridge = int(np.sum(np.abs(w_star_ridge) < SMALL))

rprint(f"\n  {'Method':<14} {'lam*':<12} {'Test RMSE':<14} {'||w||_2':<12} {'# coeffs with |w_j| < 0.01'}")
rprint(f"  {'-'*72}")
rprint(f"  {'Lasso':<14} {lam_star_lasso:<12.4e} {best_test_lasso:<14.6f} "
       f"{np.linalg.norm(w_star_lasso):<12.4f} {n_small_lasso}")
rprint(f"  {'Ridge':<14} {lam_star_ridge:<12.4e} {best_test_ridge:<14.6f} "
       f"{np.linalg.norm(w_star_ridge):<12.4f} {n_small_ridge}")

# --------------------------------------------------------------------------- #
# Deliverable 6 – Fitted curves: Lasso at lam*, Ridge at its best lam
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 6: Fitted curves (Lasso vs Ridge at best lam) ---")

x_grid  = np.linspace(0, 1, 300)
t_sin   = np.sin(2 * np.pi * x_grid)
Phi_grid = legendre_basis(x_grid, DEGREE)

y_lasso = Phi_grid @ w_star_lasso
y_ridge = Phi_grid @ w_star_ridge

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_grid, t_sin,   "k--", lw=1,   label="sin(2pi x)")
ax.plot(x_grid, y_lasso, "b-",  lw=1.8, label=f"Lasso lam={lam_star_lasso:.2e}")
ax.plot(x_grid, y_ridge, "g-.", lw=1.8, label=f"Ridge lam={lam_star_ridge:.2e}")
ax.scatter(x_tr, t_tr, color="red", zorder=5, s=25, label="train data")
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("P3 D6: Lasso vs Ridge fitted curves (best lam each)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p3_d6_fitted_curves.png"), show=args.show)
rprint("  Saved: p3_d6_fitted_curves.png")

# --------------------------------------------------------------------------- #
# Deliverable 7 – Discussion
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 7: Discussion ---")

# Compute whether train and test RMSE are monotone for lam >= lam*.
# Use the actual arrays to determine phrasing.
star_i = best_idx_lasso
train_tail = train_rmses[star_i:]   # lam >= lam*
test_tail  = test_rmses[star_i:]
train_mono = bool(np.all(np.diff(train_tail) >= 0))
test_mono  = bool(np.all(np.diff(test_tail)  >= 0))
train_adv  = "monotonically increase" if train_mono else "generally increase"
test_adv   = "monotonically increase" if test_mono  else "generally increase"
rprint(f"  Monotonicity check for lam >= lam* = {lam_star_lasso:.4e}:")
rprint(f"    train RMSE strictly non-decreasing: {train_mono} -> phrase: '{train_adv}'")
rprint(f"    test  RMSE strictly non-decreasing: {test_mono}  -> phrase: '{test_adv}'")

# Compute citations from coefficient path (D3) and comparison table (D5)
idx_01 = int(np.argmin(np.abs(lams - 0.1)))
n_zero_01 = int(np.sum(np.abs(W_path[idx_01]) < zero_thresh(lams[idx_01])))
n_zero_max = int(np.sum(np.abs(W_path[-1]) < zero_thresh(lams[-1])))
lam_max = lams[-1]

disc = (
    "Discussion (P3): L1 regularisation (Lasso) produces exact zeros because the "
    "subdifferential optimality condition for w_j=0 requires |Phi_j^T r| <= lam, where the "
    "diamond-shaped L1 ball forces solutions to touch coordinate axes, whereas the smooth L2 "
    "ball in Ridge only shrinks weights without setting them to zero. The coefficient path (D3) "
    f"demonstrates this progressive pruning: the number of ~zero coefficients (|w_j| < 2*eta*lam) "
    f"grows from 0 at small lam to {n_zero_01} at lam=0.1, reaching {n_zero_max} at the largest "
    f"lam={lam_max:.1e}. In the D5 comparison on the Legendre basis, Lasso achieves a test RMSE of "
    f"{best_test_lasso:.6f} with {n_small_lasso} coefficients having |w_j| < 0.01, whereas Ridge "
    f"yields a test RMSE of {best_test_ridge:.6f} with only {n_small_ridge} coefficient below 0.01. "
    f"Beyond lam*={lam_star_lasso:.2e}, excessive regularisation removes true signal, so train RMSE and "
    f"test RMSE both {train_adv} as the model underfits."
)
rprint("\n  " + disc)

close_results()
rprint("\n[P3] Done. All figures saved in outputs/p3/")

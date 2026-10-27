"""
p4_bayesian_regression.py
CS F320 Foundations of Data Science – Assignment 1, Problem 4

Bayesian linear regression with fixed hyperparameters.
  Degree-3 polynomial basis, sigma^2 = 0.01 (beta=100), alpha=0.05.
  S_N^{-1} = alpha I + beta Phi^T Phi
  m_N = beta S_N Phi^T t  (computed via np.linalg.solve, no explicit inverse for m_N)
  Predictive: m(x) = phi^T m_N
              s^2(x) = 1/beta + phi^T S_N phi

Run:  python p4_bayesian_regression.py [--show]
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
    load_xy, vandermonde,
    ensure_outdir, save_fig, init_results, rprint, close_results,
    P4_FILES
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "p4")

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

ensure_outdir(OUT_DIR)
init_results(OUT_DIR)

rprint("=" * 60)
rprint("P4: Bayesian Linear Regression")
rprint("=" * 60)

DEGREE = 3
ALPHA  = 0.05
BETA   = 100.0          # beta = 1/sigma^2 = 1/0.01

rprint(f"Hyperparameters: alpha={ALPHA}, beta={BETA}, degree={DEGREE}")

x_grid = np.linspace(0, 1, 200)
t_sin  = np.sin(2 * np.pi * x_grid)   # for plotting only
Phi_grid = vandermonde(x_grid, DEGREE)  # 200 x 4

# ------------------------------------------------------------------ #
# Bayesian update: returns (m_N, S_N, S_N_inv)
# ------------------------------------------------------------------ #
def bayesian_update(x_tr, t_tr):
    Phi = vandermonde(x_tr, DEGREE)      # N x (d+1)
    D   = DEGREE + 1                     # 4
    S_N_inv = ALPHA * np.eye(D) + BETA * Phi.T @ Phi   # (d+1) x (d+1)
    # m_N = beta * S_N * Phi^T t  ->  solve S_N_inv m_N = beta * Phi^T t
    rhs = BETA * Phi.T @ t_tr
    m_N = np.linalg.solve(S_N_inv, rhs)
    # S_N for predictive variance: S_N = inv(S_N_inv) -- fine to use here
    S_N = np.linalg.inv(S_N_inv)
    return m_N, S_N, S_N_inv

def predictive(Phi_x, m_N, S_N):
    """
    Phi_x: (n_pts, d+1) grid design matrix
    Returns m_pred (n_pts,) and s_pred (n_pts,)
    """
    m_pred = Phi_x @ m_N                               # (n_pts,)
    # s^2(x) = 1/beta + phi^T S_N phi
    SN_phi = S_N @ Phi_x.T                             # (d+1, n_pts)
    var    = 1.0 / BETA + np.einsum("ij,ij->j", Phi_x.T, SN_phi)   # (n_pts,)
    s_pred = np.sqrt(np.maximum(var, 0.0))             # (n_pts,)
    return m_pred, s_pred

# --------------------------------------------------------------------------- #
# Deliverable 1 – Posterior predictive for N=10
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 1: Posterior predictive for N=10 ---")

x_tr10, t_tr10 = load_xy(P4_FILES[10])
m_N10, S_N10, SN_inv10 = bayesian_update(x_tr10, t_tr10)

m_pred10, s_pred10 = predictive(Phi_grid, m_N10, S_N10)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_grid, t_sin,     "k--", lw=1,   label="sin(2pi x)")
ax.plot(x_grid, m_pred10,  "b-",  lw=2,   label="Posterior mean m(x)")
ax.fill_between(x_grid,
                m_pred10 - 1.96 * s_pred10,
                m_pred10 + 1.96 * s_pred10,
                alpha=0.25, color="blue", label="95% predictive band")
ax.scatter(x_tr10, t_tr10, color="red", zorder=5, s=30, label="Training data (N=10)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("P4 D1: Bayesian predictive (N=10, degree=3)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

# --------------------------------------------------------------------------- #
# Deliverable 2 – Equivalence: m_N == w_MAP
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 2: Equivalence m_N == w_MAP ---")

LAM_MAP = ALPHA / BETA       # lam = alpha/beta
PhiTPhi10 = vandermonde(x_tr10, DEGREE).T @ vandermonde(x_tr10, DEGREE)
PhiTt10   = vandermonde(x_tr10, DEGREE).T @ t_tr10
A_map = PhiTPhi10 + LAM_MAP * np.eye(DEGREE + 1)
w_MAP = np.linalg.solve(A_map, PhiTt10)

equiv_err = np.linalg.norm(w_MAP - m_N10)
rprint(f"  lam = alpha/beta = {LAM_MAP:.6e}")
rprint(f"  ||w_MAP - m_N||_2 = {equiv_err:.4e}  (expect ~1e-12, machine precision)")

# Overlay w_MAP prediction
y_map = Phi_grid @ w_MAP
ax.plot(x_grid, y_map, "m--", lw=1.5, label=f"w_MAP (lam={LAM_MAP:.3e})")
ax.legend(fontsize=8)

save_fig(fig, os.path.join(OUT_DIR, "p4_d1d2_predictive_N10.png"), show=args.show)
rprint("  Saved: p4_d1d2_predictive_N10.png  (includes w_MAP overlay)")

# --------------------------------------------------------------------------- #
# Deliverable 3 – Effect of N: interval width vs x for N in {5,10,20,50}
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 3: Interval width vs N ---")

IRREDUCIBLE = 2 * 1.96 / np.sqrt(BETA)   # 2*1.96/sqrt(100) = 0.392
rprint(f"  Irreducible floor 2*1.96/sqrt(beta) = {IRREDUCIBLE:.4f}")

N_vals = [5, 10, 20, 50]
fig2, ax2 = plt.subplots(figsize=(9, 5))

for N_val in N_vals:
    x_tr_n, t_tr_n = load_xy(P4_FILES[N_val])
    m_Nn, S_Nn, _ = bayesian_update(x_tr_n, t_tr_n)
    m_n, s_n = predictive(Phi_grid, m_Nn, S_Nn)
    width = 2 * 1.96 * s_n   # interval width
    mean_w = np.mean(width)
    max_w  = np.max(width)
    rprint(f"  N={N_val:<3}: mean width={mean_w:.4f}, max width={max_w:.4f}")
    ax2.plot(x_grid, width, lw=1.5, label=f"N={N_val}")

ax2.axhline(IRREDUCIBLE, color="k", ls="--", lw=1.5, label=f"Irreducible {IRREDUCIBLE:.3f}")
ax2.set_xlabel("x")
ax2.set_ylabel("Interval width 2*1.96*s(x)")
ax2.set_title("P4 D3: Predictive interval width vs x for different N")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig2, os.path.join(OUT_DIR, "p4_d3_interval_width.png"), show=args.show)
rprint("  Saved: p4_d3_interval_width.png")

# --------------------------------------------------------------------------- #
# Deliverable 4 – Discussion
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 4: Discussion ---")

disc = (
    "Discussion (P4): As N increases, the posterior covariance S_N = "
    "(alpha I + beta Phi^T Phi)^{-1} shrinks because Phi^T Phi grows roughly "
    "proportional to N, making the precision matrix S_N^{-1} increasingly dominated "
    "by the data term beta*Phi^T Phi. Consequently phi^T S_N phi -- the parameter "
    "uncertainty contribution to the predictive variance -- falls roughly like 1/N, "
    "and the interval width 2*1.96*s(x) approaches the irreducible floor from above. "
    "The irreducible floor 2*1.96/sqrt(beta) = " + f"{IRREDUCIBLE:.3f}" + " arises from "
    "the noise term 1/beta in s^2(x) = 1/beta + phi^T S_N phi: even with infinite data "
    "we cannot predict individual noisy observations more precisely than the noise "
    "level sigma = 1/sqrt(beta), so this term never disappears."
)
rprint("\n  " + disc)

close_results()
rprint("\n[P4] Done. All figures saved in outputs/p4/")

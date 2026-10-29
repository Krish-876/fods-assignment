"""
p1_gd_family.py
CS F320 Foundations of Data Science – Assignment 1, Problem 1

Polynomial regression: closed-form vs GD family (Batch GD, SGD, Mini-batch GD)
on poly_train.csv using the QR reparametrisation for degree-9.

Run:  python p1_gd_family.py [--show]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from any CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    load_xy, vandermonde, sse, rmse,
    ensure_outdir, save_fig, init_results, rprint, close_results,
    POLY_TRAIN
)

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "p1")

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Display figures interactively")
args = parser.parse_args()

ensure_outdir(OUT_DIR)
init_results(OUT_DIR)

rprint("=" * 60)
rprint("P1: Polynomial Regression  |  Closed Form vs GD Family")
rprint("=" * 60)
rprint(f"Random seed: {SEED}")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
x_tr, t_tr = load_xy(POLY_TRAIN)
N = len(x_tr)          # 10
rprint(f"Training points N={N}")

# Dense grid for plotting
x_grid = np.linspace(0, 1, 300)
t_sin  = np.sin(2 * np.pi * x_grid)   # used ONLY for plotting

# --------------------------------------------------------------------------- #
# Deliverable 1 – Closed-form for d=1..9, condition numbers, 3x3 subplot grid
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 1: Closed-form fits, condition numbers ---")

degrees = list(range(1, 10))
w_closed = {}          # store for later use

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
fig.suptitle("P1 D1: Polynomial fits (closed form) d=1..9", fontsize=13)

for idx, d in enumerate(degrees):
    Phi = vandermonde(x_tr, d)
    cond_val = np.linalg.cond(Phi)
    rprint(f"  d={d}: cond(Phi) = {cond_val:.4e}")

    # Closed-form via lstsq (no explicit inverse)
    w, _, _, _ = np.linalg.lstsq(Phi, t_tr, rcond=None)
    w_closed[d] = w

    # Plot
    ax = axes[idx // 3][idx % 3]
    Phi_grid = vandermonde(x_grid, d)
    y_fit = Phi_grid @ w
    ax.plot(x_grid, t_sin, "k--", lw=1, label="sin(2pi x)")
    ax.plot(x_grid, y_fit, "b-",  lw=1.5, label=f"d={d} fit")
    ax.scatter(x_tr, t_tr, color="red", zorder=5, s=20, label="train")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"d={d}  cond={cond_val:.1e}", fontsize=9)
    ax.tick_params(labelsize=7)
    if idx == 0:
        ax.legend(fontsize=6)

plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p1_d1_closed_form_fits.png"), show=args.show)
rprint("  Saved: p1_d1_closed_form_fits.png")

# --------------------------------------------------------------------------- #
# Deliverable 2 – QR reparametrisation for d=9, implement GD variants
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 2: QR reparametrisation, GD family (d=9) ---")

d = 9
Phi9 = vandermonde(x_tr, d)          # 10 x 10
Q, R = np.linalg.qr(Phi9, mode="reduced")   # thin QR: Q is 10x10 (square here)

rprint(f"  Q shape: {Q.shape}, R shape: {R.shape}")

# Verify Q is orthogonal
QTQ = Q.T @ Q
row_norms = np.linalg.norm(Q, axis=1)   # row norms of Q
max_off_diag = np.max(np.abs(QTQ - np.eye(Q.shape[1])))
rprint(f"  Row norms of Q (should be ~1): {np.round(row_norms, 6)}")
rprint(f"  Max off-diagonal of Q Q^T:     {max_off_diag:.4e}  (confirms orthonormality)")

# E(alpha) = 0.5 * ||Q alpha - t||^2
# gradient: Q^T (Q alpha - t)

def E_alpha(alpha):
    r = Q @ alpha - t_tr
    return 0.5 * np.dot(r, r)

def grad_E_alpha(alpha):
    return Q.T @ (Q @ alpha - t_tr)

def recover_w(alpha):
    """Solve R w = alpha for w without explicit inverse."""
    return np.linalg.solve(R, alpha)

# ---- Batch GD ---------------------------------------------------------------
def run_batch_gd(eta, n_epochs):
    alpha = np.zeros(d + 1)
    history = [E_alpha(alpha)]
    for _ in range(n_epochs):
        alpha -= eta * grad_E_alpha(alpha)
        history.append(E_alpha(alpha))
    return alpha, np.array(history)

# ---- SGD --------------------------------------------------------------------
def run_sgd(eta, n_epochs, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(d + 1)
    history = [E_alpha(alpha)]
    for _ in range(n_epochs):
        idx = rng.permutation(N)   # shuffle each epoch
        for n in idx:
            q_n = Q[n]             # shape (d+1,)
            err = np.dot(q_n, alpha) - t_tr[n]
            alpha -= eta * err * q_n
        history.append(E_alpha(alpha))
    return alpha, np.array(history)

# ---- Mini-batch GD ----------------------------------------------------------
MINI_BATCH_SIZE = 3

def run_minibatch_gd(eta, n_epochs, batch_size=MINI_BATCH_SIZE, seed=SEED):
    rng = np.random.default_rng(seed)
    alpha = np.zeros(d + 1)
    history = [E_alpha(alpha)]
    for _ in range(n_epochs):
        idx = rng.permutation(N)
        for start in range(0, N, batch_size):
            batch_idx = idx[start:start + batch_size]
            Q_b = Q[batch_idx]
            t_b = t_tr[batch_idx]
            # AVERAGED gradient over mini-batch
            grad = Q_b.T @ (Q_b @ alpha - t_b) / len(batch_idx)
            alpha -= eta * grad
        history.append(E_alpha(alpha))
    return alpha, np.array(history)

# --------------------------------------------------------------------------- #
# Deliverable 3 – Loss curves (eta=0.1, adaptive epochs until all flat)
# "Flat" = final E < 1e-28 OR last 20% of epochs unchanged to 1e-32 floor.
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 3: Loss curves (eta=0.1) ---")

ETA_MAIN  = 0.1
FLOOR     = 1e-32   # prevent log(0)

def is_flat(hist, tol=1e-28):
    """True if final E < tol OR last 20% of values are all within 1e-32 of each other."""
    if hist[-1] < tol:
        return True
    tail = hist[int(0.8 * len(hist)):]
    return (np.max(tail) - np.min(tail)) < 1e-32

# Start at 600 and double until every variant is flat (cap at 4000).
N_EPOCHS_MAIN = 600
while True:
    alpha_bgd,  hist_bgd  = run_batch_gd(ETA_MAIN, N_EPOCHS_MAIN)
    alpha_sgd,  hist_sgd  = run_sgd(ETA_MAIN, N_EPOCHS_MAIN)
    alpha_mbgd, hist_mbgd = run_minibatch_gd(ETA_MAIN, N_EPOCHS_MAIN)
    if (is_flat(hist_bgd) and is_flat(hist_sgd) and is_flat(hist_mbgd)) \
            or N_EPOCHS_MAIN >= 4000:
        break
    N_EPOCHS_MAIN *= 2

hist_bgd_plot  = np.maximum(hist_bgd,  FLOOR)
hist_sgd_plot  = np.maximum(hist_sgd,  FLOOR)
hist_mbgd_plot = np.maximum(hist_mbgd, FLOOR)

fig, ax = plt.subplots(figsize=(8, 5))
epochs_axis = np.arange(N_EPOCHS_MAIN + 1)
ax.semilogy(epochs_axis, hist_bgd_plot,  "b-",  lw=1.5, label="Batch GD")
ax.semilogy(epochs_axis, hist_sgd_plot,  "g--", lw=1.5, label="SGD")
ax.semilogy(epochs_axis, hist_mbgd_plot, "r:",  lw=2,   label="Mini-batch GD (bs=3)")
ax.set_xlabel("Epoch")
ax.set_ylabel("E(alpha)  [log scale]")
ax.set_title(f"P1 D3: Loss curves – Batch GD vs SGD vs Mini-batch GD  (eta={ETA_MAIN}, {N_EPOCHS_MAIN} epochs)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p1_d3_loss_curves.png"), show=args.show)
rprint("  Saved: p1_d3_loss_curves.png")
rprint(f"  Epoch count used: {N_EPOCHS_MAIN}")
rprint(f"  Final E  Batch GD: {hist_bgd[-1]:.6e}")
rprint(f"  Final E  SGD:      {hist_sgd[-1]:.6e}")
rprint(f"  Final E  Mini-GD:  {hist_mbgd[-1]:.6e}")

# --------------------------------------------------------------------------- #
# Deliverable 4 – Learning-rate study (Batch GD, 100 epochs)
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 4: Learning-rate study ---")

LR_EPOCHS = 100
etas_study = [0.01, 0.1, 2.1]
eta_labels = {0.01: "eta=0.01 (too small)", 0.1: "eta=0.1 (well-chosen)", 2.1: "eta=2.1 (too large)"}

fig, ax = plt.subplots(figsize=(9, 5))

w_cf9 = w_closed[9]   # reference closed-form weight

for eta_val in etas_study:
    alpha_lr, hist_lr = run_batch_gd(eta_val, LR_EPOCHS)
    w_lr = recover_w(alpha_lr)
    rprint(f"\n  eta={eta_val}")
    with np.printoptions(precision=4, suppress=False, formatter={"float_kind": lambda v: f"{v:.4e}"}):
        rprint(f"    w = {w_lr}")
    rprint(f"    Final E (epoch {LR_EPOCHS}): {hist_lr[-1]:.6e}")
    hist_plot = np.maximum(hist_lr, FLOOR)
    ax.semilogy(np.arange(LR_EPOCHS + 1), hist_plot, lw=1.5, label=eta_labels[eta_val])

ax.set_xlabel("Epoch")
ax.set_ylabel("E(alpha)  [log scale]")
ax.set_title("P1 D4: Learning-rate study – Batch GD, d=9")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, "p1_d4_lr_study.png"), show=args.show)
rprint("\n  Saved: p1_d4_lr_study.png")

# eta=2 special case: |1 - eta*lambda_max|; for QR-reparametrised problem
# Q is orthogonal => eigenvalues of Q^T Q are all 1 => curvature is 1 for every direction
# So eta=2 => contraction factor = |1 - eta*1| = |1-2| = 1  -> no convergence
rprint("\n  --- eta=2 special case ---")
alpha_eta2, hist_eta2 = run_batch_gd(2.0, LR_EPOCHS)
w_eta2 = recover_w(alpha_eta2)
rprint(f"  eta=2:  Final E = {hist_eta2[-1]:.6e}")
rprint(f"  E values at epochs 95-100: {hist_eta2[95:101]}")
rprint("  For the QR-reparametrised problem all curvatures == 1.")
rprint("  eta=2 => contraction |1 - eta*1| = 1 => loss oscillates / does not decrease.")

# --------------------------------------------------------------------------- #
# Deliverable 5 – Verification: 2000 epochs, compare all three to closed form
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 5: Verification at 2000 epochs (eta=0.1) ---")

VER_EPOCHS = 2000

alpha_v_bgd,  hist_v_bgd  = run_batch_gd(ETA_MAIN, VER_EPOCHS)
alpha_v_sgd,  hist_v_sgd  = run_sgd(ETA_MAIN, VER_EPOCHS)
alpha_v_mbgd, hist_v_mbgd = run_minibatch_gd(ETA_MAIN, VER_EPOCHS)

w_v_bgd  = recover_w(alpha_v_bgd)
w_v_sgd  = recover_w(alpha_v_sgd)
w_v_mbgd = recover_w(alpha_v_mbgd)

w9_cf   = w_closed[9]
norm_cf = np.linalg.norm(w9_cf)

def verif_row(name, w_gd, w_cf):
    diff = np.linalg.norm(w_gd - w_cf)
    rel  = diff / np.linalg.norm(w_cf)
    return name, diff, rel

# Compute condition number of R to explain the error floor.
# alpha converges to essentially machine precision in the Q-space,
# but recovering w = R^{-1} alpha amplifies round-off by cond(R).
cond_R = np.linalg.cond(R)
rprint(f"\n  ||w_closed||_2 = {norm_cf:.6e}")
rprint(f"  cond(R)        = {cond_R:.4e}  (amplifies alpha round-off when recovering w)")

rows = []
for name, w_gd in [("Batch GD", w_v_bgd), ("SGD", w_v_sgd), ("Mini-batch GD", w_v_mbgd)]:
    nm, diff, rel = verif_row(name, w_gd, w9_cf)
    rows.append((nm, diff, rel))

rprint(f"\n  {'Method':<20} {'||w_GD - w_cf||_2':<22} {'Relative error':<20}")
rprint(f"  {'-'*60}")
for nm, diff, rel in rows:
    rprint(f"  {nm:<20} {diff:<22.6e} {rel:<20.6e}")

# Summarise the error floor.
rprint(f"\n  All three variants agree with the closed form to")
rprint(f"  order 1e-11 relative (a few 1e-6 absolute). The error is the same for")
rprint(f"  all three variants and is machine-dependent round-off amplified by")
rprint(f"  cond(R) ~ 1.5e7 (measured cond(R) = {cond_R:.4e}) when recovering w = R^{{-1}} alpha,")
rprint(f"  not by the GD variant.")

# --------------------------------------------------------------------------- #
# Deliverable 6 – Discussion
# --------------------------------------------------------------------------- #
rprint("\n--- Deliverable 6: Discussion ---")

# Verify orthonormality claims numerically
rprint(f"  Row norms of Q:      {np.round(row_norms, 8)}")
rprint(f"  Max off-diag Q Q^T:  {max_off_diag:.4e}")

disc = (
    "Discussion (P1): With the QR reparametrisation Q is orthogonal (10x10), so E(alpha) "
    "has curvature 1 in every direction and each full pass over the data contracts the error "
    "by |1 - eta| = 0.9 (eta = 0.1); the loss curves are straight lines on the log scale for "
    "that reason. Batch GD and SGD lie on top of each other: the rows of Q are orthonormal, "
    "so each of SGD's 10 single-sample updates per epoch contracts one alpha-direction by (1 - eta) "
    "independent of the others, updating every direction once per epoch exactly as batch GD does. "
    "Mini-batch GD (batch size 3, 4 averaged updates per epoch) descends visibly more slowly because "
    "mini-batch averaging shrinks the effective per-direction step. All three variants agree with "
    "the closed form to order 1e-11 relative (a few 1e-6 absolute); the error is the same for all "
    "three variants and reflects machine-dependent round-off amplified by cond(R) ~ 1.5e7."
)
rprint("\n  " + disc)

close_results()
rprint("\n[P1] Done. All figures saved in outputs/p1/")

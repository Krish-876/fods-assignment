"""
common.py  --  shared helpers for CS F320 Assignment-1
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive by default
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "A1-datasets")

POLY_TRAIN = os.path.join(DATA_DIR, "poly_train.csv")
POLY_TEST  = os.path.join(DATA_DIR, "poly_test.csv")
P4_FILES   = {
    5:  os.path.join(DATA_DIR, "p4_train_N5.csv"),
    10: os.path.join(DATA_DIR, "p4_train_N10.csv"),
    20: os.path.join(DATA_DIR, "p4_train_N20.csv"),
    50: os.path.join(DATA_DIR, "p4_train_N50.csv"),
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
_result_fh = None   # file handle set by init_results()


def init_results(out_dir):
    """Open outputs/pN/results.txt for writing (UTF-8). Must be called once."""
    global _result_fh
    ensure_outdir(out_dir)
    _result_fh = open(os.path.join(out_dir, "results.txt"), "w", encoding="utf-8")


def rprint(*args, **kwargs):
    """Print to console AND to results.txt (tee)."""
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    if _result_fh is not None:
        end = kwargs.get("end", "\n")
        _result_fh.write(msg + end)
        _result_fh.flush()


def close_results():
    global _result_fh
    if _result_fh is not None:
        _result_fh.close()
        _result_fh = None


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def save_fig(fig, path, dpi=150, show=False):
    """Save figure to path at given dpi. Optionally display."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_xy(csv_path):
    """Return (x, y) as 1-D float64 numpy arrays."""
    df = pd.read_csv(csv_path)
    return df["x"].to_numpy(dtype=np.float64), df["y"].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------
def vandermonde(x, degree):
    """
    Return N x (degree+1) Vandermonde (monomial) design matrix:
      Phi[n, j] = x[n]^j  for j = 0, ..., degree
    """
    N = len(x)
    Phi = np.zeros((N, degree + 1), dtype=np.float64)
    for j in range(degree + 1):
        Phi[:, j] = x ** j
    return Phi


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(y_true, y_pred):
    """Root mean squared error."""
    return math.sqrt(np.mean((y_true - y_pred) ** 2))


def sse(Phi, w, t):
    """0.5 * ||Phi w - t||^2  (SSE without 1/N factor)."""
    r = Phi @ w - t
    return 0.5 * np.dot(r, r)

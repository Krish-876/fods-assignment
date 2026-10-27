"""Build a self-contained report.html (and report.pdf if a converter is available)
from outputs/pN/results.txt and the PNG figures. All numbers come from results.txt."""
import base64
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TEAM = [("B.N.R. Krishna", "2024A7PS0175H"), ("Sudhanva B H", "20240085H")]


def load(p):
    return (OUT / f"p{p}" / "results.txt").read_text(encoding="utf-8")


def grab(text, pattern, group=1):
    m = re.search(pattern, text, re.S)
    if not m:
        sys.exit(f"build_report: pattern not found: {pattern!r}")
    return m.group(group)


def section(text, n):
    """Text of 'Deliverable n' up to the next deliverable header."""
    m = re.search(rf"--- Deliverable {n}\b.*?(?=\n--- Deliverable|\Z)", text, re.S)
    if not m:
        sys.exit(f"build_report: deliverable {n} not found")
    return m.group(0)


def fig(p, name, caption):
    path = OUT / f"p{p}" / name
    if not path.exists():
        sys.exit(f"build_report: missing figure {path}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def table(header, rows):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def code(s):
    return f"<pre>{html.escape(s.strip())}</pre>"


def sub(title, *parts):
    return f'<section class="sub"><h3>{title}</h3>{"".join(parts)}</section>'


def para(s):
    return f"<p>{html.escape(s)}</p>"


# ---------------------------------------------------------------- P1
def build_p1():
    t = load(1)
    d1, d2, d3, d4, d5 = (section(t, i) for i in range(1, 6))
    conds = re.findall(r"d=(\d): cond\(Phi\) = (\S+)", d1)
    eta = grab(d3, r"Loss curves \(eta=([\d.]+)\)")
    fin = {k: grab(d3, rf"Final E\s+{k}:\s+(\S+)") for k in ("Batch GD", "SGD", "Mini-GD")}
    qshape = grab(d2, r"(Q shape: .*?R shape: \([^)]*\))")
    qoff = grab(d2, r"Max off-diagonal of Q Q\^T:\s+(\S+)")
    lr = re.findall(r"eta=([\d.]+)\n\s+w = (\[.*?\])\n\s+Final E \(epoch 100\): (\S+)", d4, re.S)
    e2 = grab(d4, r"eta=2:\s+Final E = (\S+)")
    verif = re.findall(r"^\s+(Batch GD|SGD|Mini-batch GD)\s+(\S+)\s+(\S+)\s*$", d5, re.M)
    wnorm = grab(d5, r"\|\|w_closed\|\|_2 = (\S+)")
    condR = grab(d5, r"cond\(R\)\s+= (\S+)")
    contraction = f"{1 - float(eta):.1f}"
    disc = (
        f"With the QR reparametrisation Q is orthogonal, so E(alpha) has curvature 1 in every "
        f"direction and each full pass over the data contracts the error by |1 - eta| = {contraction} "
        f"(eta = {eta}); the loss curves in the figure are straight lines on the log scale for that reason. "
        f"Batch GD and SGD lie on top of each other: the rows of Q are orthonormal, so each single-sample "
        f"step in SGD touches one alpha-direction and one epoch updates every direction once, exactly as "
        f"batch GD does. Mini-batch GD descends visibly more slowly because averaging the gradient over a "
        f"batch shrinks the per-sample step, yet all three reach a round-off floor (final E "
        f"{fin['Batch GD']}, {fin['SGD']} and {fin['Mini-GD']} respectively)."
    )
    return (
        "<h2>P1 &ndash; Closed form and the gradient-descent family</h2>"
        + sub("D1 &ndash; Closed-form fits and condition numbers",
              fig(1, "p1_d1_closed_form_fits.png", "Closed-form polynomial fits, degrees 1&ndash;9"),
              table(["Degree", "cond(Phi)"], conds))
        + sub("D2 &ndash; QR reparametrisation (d = 9)",
              code(f"{qshape}\nMax off-diagonal of Q Q^T: {qoff}"))
        + sub("D3 &ndash; Loss curves",
              fig(1, "p1_d3_loss_curves.png", f"Batch GD, SGD and mini-batch GD loss (eta = {eta})"),
              table(["Method", "Final E"], [("Batch GD", fin["Batch GD"]), ("SGD", fin["SGD"]),
                                             ("Mini-batch GD", fin["Mini-GD"])]))
        + sub("D4 &ndash; Learning-rate study",
              fig(1, "p1_d4_lr_study.png", "Learning-rate study, 100 epochs"),
              table(["eta", "Final E (epoch 100)"], [(e, f) for e, _, f in lr] + [("2 (special case)", e2)]),
              code("\n".join(f"eta={e}: w = {w}" for e, w, _ in lr)))
        + sub("D5 &ndash; Verification against closed form (2000 epochs)",
              table(["Method", "||w_GD - w_cf||_2", "Relative error"], verif),
              code(f"||w_closed||_2 = {wnorm}\ncond(R) = {condR}"))
        + sub("D6 &ndash; Discussion", para(disc))
    )


# ---------------------------------------------------------------- P2
def build_p2():
    t = load(2)
    d2, d3, d4 = section(t, 2), section(t, 3), section(t, 4)
    c0 = grab(d2, r"cond\(Phi\^T Phi\)\s+= (\S+)")
    c1 = grab(d2, r"cond\(Phi\^T Phi \+ lam I\)\s+= (\S+)\s+\(lam=([\d.e-]+)\)", 1)
    lam_reg = grab(d2, r"cond\(Phi\^T Phi \+ lam I\)\s+= \S+\s+\(lam=([\d.e-]+)\)")
    it = grab(d2, r"GD converged at iteration (\d+)")
    rel = grab(d2, r"\|\|w_GD - w_closed\|\|_2 / \|\|w_closed\|\|_2 = (\S+)")
    lam_star = grab(d3, r"lam\* = (\S+)\s+->")
    rmse = grab(d3, r"test RMSE = (\S+)")
    lo, hi = re.search(r"\[(\S+), (\S+)\]", d3).groups()
    wn = re.findall(r"lam=(\S+) \((\w+)\): actual lam=\S+, \|\|w\|\|_2 = (\S+)", d4)
    wn = [(f"lam = {l}", n, v) for l, n, v in wn]
    d2s = (f"Without the penalty the normal matrix is extremely ill-conditioned (cond = {c0}), "
           f"whereas adding lam = {lam_reg} on the diagonal brings it down to {c1}, which is why "
           f"gradient descent converges ({it} iterations, relative error {rel}).")
    d3s = (f"The test-RMSE minimum is flat rather than sharp: every lam in [{lo}, {hi}] stays within 5% of the "
           f"minimum ({rmse} at lam* = {lam_star}), so lam need not be tuned precisely. Below {lo} the "
           f"model overfits (train RMSE keeps falling while test RMSE rises), and above {hi} it underfits "
           f"(both RMSEs rise as the weights are shrunk too hard).")
    return (
        "<h2>P2 &ndash; Ridge regression as MAP</h2>"
        + sub("D1 &ndash; Closed-form ridge",
              para("Ridge weights are obtained with np.linalg.solve on (Phi^T Phi + lam I) w = Phi^T t; "
                   "the resulting fits and errors are reported in D3 and D4."))
        + sub("D2 &ndash; Gradient-descent agreement and conditioning",
              fig(2, "p2_d2_gd_convergence.png", f"Ridge GD loss vs iteration (lam = {lam_reg})"),
              table(["Quantity", "Value"], [
                  ("cond(Phi^T Phi)", c0), (f"cond(Phi^T Phi + lam I), lam = {lam_reg}", c1),
                  ("GD iterations to converge", it), ("Relative error vs closed form", rel)]),
              para(d2s))
        + sub("D3 &ndash; RMSE sweep and the lam interval",
              fig(2, "p2_d3_rmse_sweep.png", "Train and test RMSE vs lam"),
              table(["Quantity", "Value"], [("lam*", lam_star), ("Test RMSE at lam*", rmse),
                                            ("Interval (test RMSE within 5% of min)", f"[{lo}, {hi}]")]),
              para(d3s))
        + sub("D4 &ndash; Fitted curves",
              fig(2, "p2_d4_fitted_curves.png", "Fitted curves at lam = 1e-10, lam* and lam = 1e-2"),
              table(["lam", "Regime", "||w||_2"], wn))
    )


# ---------------------------------------------------------------- P3
def build_p3():
    t = load(3)
    d1, d3, d4, d5, d7 = (section(t, i) for i in (1, 3, 4, 5, 7))
    q = lambda p: grab(d1, p)
    rows1 = [
        ("Final E", q(r"Final E\s+= (\S+)")), ("Best E", q(r"Best E\s+= (\S+)")),
        ("E - E*", q(r"E - E\*\s+= (\S+)")), ("E* (3-decimal reference)", q(r"\(E\* = (\S+)\)")),
        ("||w - w*||_2", q(r"\|\|w - w\*\|\|_2 = (\S+)")),
        ("max |Phi_j^T r - lam sign(w_j)| (non-zero w_j)", q(r"\(non-zero\): max .*? = (\S+)")),
        ("max |Phi_j^T r| / lam (zero w_j)", q(r"\(zero\):\s+max .*? = (\S+)")),
        ("Non-zero coefficient indices", q(r"Non-zero coefficients.*?indices (\[.*?\])")),
        ("Zero coefficient indices", q(r"Zero\s+coefficients.*?indices (\[.*?\])")),
    ]
    path = re.findall(r"^\s+(\d\.\d{4}e[+-]\d\d)\s+(\d+)\s*$", d3, re.M)
    lam_star = grab(d4, r"lam\* = (\S+)")
    rm = grab(d4, r"Test RMSE at lam\* = (\S+)")
    nz = grab(d4, r"Number of ~zero coefficients: (\d+)")
    zi = grab(d4, r"Indices of ~zero coefficients: (\[.*?\])")
    cmp_ = re.findall(r"^\s+(Lasso|Ridge)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$", d5, re.M)
    tr = "increase" if "train  RMSE strictly non-decreasing: True" in d7 or \
        "train RMSE strictly non-decreasing: True" in d7 else "do not increase monotonically"
    te = "increase" if re.search(r"test\s+RMSE strictly non-decreasing: True", d7) else "do not increase monotonically"
    d7s = (
        f"The L1 penalty yields exact zeros because a zero coefficient is optimal whenever the subgradient "
        f"condition |Phi_j^T r| <= lam holds (here max |Phi_j^T r|/lam = {rows1[6][1]} on the zero set), "
        f"and the L1 ball has corners on the axes; the L2 penalty only shrinks coefficients, which is why "
        f"ridge leaves {cmp_[1][4]} coefficient(s) below 0.01 against {cmp_[0][4]} for lasso. For "
        f"lam &ge; lam* = {lam_star} the train RMSE and the test RMSE both {te if tr == te else tr + ' / ' + te} "
        f"as lam grows, because the extra sparsity removes model capacity that the data support and the "
        f"fit moves from the noise level ({rm} at lam*) towards underfitting."
    ).replace("&ge;", "\u2265")
    return (
        "<h2>P3 &ndash; Lasso via subgradient descent</h2>"
        + sub("D1 &ndash; Detailed analysis at lam = 0.1",
              fig(3, "p3_d1a_convergence.png", "Subgradient loss and best-so-far at lam = 0.1"),
              table(["Quantity", "Value"], rows1))
        + sub("D2 &ndash; RMSE sweep", fig(3, "p3_d2_rmse_sweep.png", "Lasso train and test RMSE vs lam"))
        + sub("D3 &ndash; Coefficient path",
              fig(3, "p3_d3_coef_path.png", "Coefficient path vs lam"),
              table(["lam", "# ~zero coefficients (|w_j| < 2*eta*lam)"], path))
        + sub("D4 &ndash; Best lam",
              table(["Quantity", "Value"], [("lam*", lam_star), ("Test RMSE at lam*", rm),
                                            ("# ~zero coefficients", nz), ("Indices", zi)]))
        + sub("D5 &ndash; Lasso vs ridge (Legendre basis)",
              table(["Method", "lam*", "Test RMSE", "||w||_2", "# coeffs |w_j| < 0.01"], cmp_))
        + sub("D6 &ndash; Fitted curves",
              fig(3, "p3_d6_fitted_curves.png", "Lasso vs ridge at the best lam of each"))
        + sub("D7 &ndash; Discussion", para(d7s))
    )


# ---------------------------------------------------------------- P4
def build_p4():
    t = load(4)
    hdr = grab(t, r"Hyperparameters: ([^\n]*)")
    d2, d3 = section(t, 2), section(t, 3)
    lam = grab(d2, r"lam = alpha/beta = (\S+)")
    diff = grab(d2, r"\|\|w_MAP - m_N\|\|_2 = (\S+)")
    floor = grab(d3, r"floor 2\*1\.96/sqrt\(beta\) = (\S+)")
    widths = re.findall(r"N=(\d+)\s*: mean width=(\S+), max width=(\S+)", d3)
    n_lo, n_hi = widths[0], widths[-1]
    d4s = (
        f"As N grows the data term beta*Phi^T Phi dominates the posterior precision, so S_N shrinks and the "
        f"parameter-uncertainty part phi^T S_N phi of the predictive variance falls; the mean interval width "
        f"drops from {n_lo[1]} at N={n_lo[0]} to {n_hi[1]} at N={n_hi[0]}. It cannot go below the floor "
        f"2*1.96/sqrt(beta) = {floor}, because the noise term 1/beta in the predictive variance is irreducible "
        f"however much data we collect."
    )
    return (
        "<h2>P4 &ndash; Bayesian linear regression</h2>"
        + sub("D1 &ndash; Posterior predictive (N = 10)",
              para(f"Hyperparameters: {hdr}."),
              fig(4, "p4_d1d2_predictive_N10.png", "Posterior mean, 95% band and w_MAP overlay (N = 10)"))
        + sub("D2 &ndash; Equivalence of posterior mean and MAP",
              code(f"lam = alpha/beta = {lam}\n||w_MAP - m_N||_2 = {diff}"))
        + sub("D3 &ndash; Interval width vs N",
              fig(4, "p4_d3_interval_width.png", "Predictive interval width 2*1.96*sigma(x) for N = 5, 10, 20, 50"),
              table(["N", "Mean width", "Max width"], widths),
              para(f"Irreducible floor 2*1.96/sqrt(beta) = {floor}."))
        + sub("D4 &ndash; Discussion", para(d4s))
    )


NOTES = [
    "P1 solves the polynomial fit in the QR reparametrisation (Phi = QR, alpha = R w), where Q^T Q = I and every curvature equals 1; weights are recovered as w = R^-1 alpha with a linear solve.",
    "No explicit matrix inverses are used for the weights: np.linalg.lstsq (P1 closed form) and np.linalg.solve (P2, P4). Only the P4 posterior covariance, needed for the predictive variance, uses np.linalg.inv.",
    "P3 treats a coefficient as ~0 when |w_j| < 2*eta*lam, the chatter amplitude of constant-step subgradient descent; the P3 D5 comparison and the D1 zero/non-zero split use the fixed rule |w_j| < 0.01.",
    "Subgradient descent with a constant step does not converge exactly: iterates chatter around the optimum with amplitude of order eta*lam, so the final E and ||w - w*|| keep a small non-zero gap (P3 D1), and the best-so-far E is also reported.",
    "The P3 reference E* and w* are 3-decimal values from the assignment, so gaps below about 1e-3 are within their rounding.",
]

CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body { font: 10.5pt/1.45 "Segoe UI", Arial, sans-serif; color: #111; max-width: 180mm; margin: 0 auto; padding: 8mm 0; }
h1 { font-size: 19pt; margin: 0 0 8pt; }
h2 { font-size: 14pt; border-bottom: 1.5px solid #333; padding-bottom: 2pt; margin: 22pt 0 8pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 4pt 0; text-align: justify; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #888; padding: 2pt 8pt; text-align: left; }
th { background: #eee; }
table.team { margin-bottom: 8pt; }
figure { margin: 6pt 0; text-align: center; break-inside: avoid; page-break-inside: avoid; }
img { max-width: 100%; max-height: 120mm; height: auto; }
figcaption { font-size: 9pt; color: #444; }
pre { background: #f5f5f5; border: 1px solid #ccc; padding: 5pt 7pt; font: 8.5pt/1.35 Consolas, monospace;
      white-space: pre-wrap; word-break: break-all; break-inside: avoid; }
.sub { break-inside: auto; }
ul { margin: 4pt 0 4pt 16pt; padding: 0; }
@media print { body { padding: 0; } }
"""


def main():
    team = table(["Name", "ID"], TEAM).replace("<table>", '<table class="team">')
    setup = ("Setup: Python with NumPy implemented from scratch (no scikit-learn); "
             "random seed 42 for SGD and mini-batch shuffling.")
    notes = "<h2>Notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in NOTES) + "</ul>"
    body = (f"<h1>CS F320 Foundations of Data Science - Assignment 1</h1>{team}<p>{html.escape(setup)}</p>"
            + build_p1() + build_p2() + build_p3() + build_p4() + notes)
    doc = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
           f"<title>CS F320 Foundations of Data Science - Assignment 1</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = ROOT / "report.html"
    out_html.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_html}")

    out_pdf = ROOT / "report.pdf"
    out_pdf.unlink(missing_ok=True)
    pandoc = shutil.which("pandoc")
    edge = shutil.which("msedge") or next(
        (p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), None)
    if edge:
        subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out_pdf}", out_html.as_uri()],
                       capture_output=True, timeout=180)
    if not out_pdf.exists() and pandoc:
        subprocess.run([pandoc, str(out_html), "-o", str(out_pdf)], capture_output=True, timeout=180)
    if out_pdf.exists():
        print(f"Wrote {out_pdf}")
    else:
        print("No PDF produced: open report.html in a browser and use Print -> Save as PDF.")


if __name__ == "__main__":
    main()

"""Two-panel diagnostic: E(tau) for q=1 vs optimized q, and q(tau)."""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--q1", default="results_q1.npz")
ap.add_argument("--qopt", default="results_qopt.npz")
ap.add_argument("--label", default="w0=0.1, g=1, t=1")
ap.add_argument("--out", default=str(HERE / "q_stress_diagnostic.png"))
ap.add_argument("--qopt-label", default="q optimized")
ap.add_argument("--tau-max", type=float, default=None)
a = ap.parse_args()
LABEL, OUT = a.label, a.out


def _load(name):
    d = dict(np.load(HERE / name))
    if a.tau_max is not None:
        n = d["tau"].shape[0]
        m = d["tau"] <= a.tau_max + 1e-9
        for k in list(d):
            if d[k].ndim == 1 and d[k].shape[0] == n:
                d[k] = d[k][m]
    return d


q1 = _load(a.q1)
qo = _load(a.qopt)
eref = float(q1["eref"])

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
)

# ---- Panel 1: E(tau) ----
ax1.axhline(eref, color="0.6", ls="--", lw=1.2, label=f"trial energy = {eref:.4f}")
ax1.plot(q1["tau"], q1["E_real"], color="#1f77b4", lw=1.4, label="q = 1 (fixed)")
ax1.plot(qo["tau"], qo["E_real"], color="#d62728", lw=1.4, label=a.qopt_label)
ax1.set_ylabel(r"$E(\tau)$")
ax1.set_title(f"Free-projection stress test, {LABEL},  $d\\tau=5\\times10^{{-5}}$")
ax1.legend(loc="best", framealpha=0.9)
ax1.grid(alpha=0.25)
# Robust y-limits from finite data.
allE = np.concatenate([q1["E_real"], qo["E_real"]])
allE = allE[np.isfinite(allE)]
if allE.size:
    lo, hi = np.percentile(allE, [1, 99])
    pad = 0.05 * (hi - lo + 1e-6)
    ax1.set_ylim(lo - pad, hi + pad)

# ---- Panel 2: q(tau) ----
ax2.axhline(1.0, color="black", ls="--", lw=1.2, label="q = 1")
if "q_spread" in qo and np.any(qo["q_spread"] > 0):
    ax2.fill_between(qo["tau"], qo["q"] - qo["q_spread"], qo["q"] + qo["q_spread"],
                     color="#d62728", alpha=0.2, label="per-walker spread (±1σ)")
    ax2.plot(qo["tau"], qo["q"], color="#d62728", lw=1.6, label="mean per-walker q(τ)")
else:
    ax2.plot(qo["tau"], qo["q"], color="#d62728", lw=1.6, label="optimized q(τ)")
ax2.set_xlabel(r"imaginary time  $\tau$")
ax2.set_ylabel(r"$q(\tau)$")
ax2.set_yscale("log")
ax2.legend(loc="best", framealpha=0.9)
ax2.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
print(f"q optimized: range [{qo['q'].min():.4g}, {qo['q'].max():.4g}], "
      f"final {qo['q'][-1]:.4g}")

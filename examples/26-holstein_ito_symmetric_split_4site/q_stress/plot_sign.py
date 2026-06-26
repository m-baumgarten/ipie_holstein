"""Three-panel sign-problem diagnostic: E(tau), phase coherence(tau), q(tau)."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--q1", required=True)
ap.add_argument("--qpw", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--tau-max", type=float, default=None)
a = ap.parse_args()


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
qp = _load(a.qpw)
eref = float(q1["eref"])

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(8.4, 8.6), sharex=True,
    gridspec_kw={"height_ratios": [1.4, 1.0, 0.9]},
)

ax1.axhline(eref, color="0.6", ls="--", lw=1.2, label=f"trial energy = {eref:.4f}")
ax1.plot(q1["tau"], q1["E_real"], color="#1f77b4", lw=1.3, label="q = 1 (fixed)")
ax1.plot(qp["tau"], qp["E_real"], color="#d62728", lw=1.3, label="per-walker q")
ax1.set_ylabel(r"$E(\tau)$")
ax1.set_title(f"Sign-problem stress test, {a.label},  $d\\tau=5\\times10^{{-5}}$")
ax1.legend(loc="best", framealpha=0.9, fontsize=9)
ax1.grid(alpha=0.25)

ax2.plot(q1["tau"], q1["phase_coherence"], color="#1f77b4", lw=1.4, label="q = 1")
ax2.plot(qp["tau"], qp["phase_coherence"], color="#d62728", lw=1.4, label="per-walker q")
ax2.axhline(1.0, color="0.6", ls=":", lw=1.0)
ax2.set_ylabel("phase coherence\n" + r"$|\langle e^{i\phi}\rangle|$")
ax2.set_ylim(-0.03, 1.05)
ax2.legend(loc="best", framealpha=0.9, fontsize=9)
ax2.grid(alpha=0.25)

ax3.axhline(1.0, color="black", ls="--", lw=1.0, label="q = 1")
if "q_spread" in qp and np.any(qp["q_spread"] > 0):
    ax3.fill_between(qp["tau"], qp["q"] - qp["q_spread"], qp["q"] + qp["q_spread"],
                     color="#d62728", alpha=0.2)
ax3.plot(qp["tau"], qp["q"], color="#d62728", lw=1.5, label="mean per-walker q(τ)")
ax3.set_ylabel(r"$q(\tau)$")
ax3.set_xlabel(r"imaginary time  $\tau$")
ax3.set_yscale("log")
ax3.legend(loc="best", framealpha=0.9, fontsize=9)
ax3.grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(a.out, dpi=160, bbox_inches="tight")
print("saved", a.out)

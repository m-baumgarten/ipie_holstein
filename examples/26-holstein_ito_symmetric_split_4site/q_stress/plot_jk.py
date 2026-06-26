"""Plot E(tau) with jackknife error bars (over FP iterations) for q=1 vs per-walker q."""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios  # noqa: E402


def jk_curve(pattern):
    """Return tau, E_mean, E_err via jackknife over iterations at each block.

    ``pattern`` is a glob (relative to HERE) so several parallel-stream result
    files can be merged into one iteration ensemble.
    """
    files = sorted(HERE.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    parts = [np.load(f)["iters"] for f in files]
    nblk = min(p.shape[1] for p in parts)
    it = np.concatenate([p[:, :nblk, :] for p in parts], axis=0)
    n_it, n_blk, _ = it.shape
    tau = it[0, :, 0]
    num = it[:, :, 1] + 1j * it[:, :, 2]   # (n_iters, n_blocks)
    den = it[:, :, 3] + 1j * it[:, :, 4]
    mean = np.full(n_blk, np.nan)
    err = np.full(n_blk, np.nan)
    for b in range(n_blk):
        nb, db = num[:, b], den[:, b]
        ok = np.isfinite(nb) & np.isfinite(db)
        if ok.sum() >= 2:
            mean[b], err[b] = jackknife_ratios(nb[ok], db[ok])
    return tau, mean, err, n_it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1", required=True)
    ap.add_argument("--qpw", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--eref", type=float, default=None)
    ap.add_argument("--ed", type=float, default=None, help="exact-diagonalization reference")
    ap.add_argument("--fock", default=None, help="fock_imag_time npz: exact E(tau) curve")
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t1, m1, e1, n1 = jk_curve(a.q1)
    t2, m2, e2, n2 = jk_curve(a.qpw)
    if a.eref is not None:
        eref = a.eref
    else:
        eref = float(np.load(sorted(HERE.glob(a.q1))[0])["eref"])

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axhline(eref, color="0.6", ls="--", lw=1.1, label=f"trial energy = {eref:.4f}")
    if a.ed is not None:
        ax.axhline(a.ed, color="green", ls="-.", lw=1.4, label=f"ED = {a.ed:.6f}")
    if a.fock is not None:
        fd = np.load(HERE / a.fock)
        ax.plot(fd["tau"], fd["E"], color="black", lw=2.0, zorder=5,
                label="exact e$^{-\\tau H}$ (Fock)")
    ax.errorbar(t1, m1, yerr=e1, color="#1f77b4", lw=1.2, capsize=2, elinewidth=0.8,
                label=f"q = 1  ({n1} iters)")
    ax.errorbar(t2, m2, yerr=e2, color="#d62728", lw=1.2, capsize=2, elinewidth=0.8,
                label=f"per-walker q  ({n2} iters)")
    if a.ylim is not None:
        ax.set_ylim(*a.ylim)
    if a.xlim is not None:
        ax.set_xlim(*a.xlim)
    ax.set_xlabel(r"imaginary time  $\tau$")
    ax.set_ylabel(r"$E(\tau)$")
    ax.set_title(f"FP energy with jackknife error bars, {a.label}")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("saved", a.out)
    # quick numeric summary at the largest tau
    print(f"q=1     E(tau_max)={m1[-1]:.4f} +/- {e1[-1]:.4f}")
    print(f"per-wlk E(tau_max)={m2[-1]:.4f} +/- {e2[-1]:.4f}")


if __name__ == "__main__":
    main()

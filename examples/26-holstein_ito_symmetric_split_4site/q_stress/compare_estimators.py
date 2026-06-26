"""Compare bias-removal estimators for the FP ratio E(tau=0.5) = <num>/<den>.

Uses per-walker (weight_log, phase, E_L) logged at tau=0.5, pooling all walkers
across iterations as i.i.d. samples (valid for branchless free projection).
Methods: iteration-jackknife (baseline), walker delete-one jackknife,
walker bootstrap (+BCa), median-of-means, analytic 2nd-order ratio correction.
All compared to the exact Fock value.
"""

import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios  # noqa: E402

EXACT = -1.760053
PAT = "q_stress/results_w0_1.0_Kpi2_K1.571_qpwSW_p*_jk.npz"
TAU = 0.5
rng = np.random.default_rng(0)


def main():
    files = sorted(HERE.glob("results_w0_1.0_Kpi2_K1.571_qpwSW_p*_jk.npz"))
    wl, ph, el = [], [], []
    inum, iden = [], []
    for f in files:
        d = np.load(f)
        W = d["walkers"]                       # (n_iter, 5, nw)
        for it in range(W.shape[0]):
            wl.append(W[it, 0]); ph.append(W[it, 1] + 1j * W[it, 2])
            el.append(W[it, 3] + 1j * W[it, 4])
        I = d["iters"]                          # (n_iter, nblk, 6)
        b = int(np.argmin(np.abs(I[0, :, 0] - TAU)))
        inum.append(I[:, b, 1] + 1j * I[:, b, 2])
        iden.append(I[:, b, 3] + 1j * I[:, b, 4])
    wl = np.concatenate(wl); ph = np.concatenate(ph); el = np.concatenate(el)
    inum = np.concatenate(inum); iden = np.concatenate(iden)   # per-iteration pooled
    N = wl.size
    M = inum.size

    # per-walker num/den with a single global log-shift (cancels in the ratio)
    m = np.max(wl)
    w = np.exp(wl - m) * ph
    num = w * el
    den = w
    point = (num.sum() / den.sum()).real

    print(f"exact E(0.5) = {EXACT}")
    print(f"N_walkers = {N}   M_iterations = {M}")
    print(f"[point] pooled ratio              = {point:.5f}   (bias {point-EXACT:+.5f})")

    # (a) iteration-level jackknife (the baseline we used)
    mj, sj = jackknife_ratios(inum, iden)
    print(f"[a] iteration jackknife (M={M})    = {mj.real:.5f} +/- {sj:.5f}   (bias {mj.real-EXACT:+.5f})")

    # (b) walker delete-one jackknife (running-sum leave-one-out)
    Sn, Sd = num.sum(), den.sum()
    r_i = ((Sn - num) / (Sd - den)).real        # leave-one-walker ratios
    R_jk = N * point - (N - 1) * r_i.mean()
    s_jk = np.sqrt((N - 1) * np.var(r_i))
    print(f"[b] walker jackknife (N={N})     = {R_jk:.5f} +/- {s_jk:.5f}   (bias {R_jk-EXACT:+.5f})")

    # (c) walker bootstrap (+ bias-corrected point + percentile & BCa 68% CI)
    B = 4000
    boot = np.empty(B)
    for j in range(B):
        idx = rng.integers(0, N, N)
        boot[j] = (num[idx].sum() / den[idx].sum()).real
    bias_boot = boot.mean() - point
    R_bc = point - bias_boot
    lo, hi = np.percentile(boot, [16, 84])
    # BCa
    z0 = norm.ppf(np.mean(boot < point))
    jk_mean = r_i.mean()
    acc = np.sum((jk_mean - r_i) ** 3) / (6.0 * (np.sum((jk_mean - r_i) ** 2) ** 1.5) + 1e-300)
    def bca(alpha):
        z = norm.ppf(alpha)
        a = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        return np.percentile(boot, 100 * a)
    blo, bhi = bca(0.16), bca(0.84)
    print(f"[c] walker bootstrap bias-corr    = {R_bc:.5f}   (bias {R_bc-EXACT:+.5f})   "
          f"68% pctl [{lo:.4f},{hi:.4f}] BCa [{blo:.4f},{bhi:.4f}]")

    # (d) median-of-means over walker groups
    for g in [16, 64]:
        order = rng.permutation(N)
        groups = np.array_split(order, g)
        gr = np.array([(num[gi].sum() / den[gi].sum()).real for gi in groups])
        med = np.median(gr)
        # bootstrap error of the median over groups
        mb = np.array([np.median(rng.choice(gr, g)) for _ in range(2000)])
        print(f"[d] median-of-means (g={g:3d})       = {med:.5f} +/- {mb.std():.5f}   (bias {med-EXACT:+.5f})")

    # (e) analytic 2nd-order ratio bias correction (complex moments)
    Nb, Db = num.mean(), den.mean()
    m2_dd = np.mean((den - Db) ** 2)
    m2_nd = np.mean((num - Nb) * (den - Db))
    R = Nb / Db
    biasN = (R * m2_dd / Db ** 2 - m2_nd / (Nb * Db)) / N
    R_corr = (R - biasN).real
    print(f"[e] analytic 2nd-order correction = {R_corr:.5f}   (bias {R_corr-EXACT:+.5f})")


if __name__ == "__main__":
    main()

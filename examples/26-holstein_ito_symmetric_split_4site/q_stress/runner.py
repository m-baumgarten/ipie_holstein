"""Resumable free-projection stress-test runner for the scalar-q diffusion gauge.

Each invocation propagates for at most ``--budget`` seconds, checkpoints walker +
propagator state, and exits. Re-invoke until ``DONE`` is printed. Records, per
block: tau, E(tau) = <W phase E_L>/<W phase>, q(tau), and stability diagnostics
(log-weight spread, phase coherence, min |overlap|).

Usage:
  python runner.py <phase> --tag w0_0.25 --w0 0.25 --g 1.0 --tau-max 2.5 [--qfix Q] --budget 40

<phase> is a free label used for filenames (e.g. q1, qopt, q0.1). The gauge it
maps to is controlled by --qfix (fixed q) or --optimize.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # fixture.py

from fixture import build_holstein_ito_fp_inputs, build_workflow  # noqa: E402
from ipie.addons.free_projection.estimators.energy_eph import local_energy  # noqa: E402

NSITES, T = 4, 1.0
DT = 5.0e-5
SPB = 200          # steps per block (tau resolution = SPB*DT = 0.01)
NSTBLZ = 50
SEED = 11


def build(cfg):
    wf = np.load(HERE / f"trial_{cfg['tag']}.npz")["wavefunction"]
    inputs = build_holstein_ito_fp_inputs(
        nsites=NSITES, t=T, g=cfg["g"], w0=cfg["w0"], nwalkers=cfg["nw"],
        k=cfg.get("K", 0.0), wavefunction=wf,
    )
    common = dict(
        timestep=DT, importance_sampling=True, force_bias=cfg.get("force_bias", "overlap"),
        mean_field_subtraction=cfg.get("mf", True),
        split_gauge="phase_cancel", exponential_action=cfg.get("exp_action", "taylor"),
        exponential_taylor_order=cfg.get("taylor_order", 6),
    )
    if cfg.get("per_walker"):
        out = build_workflow(
            inputs, split_gauge_q=1.0, split_gauge_q_per_walker=True,
            split_gauge_q_stride=cfg.get("q_stride"),
            split_gauge_q_smoothing=1.0, split_gauge_electron_cost="sensitivity",
            **common,
        )
    elif cfg["optimize"]:
        out = build_workflow(
            inputs, split_gauge_q=1.0, split_gauge_q_optimize=True,
            split_gauge_q_stride=SPB, split_gauge_q_smoothing=1.0,
            split_gauge_electron_cost="sensitivity", **common,
        )
    else:
        out = build_workflow(
            inputs, split_gauge_q=cfg["qfix"], split_gauge_q_optimize=False, **common,
        )
    sys_, ham, trial, walkers, prop = out
    eref = float(np.real(trial.calc_energy(ham)[0]))
    prop.set_reference_energy(eref)
    return sys_, ham, trial, walkers, prop, eref


def block_energy(sys_, ham, trial, walkers):
    walkers.ovlp = trial.calc_overlap(walkers)
    trial.calc_greens_function(walkers)
    e = np.asarray(local_energy(sys_, ham, walkers, trial))[:, 0]
    wl = np.real(np.asarray(walkers.weight_log, dtype=np.complex128))
    ph = np.asarray(walkers.phase, dtype=np.complex128)
    ov = np.asarray(walkers.ovlp, dtype=np.complex128)
    fin = (np.isfinite(wl) & np.isfinite(e.real) & np.isfinite(e.imag)
           & np.isfinite(ph.real) & np.isfinite(ph.imag))
    if not np.any(fin):
        return complex(np.nan, np.nan), np.nan, np.nan, np.nan, 0.0
    wl_f = wl[fin]
    m = np.max(wl_f)
    w = np.exp(wl_f - m) * ph[fin]
    den = np.sum(w)
    E = np.sum(w * e[fin]) / den if np.abs(den) > 0 else complex(np.nan, np.nan)
    spread = float(np.max(wl_f) - np.min(wl_f))
    coherence = float(np.abs(den) / np.sum(np.exp(wl_f - m)))
    ov_min = float(np.min(np.abs(ov[fin])))
    frac_finite = float(np.count_nonzero(fin) / fin.size)
    return complex(E), spread, coherence, ov_min, frac_finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--tag", default="w0_0.1")
    ap.add_argument("--w0", type=float, default=0.1)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=0.0)
    ap.add_argument("--nw", type=int, default=400)
    ap.add_argument("--tau-max", type=float, default=4.0)
    ap.add_argument("--qfix", type=float, default=1.0)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--per-walker", action="store_true",
                    help="per-walker q updated every step")
    ap.add_argument("--no-mf", action="store_true", help="disable mean-field subtraction")
    ap.add_argument("--force-bias", choices=("overlap", "zero"), default="overlap")
    ap.add_argument("--budget", type=float, default=40.0)
    args = ap.parse_args()

    cfg = dict(tag=args.tag, w0=args.w0, g=args.g, K=args.K, nw=args.nw,
               qfix=args.qfix, optimize=args.optimize, mf=not args.no_mf,
               force_bias=args.force_bias, per_walker=args.per_walker)
    total_steps = int(round(args.tau_max / DT))
    stem = f"{args.tag}_{args.phase}"
    state_path = HERE / f"state_{stem}.pkl"
    res_path = HERE / f"results_{stem}.npz"

    if state_path.exists():
        st = pickle.load(open(state_path, "rb"))
        sys_, ham, trial, _, _, eref = build(cfg)
        walkers, prop = st["walkers"], st["prop"]
        np.random.set_state(st["rng"])
        step, rows = st["step"], st["rows"]
        rows = [tuple(r) + (0.0,) * (9 - len(r)) for r in rows]  # pad legacy 8-col rows
    else:
        sys_, ham, trial, walkers, prop, eref = build(cfg)
        np.random.seed(SEED)
        prop.initialize_importance_weights(walkers, trial)
        walkers.orthogonalise(free_projection=False)
        step, rows = 0, []

    t0 = time.time()
    while step < total_steps and (time.time() - t0) < args.budget:
        step += 1
        if step % NSTBLZ == 0:
            walkers.orthogonalise(free_projection=False)
        prop.propagate_walkers(walkers, ham, trial, eref)
        if step % SPB == 0:
            E, spread, coh, ovm, ff = block_energy(sys_, ham, trial, walkers)
            qv = prop.split_gauge_q
            q_mean = float(np.mean(qv))
            q_spread = float(np.std(qv)) if np.ndim(qv) > 0 else 0.0
            rows.append((step * DT, E.real, E.imag, q_mean,
                         spread, coh, ovm, ff, q_spread))

    pickle.dump({"walkers": walkers, "prop": prop, "rng": np.random.get_state(),
                 "step": step, "rows": rows, "eref": eref}, open(state_path, "wb"))
    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 9))
    np.savez(res_path, tau=arr[:, 0], E_real=arr[:, 1], E_imag=arr[:, 2], q=arr[:, 3],
             log_weight_spread=arr[:, 4], phase_coherence=arr[:, 5],
             overlap_min=arr[:, 6], frac_finite=arr[:, 7], q_spread=arr[:, 8],
             eref=np.asarray(eref))

    last = rows[-1] if rows else None
    msg = f"[{stem}] step {step}/{total_steps} tau={step*DT:.3f}"
    if last is not None:
        msg += f"  E={last[1]:.4f} q={last[3]:.4g} coh={last[5]:.3f} spread={last[4]:.1f} ovmin={last[6]:.1e}"
    print(msg)
    print("DONE" if step >= total_steps else "CONTINUE")


if __name__ == "__main__":
    main()

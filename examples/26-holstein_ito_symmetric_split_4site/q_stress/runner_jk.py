"""Multi-iteration free-projection runner for jackknife error bars.

Runs N_iters independent FP trajectories (fresh walkers + distinct seed each),
recording per block the *raw* numerator and denominator sums
  ENumer = sum_w weight*phase*E_L,  EDenom = sum_w weight*phase,
so the ratio E(tau)=ENumer/EDenom can be jackknifed over iterations (exactly as
in ipie's FP examples). Resumable at the iteration level across 45s calls.

Usage:
  python runner_jk.py <phase> --tag w0_0.25 --w0 0.25 --g 1.0 --K 0 --nw 200 \
      --tau-max 2.0 --iters 12 [--qfix Q | --optimize | --per-walker] --budget 40
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner as R  # build(), block_energy(), constants  # noqa: E402
from ipie.addons.free_projection.estimators.energy_eph import local_energy  # noqa: E402


def block_sums(sys_, ham, trial, walkers):
    """Return (ENumer, EDenom) raw complex sums for this population (logsumexp-safe)."""
    walkers.ovlp = trial.calc_overlap(walkers)
    trial.calc_greens_function(walkers)
    e = np.asarray(local_energy(sys_, ham, walkers, trial))[:, 0]
    wl = np.real(np.asarray(walkers.weight_log, dtype=np.complex128))
    ph = np.asarray(walkers.phase, dtype=np.complex128)
    fin = (np.isfinite(wl) & np.isfinite(e.real) & np.isfinite(e.imag)
           & np.isfinite(ph.real) & np.isfinite(ph.imag))
    if not np.any(fin):
        return complex(np.nan), complex(np.nan)
    wl_f = wl[fin]
    m = np.max(wl_f)                      # common shift cancels in the ratio
    w = np.exp(wl_f - m) * ph[fin]
    return complex(np.sum(w * e[fin])), complex(np.sum(w))


def walker_terms(sys_, ham, trial, walkers):
    """Per-walker (log|weight|, phase, E_L) for walker-level resampling."""
    walkers.ovlp = trial.calc_overlap(walkers)
    trial.calc_greens_function(walkers)
    e = np.asarray(local_energy(sys_, ham, walkers, trial))[:, 0]
    wl = np.real(np.asarray(walkers.weight_log, dtype=np.complex128))
    ph = np.asarray(walkers.phase, dtype=np.complex128)
    return wl, ph, e


def run_iteration(cfg, seed, tau_max, budget_left, st_iter):
    """Advance one iteration; return (state, done_bool). Resumable mid-iteration."""
    total_steps = int(round(tau_max / R.DT))
    if st_iter is None:
        sys_, ham, trial, walkers, prop, eref = R.build(cfg)
        np.random.seed(seed)
        prop.initialize_importance_weights(walkers, trial)
        walkers.orthogonalise(free_projection=False)
        st_iter = dict(walkers=walkers, prop=prop, step=0, rows=[],
                       rng=np.random.get_state(), eref=eref)
    else:
        sys_, ham, trial, _, _, eref = R.build(cfg)
        walkers, prop = st_iter["walkers"], st_iter["prop"]
        np.random.set_state(st_iter["rng"])

    step, rows = st_iter["step"], st_iter["rows"]
    t0 = time.time()
    while step < total_steps and (time.time() - t0) < budget_left:
        step += 1
        if step % R.NSTBLZ == 0:
            walkers.orthogonalise(free_projection=False)
        prop.propagate_walkers(walkers, ham, trial, eref)
        if step % R.SPB == 0:
            num, den = block_sums(sys_, ham, trial, walkers)
            q = prop.split_gauge_q
            rows.append((step * R.DT, num.real, num.imag, den.real, den.imag,
                         float(np.mean(q))))
    st_iter.update(step=step, rows=rows, rng=np.random.get_state())
    done = step >= total_steps
    if done and cfg.get("save_walkers"):
        wl, ph, e = walker_terms(sys_, ham, trial, walkers)
        st_iter["wdata"] = (wl, ph.real, ph.imag, e.real, e.imag)
    return st_iter, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--tag", default="w0_0.25")
    ap.add_argument("--w0", type=float, default=0.25)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=0.0)
    ap.add_argument("--nw", type=int, default=200)
    ap.add_argument("--tau-max", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--qfix", type=float, default=1.0)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--per-walker", action="store_true")
    ap.add_argument("--q-stride", type=int, default=None,
                    help="recompute per-walker q every N steps (default: every step)")
    ap.add_argument("--no-mf", action="store_true")
    ap.add_argument("--force-bias", choices=("overlap", "zero"), default="overlap")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="offset so parallel streams use independent seeds")
    ap.add_argument("--exp-action", choices=("taylor", "expm"), default="taylor")
    ap.add_argument("--taylor-order", type=int, default=6)
    ap.add_argument("--save-walkers", action="store_true",
                    help="log per-walker (weight_log, phase, E_L) at the final block")
    ap.add_argument("--budget", type=float, default=40.0)
    args = ap.parse_args()

    cfg = dict(tag=args.tag, w0=args.w0, g=args.g, K=args.K, nw=args.nw,
               qfix=args.qfix, optimize=args.optimize, mf=not args.no_mf,
               force_bias=args.force_bias, per_walker=args.per_walker,
               q_stride=args.q_stride,
               exp_action=args.exp_action, taylor_order=args.taylor_order,
               save_walkers=args.save_walkers)
    stem = f"{args.tag}_K{args.K:.3f}_{args.phase}_jk"
    state_path = HERE / f"state_{stem}.pkl"
    res_path = HERE / f"results_{stem}.npz"

    if state_path.exists():
        st = pickle.load(open(state_path, "rb"))
        st.setdefault("wdata", [])
    else:
        st = dict(iters=[], cur=None, idx=0, eref=None, wdata=[])

    t0 = time.time()
    while st["idx"] < args.iters and (time.time() - t0) < args.budget:
        seed = R.SEED + 1000 * (args.seed_offset + st["idx"] + 1)
        st_iter, done = run_iteration(cfg, seed, args.tau_max,
                                      args.budget - (time.time() - t0), st["cur"])
        st["cur"] = st_iter
        st["eref"] = st_iter["eref"]
        if done:
            st["iters"].append(np.array(st_iter["rows"], dtype=np.float64))
            if "wdata" in st_iter:
                st["wdata"].append(np.array(st_iter["wdata"], dtype=np.float64))
            st["cur"] = None
            st["idx"] += 1

    pickle.dump(st, open(state_path, "wb"))
    if st["iters"]:
        out = dict(iters=np.stack(st["iters"]),
                   eref=np.asarray(st["eref"] if st["eref"] is not None else 0.0))
        if st["wdata"]:
            # (n_iter, 5, nw): rows = weight_log, phase_re, phase_im, EL_re, EL_im
            out["walkers"] = np.stack(st["wdata"])
        np.savez(res_path, **out)
    cur_step = st["cur"]["step"] if st["cur"] is not None else 0
    print(f"[{stem}] completed {st['idx']}/{args.iters} iters; "
          f"current iter step {cur_step}/{int(round(args.tau_max/R.DT))}")
    print("DONE" if st["idx"] >= args.iters else "CONTINUE")


if __name__ == "__main__":
    main()

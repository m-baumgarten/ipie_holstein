# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run four-site Holstein free projection with the symmetric-split Ito CS propagator."""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_cache_dirs():
    """Keep optional plotting/font caches out of the user's home directory."""
    cache_roots = {
        "MPLCONFIGDIR": "ipie_mplconfig",
        "XDG_CACHE_HOME": "ipie_xdg_cache",
    }
    for env_name, dirname in cache_roots.items():
        cache_dir = Path(os.environ.get(env_name, str(Path("/tmp") / dirname)))
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(cache_dir))


configure_cache_dirs()

import ipie
from ipie.addons.free_projection.qmc.fp_afqmc_eph import FPAFQMC
from ipie.addons.free_projection.qmc.options import QMCParamsFP
from ipie.utils.mpi import MPIHandler

from fixture import build_holstein_ito_fp_inputs, build_workflow, load_trial_wavefunction


class LiveTee:
    """Line-flushing stdout tee for long AFQMC runs."""

    def __init__(self, stream, filename):
        self.stream = stream
        path = Path(filename)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        if "\n" in text:
            self.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.flush()
        self.file.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nwalkers", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=500)
    parser.add_argument("--steps-per-block", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timestep", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--stabilize-freq", type=int, default=20)
    parser.add_argument("--pop-control-freq", type=int, default=-1)
    parser.add_argument("--estimator-file", default="estimate.h5")
    parser.add_argument("--output-file", default="fp_ito.out")
    parser.add_argument(
        "--importance-sampling",
        action="store_true",
        help="Use the importance-sampled symmetric-split Ito propagator.",
    )
    parser.add_argument(
        "--force-bias",
        choices=("overlap", "zero"),
        default="overlap",
        help="Force-bias mode for importance-sampled propagation.",
    )
    parser.add_argument(
        "--split-gauge",
        choices=("static", "phase_cancel"),
        default="static",
        help="Split gauge for importance-sampled creation propagation.",
    )
    parser.add_argument(
        "--split-gauge-scale",
        type=float,
        default=1.0,
        help="Scale applied to the dynamic split gauge.",
    )
    parser.add_argument(
        "--split-gauge-max-norm",
        type=float,
        default=None,
        help="Optional per-walker L2 cap for the dynamic split gauge.",
    )
    parser.add_argument(
        "--split-gauge-q",
        type=float,
        default=1.0,
        help="Global scalar Q gauge for phase-cancel importance sampling.",
    )
    parser.add_argument(
        "--split-gauge-q-optimize",
        action="store_true",
        help="Periodically re-optimize the scalar q gauge (Green-Kubo, Sec. 11.5).",
    )
    parser.add_argument(
        "--split-gauge-q-stride",
        type=int,
        default=None,
        help="Refresh q every N steps (default: once at the first step).",
    )
    parser.add_argument(
        "--split-gauge-q-min",
        type=float,
        default=None,
        help="Lower clamp for the optimized q (trust region).",
    )
    parser.add_argument(
        "--split-gauge-q-max",
        type=float,
        default=None,
        help="Upper clamp for the optimized q (trust region).",
    )
    parser.add_argument(
        "--split-gauge-q-smoothing",
        type=float,
        default=1.0,
        help="Geometric blend alpha in (0,1] for q updates (1.0 = full update).",
    )
    parser.add_argument(
        "--split-gauge-electron-cost",
        choices=("auto", "sensitivity", "kick"),
        default="auto",
        help="Electron cost metric for q optimization.",
    )
    parser.add_argument(
        "--exponential-action",
        choices=("expm", "taylor"),
        default="expm",
        help="How to apply per-walker matrix exponentials in the symmetric split.",
    )
    parser.add_argument(
        "--exponential-taylor-order",
        type=int,
        default=6,
        help="Taylor order used when --exponential-action=taylor.",
    )
    parser.add_argument(
        "--reference-energy",
        type=float,
        default=None,
        help="Constant E_ref subtracted from H. Defaults to the trial variational energy.",
    )
    parser.add_argument(
        "--no-reference-energy-shift",
        action="store_true",
        help="Disable the scalar exp(dt * E_ref) reference-energy weight factor.",
    )
    parser.add_argument(
        "--diagnostics-file",
        default=None,
        help="Optional .npz path for per-step importance-propagator diagnostics.",
    )
    parser.add_argument(
        "--diagnostics-stride",
        type=int,
        default=1,
        help="Record every Nth propagation step when diagnostics are enabled.",
    )
    parser.add_argument(
        "--diagnostics-max-records",
        type=int,
        default=None,
        help="Optional cap on the number of diagnostic step summaries per rank.",
    )

    parser.add_argument("--nsites", type=int, default=4)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0.0)
    parser.add_argument("--trial-file", default=None)
    return parser.parse_args()


def energy_statistics(enumer, edenom, jackknife_ratios):
    """Return ratio-of-means statistics, allowing one-sample smoke runs."""
    enumer = np.asarray(enumer, dtype=np.complex128)
    edenom = np.asarray(edenom, dtype=np.complex128)
    active = np.isfinite(enumer) & np.isfinite(edenom) & (np.abs(edenom) > 0.0)
    enumer = enumer[active]
    edenom = edenom[active]
    if enumer.size == 0:
        return np.nan + 0.0j, np.nan
    if enumer.size == 1:
        return enumer[0] / edenom[0], np.nan
    return jackknife_ratios(enumer, edenom)


def compute_local_walker_count(requested_total_walkers, comm_size):
    if requested_total_walkers < comm_size:
        raise ValueError(
            "--nwalkers must be at least the number of MPI ranks "
            f"({comm_size}); got {requested_total_walkers}."
        )
    if requested_total_walkers % comm_size != 0:
        raise ValueError(
            "--nwalkers must be divisible by the number of MPI ranks "
            f"({comm_size}) for equal local walker batches; got {requested_total_walkers}."
        )
    return requested_total_walkers // comm_size


def main():
    args = parse_args()
    mpi_handler = MPIHandler()
    comm = mpi_handler.comm
    output_tee = None
    if comm.rank == 0 and args.output_file:
        output_tee = LiveTee(sys.stdout, args.output_file)
        sys.stdout = output_tee

    try:
        nwalkers_local = compute_local_walker_count(args.nwalkers, comm.size)
        nwalkers_actual = nwalkers_local * comm.size
        wavefunction = None
        trial_source = "built-in deterministic seed"
        if args.trial_file is not None:
            wavefunction = load_trial_wavefunction(args.trial_file, args.nsites)
            trial_source = f"loaded from {args.trial_file}"

        inputs = build_holstein_ito_fp_inputs(
            nsites=args.nsites,
            t=args.t,
            g=args.g,
            w0=args.w0,
            nwalkers=nwalkers_local,
            k=args.k,
            beta_scale=args.beta_scale,
            wavefunction=wavefunction,
        )
        system, ham, trial, walkers, propagator = build_workflow(
            inputs,
            timestep=args.timestep,
            importance_sampling=args.importance_sampling,
            force_bias=args.force_bias,
            split_gauge=args.split_gauge,
            split_gauge_scale=args.split_gauge_scale,
            split_gauge_max_norm=args.split_gauge_max_norm,
            split_gauge_q=args.split_gauge_q,
            split_gauge_q_optimize=args.split_gauge_q_optimize,
            split_gauge_q_stride=args.split_gauge_q_stride,
            split_gauge_q_bounds=(
                None
                if (args.split_gauge_q_min is None and args.split_gauge_q_max is None)
                else (
                    args.split_gauge_q_min if args.split_gauge_q_min is not None else 1e-8,
                    args.split_gauge_q_max if args.split_gauge_q_max is not None else 1e8,
                )
            ),
            split_gauge_q_smoothing=args.split_gauge_q_smoothing,
            split_gauge_electron_cost=args.split_gauge_electron_cost,
            exponential_action=args.exponential_action,
            exponential_taylor_order=args.exponential_taylor_order,
            mpi_handler=mpi_handler,
        )
        trial_total_energy = trial.calc_energy(ham)[0]
        reference_energy = None
        if not args.no_reference_energy_shift:
            reference_energy = (
                args.reference_energy
                if args.reference_energy is not None
                else float(np.real(trial_total_energy))
            )
            propagator.set_reference_energy(reference_energy)
        diagnostics_path = None
        if args.diagnostics_file is not None:
            diagnostics_path = propagator.configure_diagnostics(
                args.diagnostics_file,
                mpi_handler=mpi_handler,
                stride=args.diagnostics_stride,
                max_records=args.diagnostics_max_records,
                steps_per_block=args.steps_per_block,
            )

        if comm.rank == 0:
            print("# Four-site Holstein free projection with symmetric-split Ito coherent-state walkers.")
            print(f"# run script = {Path(__file__).resolve()}")
            print(f"# repo root = {REPO_ROOT}")
            print(f"# ipie module = {Path(ipie.__file__).resolve()}")
            print(f"# propagator = {type(propagator).__name__}")
            print(f"# importance sampling = {args.importance_sampling}")
            if args.importance_sampling:
                print(f"# force bias = {args.force_bias}")
                print(f"# split gauge = {args.split_gauge}")
                print(f"# split gauge scale = {args.split_gauge_scale}")
                print(f"# split gauge max norm = {args.split_gauge_max_norm}")
                print(f"# split gauge q = {args.split_gauge_q}")
                print(f"# split gauge q optimize = {args.split_gauge_q_optimize}")
                if args.split_gauge_q_optimize:
                    print(f"# split gauge q stride = {args.split_gauge_q_stride}")
                    print(f"# split gauge q min = {args.split_gauge_q_min}")
                    print(f"# split gauge q max = {args.split_gauge_q_max}")
                    print(f"# split gauge q smoothing = {args.split_gauge_q_smoothing}")
                    print(f"# split gauge electron cost = {args.split_gauge_electron_cost}")
            print(f"# exponential action = {args.exponential_action}")
            if args.exponential_action == "taylor":
                print(f"# exponential Taylor order = {args.exponential_taylor_order}")
            print(f"# nsites = {args.nsites}")
            print(f"# t = {args.t}")
            print(f"# g = {args.g}")
            print(f"# w0 = {args.w0}")
            print(f"# K = {args.k}")
            print(f"# MPI ranks = {comm.size}")
            print(f"# requested total walkers = {args.nwalkers}")
            print(f"# actual total walkers = {nwalkers_actual}")
            print(f"# walkers per rank = {nwalkers_local}")
            print(f"# timestep = {args.timestep}")
            print(f"# blocks = {args.blocks}")
            print(f"# steps per block = {args.steps_per_block}")
            print(f"# FP iterations = {args.iterations}")
            print(f"# stabilize freq = {args.stabilize_freq}")
            print(f"# pop-control freq = {args.pop_control_freq}")
            print(f"# trial source = {trial_source}")
            print(f"# trial variational energy = {trial_total_energy}")
            if reference_energy is None:
                print("# reference energy shift = disabled")
            else:
                print(f"# reference energy shift = {reference_energy}")
            if diagnostics_path is None:
                print("# diagnostics file = disabled")
            else:
                print(f"# diagnostics file = {diagnostics_path}")
                print(f"# diagnostics stride = {args.diagnostics_stride}")
                print(f"# diagnostics max records = {args.diagnostics_max_records}")

        params = QMCParamsFP(
            num_walkers=nwalkers_local,
            total_num_walkers=nwalkers_actual,
            num_blocks=args.blocks,
            num_steps_per_block=args.steps_per_block,
            timestep=args.timestep,
            num_stblz=args.stabilize_freq,
            pop_control_freq=args.pop_control_freq,
            rng_seed=args.seed,
            num_iterations_fp=args.iterations,
        )
        qmc = FPAFQMC(
            system,
            ham,
            trial,
            walkers,
            propagator,
            params,
            verbose=(comm.rank == 0),
        )
        try:
            qmc.run(
                estimator_filename=args.estimator_file,
                importance_sampling=args.importance_sampling,
                verbose=(comm.rank == 0),
            )
            qmc.finalise(verbose=(comm.rank == 0))

            # analysis
            if comm.rank == 0:
                from ipie.addons.free_projection.analysis.extraction import extract_observable
                from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios

                data = np.zeros((qmc.params.num_blocks, 3), dtype=np.complex128)
                for i in range(qmc.params.num_blocks):
                    data[i, 0] = (i+1) * qmc.params.num_steps_per_block * qmc.params.timestep
                    print(
                        f"\nEnergy statistics at time {(i+1) * qmc.params.num_steps_per_block * qmc.params.timestep}:"
                    )
                    qmc_data = extract_observable(qmc.estimators[i].filename, "energy")
                    energy_mean, energy_err = energy_statistics(
                        qmc_data["ENumer"], qmc_data["EDenom"], jackknife_ratios
                    )
                    data[i, 1], data[i, 2] = energy_mean, energy_err
                    print(f"Energy: {energy_mean:.8e} +/- {energy_err:.8e}")
                np.save('fp_data.npy', data)
        finally:
            if diagnostics_path is not None:
                propagator.save_diagnostics(reason="driver_finally")


    finally:
        if output_tee is not None:
            sys.stdout = output_tee.stream
            output_tee.close()


def abort_mpi_on_exception():
    """Tear down all MPI ranks after an uncaught driver exception."""
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        if comm.Get_size() > 1:
            comm.Abort(1)
    except Exception:
        pass
    raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        abort_mpi_on_exception()

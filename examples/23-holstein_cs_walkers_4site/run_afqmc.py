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
"""Run a four-site Holstein model with coherent-state walkers."""

import argparse
import os
import sys
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
        cache_dir = Path(os.environ.get(env_name, f"/private/tmp/{dirname}"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(cache_dir))


configure_cache_dirs()

import ipie
from ipie.addons.eph.estimators.energy import EnergyEstimator
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.qmc.afqmc import AFQMC
from ipie.utils.mpi import MPIHandler

from fixture import build_holstein_cs_inputs, build_workflow, load_trial_wavefunction


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
    parser.add_argument("--nwalkers", type=int, default=10000)
    parser.add_argument("--blocks", type=int, default=200)
    parser.add_argument("--steps-per-block", type=int, default=5)
    parser.add_argument("--timestep", type=float, default=0.00005)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--stabilize-freq", type=int, default=3)
    parser.add_argument("--pop-control-freq", type=int, default=3)
    parser.add_argument("--estimator-file", default="estimates.0.h5")
    parser.add_argument("--energy-file", default="energy.dat")
    parser.add_argument("--output-file", default="afqmc.out")

    parser.add_argument("--nsites", type=int, default=4)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0.0)
    parser.add_argument("--trial-file", default=None)
    parser.add_argument("--optimize-trial", action="store_true")
    parser.add_argument("--save-trial", default=None)
    parser.add_argument("--optimization-maxiter", type=int, default=1000)
    parser.add_argument("--optimization-starts", type=int, default=4)
    parser.add_argument("--optimization-seed", type=int, default=7)
    parser.add_argument("--optimization-noise", type=float, default=0.05)
    parser.add_argument("--complex-trial", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mpi_handler = MPIHandler()
    comm = mpi_handler.comm
    output_tee = None
    if comm.rank == 0 and args.output_file:
        output_tee = LiveTee(sys.stdout, args.output_file)
        sys.stdout = output_tee

    try:
        nwalkers_local = max(1, args.nwalkers // comm.size)
        wavefunction = None
        trial_source = "built-in unoptimized seed"
        if args.trial_file is not None:
            wavefunction = load_trial_wavefunction(args.trial_file, args.nsites)
            trial_source = f"loaded from {args.trial_file}"

        inputs = build_holstein_cs_inputs(
            nsites=args.nsites,
            t=args.t,
            g=args.g,
            w0=args.w0,
            nwalkers=nwalkers_local,
            k=args.k,
            beta_scale=args.beta_scale,
            wavefunction=wavefunction,
        )

        optimization_result = None
        if args.optimize_trial:
            from ipie.addons.eph.trial_wavefunction.variational.jax.site_toyozawa import (
                optimise_toyozawa_site_jax_restarts,
            )

            opt_ham = HolsteinModel(
                g=args.g,
                t=args.t,
                w0=args.w0,
                nsites=args.nsites,
                pbc=True,
            )
            opt_ham.build()
            optimization_result = optimise_toyozawa_site_jax_restarts(
                opt_ham,
                inputs.beta_shift,
                inputs.electron_orbital,
                K=args.k,
                complex_params=args.complex_trial,
                maxiter=args.optimization_maxiter,
                nstarts=args.optimization_starts,
                seed=args.optimization_seed,
                noise=args.optimization_noise,
            )
            inputs = build_holstein_cs_inputs(
                nsites=args.nsites,
                t=args.t,
                g=args.g,
                w0=args.w0,
                nwalkers=nwalkers_local,
                k=args.k,
                beta_scale=args.beta_scale,
                wavefunction=optimization_result.wavefunction,
            )
            trial_source = "optimized with JAX in this run"

            if comm.rank == 0 and args.save_trial is not None:
                save_path = Path(args.save_trial)
                if save_path.parent != Path("."):
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                numpy_data = {
                    "wavefunction": optimization_result.wavefunction,
                    "beta": optimization_result.beta,
                    "orbital": optimization_result.orbital,
                    "energy": optimization_result.energy,
                    "initial_energy": optimization_result.initial_energy,
                    "success": optimization_result.success,
                    "message": optimization_result.message,
                    "nit": optimization_result.nit,
                    "nfev": optimization_result.nfev,
                    "nsites": args.nsites,
                    "t": args.t,
                    "g": args.g,
                    "w0": args.w0,
                    "k": args.k,
                }
                np.savez(save_path, **numpy_data)

        system, ham, trial, walkers, _ = build_workflow(inputs, timestep=args.timestep)
        trial_total_energy = trial.calc_energy(ham)[0]

        if comm.rank == 0:
            print("# Four-site Holstein with coherent-state walkers.")
            print(f"# run script = {Path(__file__).resolve()}")
            print(f"# repo root = {REPO_ROOT}")
            print(f"# ipie module = {Path(ipie.__file__).resolve()}")
            print(f"# nsites = {args.nsites}")
            print(f"# t = {args.t}")
            print(f"# g = {args.g}")
            print(f"# w0 = {args.w0}")
            print(f"# K = {args.k}")
            print(f"# nwalkers per rank = {nwalkers_local}")
            print(f"# timestep = {args.timestep}")
            print(f"# steps per block = {args.steps_per_block}")
            print(f"# pop-control freq = {args.pop_control_freq}")
            print(f"# trial source = {trial_source}")
            print(f"# trial variational energy = {trial_total_energy}")
            if args.trial_file is not None:
                print(f"# trial file = {args.trial_file}")
            if optimization_result is not None:
                print(f"# Toyozawa initial energy = {optimization_result.initial_energy}")
                print(f"# Toyozawa optimized energy = {optimization_result.energy}")
                print(f"# Toyozawa optimizer success = {optimization_result.success}")
                if args.save_trial is not None:
                    print(f"# saved optimized trial = {args.save_trial}")

        qmc = AFQMC.build(
            num_elec=inputs.nelec,
            hamiltonian=ham,
            trial_wavefunction=trial,
            walkers=walkers,
            num_walkers=nwalkers_local,
            seed=args.seed,
            num_steps_per_block=args.steps_per_block,
            num_blocks=args.blocks,
            timestep=args.timestep,
            stabilize_freq=args.stabilize_freq,
            pop_control_freq=args.pop_control_freq,
            verbose=(comm.rank == 0),
            mpi_handler=mpi_handler,
        )

        additional_estimators = {
            "energy": EnergyEstimator(
                system=qmc.system,
                ham=ham,
                trial=trial,
                filename=args.energy_file,
            )
        }
        qmc.run(
            estimator_filename=args.estimator_file,
            additional_estimators=additional_estimators,
            verbose=(comm.rank == 0),
        )
        qmc.finalise(verbose=(comm.rank == 0))
    finally:
        if output_tee is not None:
            sys.stdout = output_tee.stream
            output_tee.close()


if __name__ == "__main__":
    main()

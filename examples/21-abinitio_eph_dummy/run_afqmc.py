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
"""Run the ab-initio EPh dD2 fixture through ipie's AFQMC driver."""

import argparse
import os
import sys
from pathlib import Path


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

from ipie.addons.eph.estimators.energy import EnergyEstimator
from ipie.qmc.afqmc import AFQMC
from ipie.utils.mpi import MPIHandler

from fixture import build_fixture_inputs, build_workflow


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
    parser.add_argument("--nwalkers", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--steps-per-block", type=int, default=5)
    parser.add_argument("--timestep", type=float, default=1.0e-7)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--fixture-seed", type=int, default=7)
    parser.add_argument("--stabilize-freq", type=int, default=20)
    parser.add_argument("--pop-control-freq", type=int, default=20)
    parser.add_argument("--estimator-file", default="estimates.0.h5")
    parser.add_argument("--energy-file", default="energy.dat")
    parser.add_argument("--output-file", default="afqmc.out")
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
        inputs = build_fixture_inputs(seed=args.fixture_seed, nwalkers=nwalkers_local)
        system, ham, trial, walkers, _ = build_workflow(inputs, timestep=args.timestep)

        qmc = AFQMC.build(
            num_elec=(1, 0),
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

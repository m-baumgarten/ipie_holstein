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
"""Optimize the four-site Holstein Toyozawa seed in the site basis with JAX."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def configure_cache_dirs():
    cache_roots = {
        "MPLCONFIGDIR": "ipie_mplconfig",
        "XDG_CACHE_HOME": "ipie_xdg_cache",
    }
    for env_name, dirname in cache_roots.items():
        cache_dir = Path(os.environ.get(env_name, f"/private/tmp/{dirname}"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(cache_dir))


configure_cache_dirs()

from fixture import build_holstein_cs_inputs
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.variational.jax.site_toyozawa import (
    optimise_toyozawa_site_jax_restarts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsites", type=int, default=4)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--complex-trial", action="store_true")
    parser.add_argument("--output", default="optimized_toyozawa_trial.npz")
    return parser.parse_args()


def main():
    args = parse_args()
    ham = HolsteinModel(g=args.g, t=args.t, w0=args.w0, nsites=args.nsites, pbc=True)
    ham.build()
    inputs = build_holstein_cs_inputs(
        nsites=args.nsites,
        t=args.t,
        g=args.g,
        w0=args.w0,
        k=args.k,
        beta_scale=args.beta_scale,
    )
    result = optimise_toyozawa_site_jax_restarts(
        ham,
        inputs.beta_shift,
        inputs.electron_orbital,
        K=args.k,
        complex_params=args.complex_trial,
        maxiter=args.maxiter,
        nstarts=args.starts,
        seed=args.seed,
        noise=args.noise,
    )

    output = Path(args.output)
    if output.parent != Path("."):
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        wavefunction=result.wavefunction,
        beta=result.beta,
        orbital=result.orbital,
        energy=result.energy,
        initial_energy=result.initial_energy,
        success=result.success,
        message=result.message,
        nit=result.nit,
        nfev=result.nfev,
        nsites=args.nsites,
        t=args.t,
        g=args.g,
        w0=args.w0,
        k=args.k,
    )

    print(f"# initial energy = {result.initial_energy:.16e}")
    print(f"# optimized energy = {result.energy:.16e}")
    print(f"# success = {result.success}; nit = {result.nit}; nfev = {result.nfev}")
    print(f"# message = {result.message}")
    print(f"# wrote {output}")
    print("# beta")
    print(result.beta)
    print("# orbital")
    print(result.orbital.ravel())


if __name__ == "__main__":
    main()

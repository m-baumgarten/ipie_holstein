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
"""Four-site Holstein model using free-projection Ito coherent-state walkers."""

from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.toyozawa_cs_unnormalized import (
    ToyozawaTrialUnnormalizedCoherentState,
)
from ipie.addons.free_projection.propagation.ito_second_order_fp import (
    ItoSymmSplitImportancePropagatorFP,
    ItoSymmSplitPropagatorFP,
)
from ipie.addons.free_projection.walkers.cs_walkers import EPhCSWalkersFP
from ipie.systems import Generic


@dataclass
class HolsteinItoFPInputs:
    wavefunction: np.ndarray
    beta_shift: np.ndarray
    electron_orbital: np.ndarray
    nelec: tuple
    nwalkers: int
    k: float
    t: float
    g: float
    w0: float
    nsites: int
    pbc: bool


def load_trial_wavefunction(filename: str, nsites: int) -> np.ndarray:
    data = np.load(filename)
    if "wavefunction" in data:
        wavefunction = data["wavefunction"]
    elif "beta" in data and "orbital" in data:
        wavefunction = np.column_stack([data["beta"], np.asarray(data["orbital"]).reshape(nsites)])
    else:
        raise ValueError("Trial file must contain 'wavefunction' or both 'beta' and 'orbital'.")

    wavefunction = np.asarray(wavefunction, dtype=np.complex128)
    if wavefunction.shape != (nsites, 2):
        raise ValueError(f"Expected trial wavefunction shape {(nsites, 2)}, got {wavefunction.shape}.")
    return wavefunction


def build_holstein_ito_fp_inputs(
    nsites: int = 4,
    t: float = 1.0,
    g: float = 1.0,
    w0: float = 1.0,
    nwalkers: int = 64,
    k: float = 0.0,
    pbc: bool = True,
    beta_scale: float = 1.0,
    wavefunction: np.ndarray = None,
) -> HolsteinItoFPInputs:
    """Build a deterministic Toyozawa/coherent-state FP walker input."""
    nelec = (1, 0)

    if wavefunction is None:
        distance = np.minimum(np.arange(nsites), nsites - np.arange(nsites))
        beta_shift = -beta_scale * g / w0 * np.exp(-distance)
        electron_orbital = np.exp(-0.75 * distance).astype(np.complex128)
        electron_orbital /= np.linalg.norm(electron_orbital)
        wavefunction = np.column_stack([beta_shift, electron_orbital])
    else:
        wavefunction = np.asarray(wavefunction, dtype=np.complex128)
        if wavefunction.shape != (nsites, 2):
            raise ValueError(f"Expected trial wavefunction shape {(nsites, 2)}, got {wavefunction.shape}.")
        beta_shift = wavefunction[:, 0]
        electron_orbital = wavefunction[:, 1]

    return HolsteinItoFPInputs(
        wavefunction=wavefunction.astype(np.complex128),
        beta_shift=beta_shift.astype(np.complex128),
        electron_orbital=electron_orbital.astype(np.complex128)[:, None],
        nelec=nelec,
        nwalkers=nwalkers,
        k=k,
        t=t,
        g=g,
        w0=w0,
        nsites=nsites,
        pbc=pbc,
    )


def build_workflow(
    inputs: HolsteinItoFPInputs,
    timestep: float = 1.0e-4,
    reference_energy: float = None,
    importance_sampling: bool = False,
    force_bias: str = "overlap",
    split_gauge: str = "static",
    split_gauge_scale: float = 1.0,
    split_gauge_max_norm: float = None,
    mpi_handler=None,
):
    """Construct the free-projection Ito coherent-state Holstein pipeline."""
    system = Generic(inputs.nelec)
    ham = HolsteinModel(
        g=inputs.g,
        t=inputs.t,
        w0=inputs.w0,
        nsites=inputs.nsites,
        pbc=inputs.pbc,
    )
    ham.build()

    trial = ToyozawaTrialUnnormalizedCoherentState(
        wavefunction=inputs.wavefunction,
        w0=ham.w0,
        num_elec=inputs.nelec,
        num_basis=inputs.nsites,
        K=inputs.k,
    )
    trial.set_etrial(ham)

    walkers = EPhCSWalkersFP(
        initial_walker=inputs.wavefunction,
        nup=inputs.nelec[0],
        ndown=inputs.nelec[1],
        nbasis=inputs.nsites,
        nwalkers=inputs.nwalkers,
        mpi_handler=mpi_handler,
    )
    walkers.build(trial)

    if importance_sampling:
        propagator = ItoSymmSplitImportancePropagatorFP(
            time_step=timestep,
            mean_field_subtraction=True,
            reference_energy=reference_energy,
            force_bias=force_bias,
            split_gauge=split_gauge,
            split_gauge_scale=split_gauge_scale,
            split_gauge_max_norm=split_gauge_max_norm,
        )
    else:
        propagator = ItoSymmSplitPropagatorFP(
            time_step=timestep,
            mean_field_subtraction=True,
            reference_energy=reference_energy,
        )
    propagator.build(ham, trial, walkers, mpi_handler=mpi_handler)
    return system, ham, trial, walkers, propagator

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
"""Four-site Holstein model using coherent-state walkers."""

from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.propagation.cs_propagator import CoherentStatePropagator
from ipie.addons.eph.trial_wavefunction.toyozawa_cs import ToyozawaTrialCoherentState
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers
from ipie.systems import Generic


@dataclass
class HolsteinCSInputs:
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


def build_holstein_cs_inputs(
    nsites: int = 4,
    t: float = 1.0,
    g: float = 1.0,
    w0: float = 1.0,
    nwalkers: int = 200,
    k: float = 0.0,
    pbc: bool = True,
    beta_scale: float = 1.0,
    wavefunction: np.ndarray = None,
) -> HolsteinCSInputs:
    """Build a small Toyozawa/coherent-state walker input.

    The trial is intentionally deterministic and mildly localized around one
    site; the Toyozawa momentum projection restores translation symmetry.
    """
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

    return HolsteinCSInputs(
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


def build_workflow(inputs: HolsteinCSInputs, timestep: float = 0.005):
    """Construct the coherent-state walker pipeline for the 4-site Holstein model."""
    system = Generic(inputs.nelec)
    ham = HolsteinModel(
        g=inputs.g,
        t=inputs.t,
        w0=inputs.w0,
        nsites=inputs.nsites,
        pbc=inputs.pbc,
    )
    ham.build()

    trial = ToyozawaTrialCoherentState(
        wavefunction=inputs.wavefunction,
        w0=ham.w0,
        num_elec=inputs.nelec,
        num_basis=inputs.nsites,
        K=inputs.k,
    )
    trial.set_etrial(ham)

    walkers = EPhCSWalkers(
        initial_walker=inputs.wavefunction,
        nup=inputs.nelec[0],
        ndown=inputs.nelec[1],
        nbasis=inputs.nsites,
        nwalkers=inputs.nwalkers,
    )
    walkers.build(trial)

    propagator = CoherentStatePropagator(time_step=timestep)
    propagator.build(ham, trial, walkers)
    return system, ham, trial, walkers, propagator

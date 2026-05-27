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
"""Four-site Holstein model represented through the ab-initio EPh API."""

from dataclasses import dataclass

import numpy as np

from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.propagation.abinitio import AbInitioEPhPropagator
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial
from ipie.addons.eph.walkers.abinitio import AbInitioEPhWalkers
from ipie.systems import Generic


@dataclass
class HolsteinAbInitioInputs:
    eps_kj: np.ndarray
    g_qnu_kmn: np.ndarray
    omega_qnu: np.ndarray
    psi_kj: np.ndarray
    beta_qnu: np.ndarray
    initial_phi_kj: np.ndarray
    initial_X_qnu: np.ndarray


def build_holstein_inputs(
    nsites: int = 4,
    t: float = 1.0,
    g: float = 1.0,
    w0: float = 1.0,
    nwalkers: int = 200,
    seed: int = 7,
    electron_noise: float = 0.02,
    phonon_noise: float = 1.0e-4,
    beta_scale: float = 1.0,
) -> HolsteinAbInitioInputs:
    """Create k-space Holstein tensors with one electron and one phonon band.

    The ab-initio Hamiltonian consumes ``g_qnu_kmn`` directly, so the Holstein
    Fourier normalization is included here as ``g / sqrt(nsites)``.
    """
    rng = np.random.default_rng(seed)

    k = 2.0 * np.pi * np.arange(nsites) / nsites
    eps_kj = (-2.0 * t * np.cos(k))[:, None]
    omega_qnu = w0 * np.ones((nsites, 1), dtype=np.float64)
    g_qnu_kmn = (g / np.sqrt(nsites)) * np.ones(
        (nsites, 1, nsites, 1, 1), dtype=np.complex128
    )

    psi_kj = np.ones((nsites, 1), dtype=np.complex128) / np.sqrt(nsites)
    beta_qnu = -beta_scale * g / (np.sqrt(nsites) * w0) * np.ones(
        (nsites, 1), dtype=np.complex128
    )

    initial_phi_kj = psi_kj[None, :, :] + electron_noise * (
        rng.normal(size=(nwalkers, nsites, 1))
        + 1.0j * rng.normal(size=(nwalkers, nsites, 1))
    )
    initial_phi_kj /= np.linalg.norm(initial_phi_kj.reshape(nwalkers, -1), axis=1)[
        :, None, None
    ]

    initial_X_qnu = phonon_noise * (
        rng.normal(size=(nwalkers, nsites, 1))
        + 1.0j * rng.normal(size=(nwalkers, nsites, 1))
    )
    enforce_reality_constraint(initial_X_qnu)

    return HolsteinAbInitioInputs(
        eps_kj=eps_kj,
        g_qnu_kmn=g_qnu_kmn,
        omega_qnu=omega_qnu,
        psi_kj=psi_kj,
        beta_qnu=beta_qnu,
        initial_phi_kj=initial_phi_kj,
        initial_X_qnu=initial_X_qnu,
    )


def enforce_reality_constraint(X_qnu: np.ndarray) -> None:
    """Project full-BZ phonon coordinates onto ``X[-q] = X[q].conj()``."""
    nq = X_qnu.shape[-2]
    minus_q = (-np.arange(nq, dtype=np.int64)) % nq
    for iq, imq in enumerate(minus_q):
        if iq > imq:
            continue
        if iq == imq:
            X_qnu[..., iq, :] = X_qnu[..., iq, :].real
            continue
        x_pair = 0.5 * (X_qnu[..., iq, :] + X_qnu[..., imq, :].conj())
        X_qnu[..., iq, :] = x_pair
        X_qnu[..., imq, :] = x_pair.conj()


def build_workflow(
    inputs: HolsteinAbInitioInputs,
    timestep: float = 0.005,
):
    """Construct the ab-initio pipeline for the four-site Holstein model."""
    system = Generic((1, 0))
    ham = AbInitioEPhHamiltonian(
        eps_kj=inputs.eps_kj,
        g_qnu_kmn=inputs.g_qnu_kmn,
        omega_qnu=inputs.omega_qnu,
    )
    trial = AbInitioDD2Trial(inputs.psi_kj, inputs.beta_qnu, K=0.0).build(ham)
    walkers = AbInitioEPhWalkers(
        inputs.initial_phi_kj,
        inputs.initial_X_qnu,
        nwalkers=inputs.initial_phi_kj.shape[0],
    )
    walkers.build(trial)

    propagator = AbInitioEPhPropagator(
        time_step=timestep,
        electron_step="split",
        enforce_reality_constraint=True,
    )
    propagator.build(ham, trial, walkers)
    return system, ham, trial, walkers, propagator

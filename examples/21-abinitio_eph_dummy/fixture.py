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
"""Shared fixture loading for the ab-initio EPh dD2 example."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.propagation.abinitio import AbInitioEPhPropagator
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial
from ipie.addons.eph.walkers.abinitio import AbInitioEPhWalkers
from ipie.systems import Generic


@dataclass
class AbInitioFixtureInputs:
    eps_kj: np.ndarray
    g_qnu_kmn: np.ndarray
    omega_qnu: np.ndarray
    psi_kj: np.ndarray
    beta_qnu: np.ndarray
    initial_phi_kj: np.ndarray
    initial_X_qnu: np.ndarray


def read_complex_dataset(handle, stem: str) -> np.ndarray:
    """Read datasets stored as ``<stem>_real`` and ``<stem>_imag``."""
    return handle[f"{stem}_real"][()] + 1.0j * handle[f"{stem}_imag"][()]


def reconstruct_full_g_from_factors(Uk, Vq, longrange, bloch) -> np.ndarray:
    r"""Reconstruct dense :math:`g^\nu_{mn}(k,q)` from low-rank factors."""
    g_orb = np.einsum("rijk,vrijq->qvkij", Uk, Vq, optimize=True)

    nq, nmode, nk, norb, _ = g_orb.shape
    eye_orb = np.eye(norb, dtype=np.complex128)
    g_orb += longrange[:, :, None, None, None] * eye_orb[None, None, None, :, :]

    if bloch.shape[0] != nk or bloch.shape[1] != norb:
        raise ValueError(
            "Uk_bloch must have shape (nk, norb, nband) for this example; "
            f"got {bloch.shape} with nk={nk}, norb={norb}."
        )

    nband = bloch.shape[2]
    g_qnu_kmn = np.zeros((nq, nmode, nk, nband, nband), dtype=np.complex128)
    for iq in range(nq):
        for ik in range(nk):
            ik_plus_q = (ik + iq) % nk
            left = bloch[ik_plus_q].conj().T
            right = bloch[ik]
            for inu in range(nmode):
                g_qnu_kmn[iq, inu, ik] = left @ g_orb[iq, inu, ik] @ right

    return g_qnu_kmn


def build_fixture_inputs(
    seed: int = 7,
    nwalkers: int = 200,
    frequency_floor: float = 1.0e-3,
    timestep_frequency_floor: float = 1.0e-3,
) -> AbInitioFixtureInputs:
    """Load the example fixture and reconstruct dense tensors for testing.

    Modes with frequencies below ``frequency_floor`` are treated as acoustic
    zero modes for the e-ph coupling: the SVD and long-range contributions are
    zeroed before dense reconstruction.  The Hamiltonian still receives a small
    positive frequency ``timestep_frequency_floor`` for those modes so the
    current coherent-coordinate implementation remains finite.
    """
    rng = np.random.default_rng(seed)
    here = Path(__file__).resolve().parent
    h5_file = here / "svd_kq.h5"

    with h5py.File(h5_file, "r") as handle:
        eps_kj = handle["bands"][()]
        omega_raw = handle["phfreq"][()]
        Uk = read_complex_dataset(handle, "Uk")
        Vq = read_complex_dataset(handle, "Vq")
        longrange = read_complex_dataset(handle, "longrange")
        bloch = read_complex_dataset(handle, "Uk_bloch")

    acoustic_mask = omega_raw < frequency_floor
    if np.any(acoustic_mask):
        for iq, inu in np.argwhere(acoustic_mask):
            Vq[inu, :, :, :, iq] = 0.0
            longrange[iq, inu] = 0.0

    omega_qnu = omega_raw.copy()
    omega_qnu[acoustic_mask] = timestep_frequency_floor
    g_qnu_kmn = reconstruct_full_g_from_factors(Uk, Vq, longrange, bloch)

    np.save("g_qnu_kmn.npy", g_qnu_kmn)
    np.save("omega_qnu.npy", omega_qnu)
    np.save("eps_kj.npy", eps_kj)

    psi_kj = np.conj(np.load(here / "electron_3.npy"))
    beta_qnu = np.load(here / "shift_3.npy")
    beta_qnu = beta_qnu.copy()
    beta_qnu[acoustic_mask] = 0.0

    nk, nband = eps_kj.shape
    nq, nmode = omega_qnu.shape

    if psi_kj.shape != (nk, nband):
        raise ValueError(f"electron_3.npy has shape {psi_kj.shape}, expected {(nk, nband)}.")
    if beta_qnu.shape != (nq, nmode):
        raise ValueError(f"shift_3.npy has shape {beta_qnu.shape}, expected {(nq, nmode)}.")

    psi_norm = np.linalg.norm(psi_kj)
    if psi_norm > 0.0:
        psi_kj = psi_kj / psi_norm

    initial_phi_kj = psi_kj[None, :, :] + 0.05 * (
        rng.normal(size=(nwalkers, nk, nband))
        + 1.0j * rng.normal(size=(nwalkers, nk, nband))
    )
    initial_phi_kj /= np.linalg.norm(initial_phi_kj.reshape(nwalkers, -1), axis=1)[
        :, None, None
    ]

    initial_X_qnu = 1.0e-4 * (
        rng.normal(size=(nwalkers, nq, nmode))
        + 1.0j * rng.normal(size=(nwalkers, nq, nmode))
    )
    initial_X_qnu[:, acoustic_mask] = 0.0

    return AbInitioFixtureInputs(
        eps_kj=eps_kj,
        g_qnu_kmn=g_qnu_kmn,
        omega_qnu=omega_qnu,
        psi_kj=psi_kj,
        beta_qnu=beta_qnu,
        initial_phi_kj=initial_phi_kj,
        initial_X_qnu=initial_X_qnu,
    )


def build_workflow(
    inputs: AbInitioFixtureInputs,
    timestep: float = 1.0e-7,
):
    """Construct the Hamiltonian, trial, walkers, propagator, and system."""
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
        enforce_reality_constraint=False,
    )
    propagator.build(ham, trial, walkers)
    return system, ham, trial, walkers, propagator

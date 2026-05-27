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

import numpy as np

from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial
from ipie.addons.eph.walkers.abinitio import AbInitioEPhWalkers


def local_energy_abinitio(
    system,
    hamiltonian: AbInitioEPhHamiltonian,
    walkers: AbInitioEPhWalkers,
    trial: AbInitioDD2Trial,
) -> np.ndarray:
    r"""Compute the dD2 local energy for ab-initio EPh walkers.

    Returns columns ``(total, electronic, electron-phonon, phonon)``.  The
    implementation consumes the compact walker caches ``S[k]``, ``M[q]``, and
    ``O`` and never forms a three-momentum intermediate.
    """
    trial.calc_overlap(walkers)

    energy = np.zeros((walkers.nwalkers, 4), dtype=np.complex128)
    energy[:, 1] = electronic_local_energy(hamiltonian, walkers, trial)
    energy[:, 2] = electron_phonon_local_energy(hamiltonian, walkers, trial)
    energy[:, 3] = phonon_local_energy(hamiltonian, walkers, trial)
    energy[:, 0] = np.sum(energy[:, 1:], axis=1)
    return energy


def electronic_local_energy(
    hamiltonian: AbInitioEPhHamiltonian,
    walkers: AbInitioEPhWalkers,
    trial: AbInitioDD2Trial,
) -> np.ndarray:
    r"""Return :math:`\langle\Psi_T|H_{\rm el}|W\rangle/\langle\Psi_T|W\rangle`."""
    numerator = np.einsum(
        "wk,kj,kj,wkj->w",
        walkers.S,
        hamiltonian.eps_kj,
        trial.psi_kj.conj(),
        walkers.phi_kj,
        optimize=True,
    )
    return numerator / walkers.O + hamiltonian.ecore


def electron_phonon_local_energy(
    hamiltonian: AbInitioEPhHamiltonian,
    walkers: AbInitioEPhWalkers,
    trial: AbInitioDD2Trial,
) -> np.ndarray:
    r"""Return the electron-phonon local energy using the ``S[k+q]`` moment."""
    k_plus_q = hamiltonian.k_plus_q.T
    S_k_plus_q = np.take(walkers.S, k_plus_q, axis=1)
    psi_k_plus_q = np.take(trial.psi_kj, k_plus_q, axis=0).conj()
    mode_factor = np.sqrt(2.0 * hamiltonian.omega_qnu)[None, :, :] * walkers.X_qnu

    numerator = np.einsum(
        "wqk,qvkmn,qkm,wkn,wqv->w",
        S_k_plus_q,
        hamiltonian.g_qnu_kmn,
        psi_k_plus_q,
        walkers.phi_kj,
        mode_factor,
        optimize=True,
    )
    return numerator / walkers.O


def phonon_local_energy(
    hamiltonian: AbInitioEPhHamiltonian,
    walkers: AbInitioEPhWalkers,
    trial: AbInitioDD2Trial,
) -> np.ndarray:
    r"""Return the phonon local energy using the ``M[q]`` moment."""
    omega = hamiltonian.omega_qnu
    x0 = trial.x0_qnu
    p0 = trial.p0_qnu

    linear_coeff = (
        omega[None, :, :] ** 2 * walkers.X_qnu.conj() * x0[None, :, :]
        - 1.0j * omega[None, :, :] * walkers.X_qnu.conj() * p0[None, :, :]
    )
    center_coeff = (
        -0.5 * omega**2 * np.abs(x0) ** 2
        + 0.5 * np.abs(p0) ** 2
        + 1.0j * omega * x0.conj() * p0
    )

    numerator = np.einsum("wqv,wq->w", linear_coeff, walkers.M, optimize=True)
    numerator += walkers.O * np.sum(center_coeff)
    return numerator / walkers.O

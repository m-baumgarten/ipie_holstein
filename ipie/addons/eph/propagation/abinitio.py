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

import time

import numpy as np

from ipie.addons.eph.estimators.local_energy_abinitio import phonon_local_energy
from ipie.propagation.continuous_base import PropagatorTimer


class AbInitioEPhPropagator:
    r"""Trotter propagator for ab-initio electron-phonon walkers.

    The default electronic update uses the symmetric one-body split

    .. math::
        e^{-\Delta\tau H_{\rm el}/2}
        e^{-\Delta\tau H_{\rm elph}(X)}
        e^{-\Delta\tau H_{\rm el}/2},

    with an optional dense combined mode
    :math:`e^{-\Delta\tau(H_{\rm el}+H_{\rm elph}(X))}` for debugging the
    electronic Trotter error.
    """

    def __init__(
        self,
        time_step: float,
        electron_step: str = "split",
        phaseless: bool = True,
        real_phonon_drift: bool = True,
        complex_phonon_noise: bool = False,
        enforce_reality_constraint: bool = False,
        energy_offset: float = None,
        normalize_electron: bool = True,
        verbose: bool = False,
    ) -> None:
        if electron_step not in ("split", "combined"):
            raise ValueError("electron_step must be 'split' or 'combined'.")

        self.dt = time_step
        self.dt_ph = 0.5 * time_step
        self.electron_step = electron_step
        self.phaseless = phaseless
        self.real_phonon_drift = real_phonon_drift
        self.complex_phonon_noise = complex_phonon_noise
        self.enforce_reality_constraint = enforce_reality_constraint
        self.energy_offset = energy_offset
        self.normalize_electron = normalize_electron
        self.verbose = verbose
        self.timer = PropagatorTimer()
        self.mpi_handler = None

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None) -> None:
        """Cache the diagonal electronic propagator factors."""
        self.mpi_handler = mpi_handler
        if self.energy_offset is None:
            self.energy_offset = float(np.min(hamiltonian.eps_kj))

        shifted_eps = hamiltonian.eps_kj - self.energy_offset
        self.expH1_half = np.exp(-0.5 * self.dt * shifted_eps)
        self.H1_diag = shifted_eps.reshape(hamiltonian.nbasis)

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift: float = 0.0) -> None:
        """Apply phonon/electron/phonon Trotter substeps to the walker batch."""
        weight_old = walkers.weight.copy()
        self.propagate_phonons(walkers, hamiltonian, trial, self.dt_ph, eshift=eshift)

        ovlp_old = trial.calc_overlap(walkers).copy()
        self.propagate_electron(walkers, hamiltonian, trial)
        electron_norm = self.normalize_electronic_walkers(walkers)
        ovlp_new = trial.calc_overlap(walkers).copy() * electron_norm
        self.update_weight_overlap(walkers, ovlp_old, ovlp_new)

        self.propagate_phonons(walkers, hamiltonian, trial, self.dt_ph, eshift=eshift)
        walkers.ovlp = trial.calc_overlap(walkers)
        self.update_hybrid_energy(walkers, weight_old, eshift)

    def propagate_electron(self, walkers, hamiltonian, trial) -> None:
        """Propagate the electronic amplitudes at fixed phonon coordinates."""
        start_time = time.time()
        if self.electron_step == "split":
            self._propagate_electron_split(walkers, hamiltonian)
        else:
            self._propagate_electron_combined(walkers, hamiltonian)
        self.timer.tgemm += time.time() - start_time

    def _propagate_electron_split(self, walkers, hamiltonian) -> None:
        scipy_linalg = self._scipy_linalg()
        walkers.phi_kj *= self.expH1_half[None, :, :]

        phi_flat = walkers.phi_kj.reshape(walkers.nwalkers, hamiltonian.nbasis)
        for iw in range(walkers.nwalkers):
            exp_eph = scipy_linalg.expm(
                -self.dt * hamiltonian.construct_eph_matrix(walkers.X_qnu[iw])
            )
            phi_flat[iw] = exp_eph @ phi_flat[iw]

        walkers.phi_kj *= self.expH1_half[None, :, :]
        walkers.phia = walkers.phi_kj.reshape(walkers.nwalkers, walkers.nbasis, 1)

    def _propagate_electron_combined(self, walkers, hamiltonian) -> None:
        scipy_linalg = self._scipy_linalg()
        phi_flat = walkers.phi_kj.reshape(walkers.nwalkers, hamiltonian.nbasis)
        h1 = np.diag(self.H1_diag).astype(np.complex128)

        for iw in range(walkers.nwalkers):
            h_eff = h1 + hamiltonian.construct_eph_matrix(walkers.X_qnu[iw])
            phi_flat[iw] = scipy_linalg.expm(-self.dt * h_eff) @ phi_flat[iw]

        walkers.phia = walkers.phi_kj.reshape(walkers.nwalkers, walkers.nbasis, 1)

    def propagate_phonons(
        self,
        walkers,
        hamiltonian,
        trial,
        dt_step: float = None,
        eshift: float = 0.0,
    ) -> None:
        """Importance-sampled phonon drift-diffusion and branching step."""
        start_time = time.time()
        dt_step = self.dt_ph if dt_step is None else dt_step

        e_old = self._phonon_branch_energy(hamiltonian, walkers, trial)
        drift = trial.calc_phonon_gradient(walkers)
        if self.real_phonon_drift:
            drift = drift.real.astype(np.complex128)

        walkers.X_qnu += dt_step * drift + self._sample_phonon_noise(walkers, dt_step)
        if self.enforce_reality_constraint:
            self._project_reality_constraint(walkers, trial)

        e_new = self._phonon_branch_energy(hamiltonian, walkers, trial)
        walkers.weight *= np.exp(-0.5 * dt_step * (e_old + e_new) + dt_step * eshift)
        walkers.phonon_disp = walkers.X_qnu
        self.timer.tgemm += time.time() - start_time

    def update_weight_overlap(self, walkers, ovlp_old, ovlp_new) -> None:
        """Apply the importance-sampling overlap ratio for a deterministic step."""
        ratio = ovlp_new / ovlp_old
        if not self.phaseless:
            walkers.weight *= ratio
            return

        phase = np.angle(ratio)
        projection = np.maximum(0.0, np.cos(phase))
        walkers.weight *= np.abs(ratio) * projection

    def normalize_electronic_walkers(self, walkers) -> np.ndarray:
        """Normalize electronic amplitudes and return the removed norms."""
        if not self.normalize_electron:
            return np.ones(walkers.nwalkers, dtype=np.float64)

        flat = walkers.phi_kj.reshape(walkers.nwalkers, walkers.nbasis)
        norms = np.linalg.norm(flat, axis=1)
        bad = (~np.isfinite(norms)) | (norms <= np.finfo(np.float64).tiny)
        if np.any(bad):
            raise ValueError(
                "Electronic walker norm underflowed during ab-initio propagation. "
                "Try a smaller timestep or a larger electronic energy_offset."
            )

        flat /= norms[:, None]
        walkers.phia = walkers.phi_kj.reshape(walkers.nwalkers, walkers.nbasis, 1)
        return norms

    def update_hybrid_energy(self, walkers, weight_old, eshift: float) -> None:
        """Store the residual hybrid energy used by ipie's adaptive shift."""
        old_abs = np.abs(weight_old)
        new_abs = np.abs(walkers.weight)
        hybrid_energy = np.full(walkers.nwalkers, eshift, dtype=np.complex128)

        tiny = np.finfo(np.float64).tiny
        active = (
            np.isfinite(old_abs)
            & np.isfinite(new_abs)
            & (old_abs > tiny)
            & (new_abs > tiny)
        )
        factor = new_abs[active] / old_abs[active]
        hybrid_energy[active] = eshift - np.log(factor) / self.dt
        walkers.hybrid_energy = hybrid_energy

    def _phonon_branch_energy(self, hamiltonian, walkers, trial) -> np.ndarray:
        trial.calc_overlap(walkers)
        return phonon_local_energy(hamiltonian, walkers, trial).real

    def _sample_phonon_noise(self, walkers, dt_step: float) -> np.ndarray:
        shape = walkers.X_qnu.shape
        if self.complex_phonon_noise:
            scale = np.sqrt(0.5 * dt_step)
            return scale * (
                np.random.normal(size=shape) + 1.0j * np.random.normal(size=shape)
            )
        return np.random.normal(scale=np.sqrt(dt_step), size=shape).astype(np.complex128)

    @staticmethod
    def _project_reality_constraint(walkers, trial) -> None:
        minus_q = trial.minus_q
        for iq, imq in enumerate(minus_q):
            if iq > imq:
                continue
            if iq == imq:
                walkers.X_qnu[:, iq, :] = walkers.X_qnu[:, iq, :].real
                continue
            x_pair = 0.5 * (
                walkers.X_qnu[:, iq, :] + walkers.X_qnu[:, imq, :].conj()
            )
            walkers.X_qnu[:, iq, :] = x_pair
            walkers.X_qnu[:, imq, :] = x_pair.conj()

    @staticmethod
    def _scipy_linalg():
        import scipy.linalg

        return scipy.linalg


AbInitioDD2Propagator = AbInitioEPhPropagator

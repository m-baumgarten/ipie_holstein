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

from ipie.walkers.base_walkers import BaseWalkers


class AbInitioEPhWalkers(BaseWalkers):
    r"""Single-polaron walkers for band-basis ab-initio EPh calculations.

    The walker state is the single-particle electronic amplitude
    ``phi_kj`` and the phonon position coordinates ``X_qnu``.  The dD2 overlap
    contractions are cached in the compact translation moments

    ``A_el[R]``, ``A_ph[R]``, ``T[R]``, ``O``, ``S[k]``, and ``M[q]``.

    No object with three momentum indices is stored.
    """

    def __init__(
        self,
        initial_electron: np.ndarray,
        initial_phonon: np.ndarray,
        nwalkers: int,
        verbose: bool = False,
    ) -> None:
        self.phi_kj = self._as_walker_batch(initial_electron, nwalkers, "initial_electron")
        self.X_qnu = self._as_walker_batch(initial_phonon, nwalkers, "initial_phonon")

        self.nwalkers = nwalkers
        self.nk, self.nband = self.phi_kj.shape[1:]
        self.nq, self.nmode = self.X_qnu.shape[1:]
        self.nbasis = self.nk * self.nband
        self.nphonon_modes = self.nq * self.nmode
        self.nup = 1
        self.ndown = 0

        super().__init__(nwalkers, verbose=verbose)

        self.weight = np.ones(self.nwalkers, dtype=np.complex128)
        self.phia = self.phi_kj.reshape(self.nwalkers, self.nbasis, 1)
        self.phib = np.zeros((self.nwalkers, self.nbasis, 0), dtype=np.complex128)
        self.phonon_disp = self.X_qnu

        self.A_el = None
        self.A_ph = None
        self.T = None
        self.O = None
        self.S = None
        self.M = None

        self.buff_names += ["phi_kj", "X_qnu"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = np.zeros(self.buff_size, dtype=np.complex128)

    @staticmethod
    def _as_walker_batch(array: np.ndarray, nwalkers: int, name: str) -> np.ndarray:
        array = np.asarray(array, dtype=np.complex128)
        if array.ndim == 2:
            return np.repeat(array[None, :, :], nwalkers, axis=0).copy()
        if array.ndim == 3 and array.shape[0] == nwalkers:
            return array.copy()
        raise ValueError(
            f"{name} must have shape (n0, n1) or ({nwalkers}, n0, n1); "
            f"got {array.shape}."
        )

    def build(self, trial) -> None:
        """Allocate dD2 contraction caches and compute the initial overlap."""
        self._validate_trial(trial)
        self._allocate_cache(trial)
        self.ovlp = self.update_overlap(trial)

    def _allocate_cache(self, trial) -> None:
        self.A_el = np.zeros((self.nwalkers, trial.nR), dtype=np.complex128)
        self.A_ph = np.zeros_like(self.A_el)
        self.T = np.zeros_like(self.A_el)
        self.O = np.zeros(self.nwalkers, dtype=np.complex128)
        self.S = np.zeros((self.nwalkers, trial.nk), dtype=np.complex128)
        self.M = np.zeros((self.nwalkers, trial.nq), dtype=np.complex128)

    def update_overlap(self, trial) -> np.ndarray:
        """Refresh all compact dD2 overlap contractions."""
        self._validate_trial(trial)
        self._require_cache(trial)
        self.update_electronic_overlap(trial)
        self.update_phonon_overlap(trial)
        self.update_translation_moments(trial)
        self.ovlp = self.O.copy()
        return self.ovlp

    def update_electronic_overlap(self, trial) -> np.ndarray:
        r"""Compute ``A_el[R]`` from the walker and trial amplitudes."""
        self._validate_trial(trial)
        self._require_cache(trial)
        B_k = np.einsum("kj,wkj->wk", trial.psi_kj.conj(), self.phi_kj, optimize=True)
        self.A_el[:, :] = B_k @ trial.phase_kR
        return self.A_el

    def update_phonon_overlap(self, trial) -> np.ndarray:
        r"""Compute ``A_ph[R]`` using expanded translated coherent centers."""
        self._validate_trial(trial)
        self._require_cache(trial)
        self._require_trial_centers(trial)

        omega = trial.omega_qnu
        x0 = trial.x0_qnu
        p0 = trial.p0_qnu
        X = self.X_qnu

        log_norm = 0.25 * np.sum(np.log(omega / np.pi))
        const = (
            log_norm
            - 0.5 * np.einsum("qv,wqv->w", omega, np.abs(X) ** 2, optimize=True)
            - 0.5 * np.sum(omega * np.abs(x0) ** 2)
            + 0.5j * np.sum(x0.conj() * p0)
        )
        coeff_minus = 0.5 * omega[None, :, :] * X.conj() * x0[None, :, :]
        coeff_plus = (
            0.5 * omega[None, :, :] * X * x0.conj()[None, :, :]
            - 1.0j * p0.conj()[None, :, :] * X
        )
        log_A_ph = const[:, None]
        log_A_ph = log_A_ph + np.einsum(
            "wqv,qR->wR", coeff_minus, trial.phase_qR.conj(), optimize=True
        )
        log_A_ph = log_A_ph + np.einsum(
            "wqv,qR->wR", coeff_plus, trial.phase_qR, optimize=True
        )
        self.A_ph[:, :] = np.exp(log_A_ph)
        return self.A_ph

    def update_translation_moments(self, trial) -> np.ndarray:
        """Compute ``T[R]``, ``O``, ``S[k]``, and ``M[q]``."""
        self._validate_trial(trial)
        self._require_cache(trial)
        self.T[:, :] = self.A_el * self.A_ph * trial.phase_KR[None, :]
        self.O[:] = np.sum(self.T, axis=1)
        self.S[:, :] = np.einsum(
            "wR,R,kR->wk", self.A_ph, trial.phase_KR, trial.phase_kR, optimize=True
        )
        self.M[:, :] = np.einsum("wR,qR->wq", self.T, trial.phase_qR.conj(), optimize=True)
        return self.O

    def phonon_log_gradient(self, trial) -> np.ndarray:
        r"""Return the dD2 phonon log derivative for each walker and mode."""
        self.update_overlap(trial)
        prefactor = trial.omega_qnu * trial.x0_qnu - 1.0j * trial.p0_qnu
        return -trial.omega_qnu[None, :, :] * self.X_qnu + (
            self.M[:, :, None] / self.O[:, None, None]
        ) * prefactor[None, :, :]

    def set_electron_amplitudes(self, phi_kj: np.ndarray) -> None:
        """Replace the electronic amplitudes and refresh the flattened view."""
        self.phi_kj = self._as_walker_batch(phi_kj, self.nwalkers, "phi_kj")
        if self.phi_kj.shape[1:] != (self.nk, self.nband):
            raise ValueError(f"phi_kj must have final axes {(self.nk, self.nband)}.")
        self.phia = self.phi_kj.reshape(self.nwalkers, self.nbasis, 1)

    def set_phonon_coordinates(self, X_qnu: np.ndarray) -> None:
        """Replace the phonon coordinates and refresh compatibility aliases."""
        self.X_qnu = self._as_walker_batch(X_qnu, self.nwalkers, "X_qnu")
        if self.X_qnu.shape[1:] != (self.nq, self.nmode):
            raise ValueError(f"X_qnu must have final axes {(self.nq, self.nmode)}.")
        self.phonon_disp = self.X_qnu

    def cast_to_cupy(self, verbose=False):
        from ipie.utils.backend import cast_to_device

        cast_to_device(self, verbose)

    def reortho(self):
        """Normalize each single-particle electronic walker."""
        detR = []
        flat = self.phi_kj.reshape(self.nwalkers, self.nbasis)
        for iw in range(self.nwalkers):
            norm = np.linalg.norm(flat[iw])
            if norm == 0.0:
                raise ValueError("Cannot reorthogonalize a zero electronic walker.")

            flat[iw] /= norm
            detR.append(norm)
            self.detR[iw] = norm
            self.log_detR[iw] += np.log(norm)

            self.ovlp[iw] /= norm
            if self.A_el is not None:
                self.A_el[iw] /= norm
                self.T[iw] /= norm
                self.O[iw] /= norm
                self.M[iw] /= norm

        self.phia = self.phi_kj.reshape(self.nwalkers, self.nbasis, 1)
        return detR

    def reortho_batched(self):
        """GPU placeholder; single-particle normalization is cheap on CPU."""
        return self.reortho()

    def _validate_trial(self, trial) -> None:
        if trial.psi_kj.shape != (self.nk, self.nband):
            raise ValueError(
                f"trial.psi_kj shape {trial.psi_kj.shape} does not match "
                f"walker electron shape {(self.nk, self.nband)}."
            )
        if trial.beta_qnu.shape != (self.nq, self.nmode):
            raise ValueError(
                f"trial.beta_qnu shape {trial.beta_qnu.shape} does not match "
                f"walker phonon shape {(self.nq, self.nmode)}."
            )

    def _require_cache(self, trial) -> None:
        if self.A_el is None:
            self._allocate_cache(trial)

    @staticmethod
    def _require_trial_centers(trial) -> None:
        if trial.omega_qnu is None or trial.x0_qnu is None or trial.p0_qnu is None:
            raise RuntimeError("Call trial.build(hamiltonian) before walker overlap updates.")


AbInitioDD2Walkers = AbInitioEPhWalkers

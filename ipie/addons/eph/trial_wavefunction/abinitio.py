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

from typing import Optional, Tuple

import numpy as np


class AbInitioDD2Trial:
    r"""Band-basis dD2 trial data for position-space phonon walkers.

    This class is deliberately a storage and convention object, not a
    variational optimizer.  The user supplies the electronic amplitudes
    ``psi_kj`` and coherent-state amplitudes ``beta_qnu``.  The Hamiltonian can
    be attached later with :meth:`build`, which validates the shapes and computes
    the coherent-state displacement and momentum centers.

    Parameters
    ----------
    psi_kj
        Electronic amplitudes with shape ``(nk, nband)``.
    beta_qnu
        Coherent-state amplitudes with shape ``(nq, nmode)``.
    K
        Total crystal momentum used in the dD2 projector.  If ``phase_KR`` is
        not supplied, ``K`` is interpreted as a scalar and used with the
        one-dimensional default translation list.
    translations
        Optional translation labels.  With the default phase convention these
        are scalar cell indices.
    phase_kR
        Optional array with shape ``(nk, nR)`` containing
        :math:`\exp(i\mathbf{k}\cdot\mathbf{R})`.
    phase_qR
        Optional array with shape ``(nq, nR)`` containing
        :math:`\exp(i\mathbf{q}\cdot\mathbf{R})`.
    phase_KR
        Optional array with shape ``(nR,)`` containing
        :math:`\exp(-i\mathbf{K}\cdot\mathbf{R})`.
    minus_q
        Optional lookup array where ``minus_q[q]`` is the index of
        :math:`-\mathbf{q}`. If omitted and ``nq`` points are present, modular
        one-dimensional indexing is used.
    """

    def __init__(
        self,
        psi_kj: np.ndarray,
        beta_qnu: np.ndarray,
        K: float = 0.0,
        translations: Optional[np.ndarray] = None,
        phase_kR: Optional[np.ndarray] = None,
        phase_qR: Optional[np.ndarray] = None,
        phase_KR: Optional[np.ndarray] = None,
        minus_q: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        self.name = "AbInitioDD2"
        self.verbose = verbose
        self.K = K

        self.psi_kj = np.asarray(psi_kj, dtype=np.complex128)
        self.beta_qnu = np.asarray(beta_qnu, dtype=np.complex128)
        self._validate_trial_shapes()

        self.nk, self.nband = self.psi_kj.shape
        self.nq, self.nmode = self.beta_qnu.shape
        self.nbasis = self.nk * self.nband
        self.nphonon_modes = self.nq * self.nmode

        self.minus_q = self._build_minus_q(minus_q)
        self.translations = self._build_translations(
            translations, phase_kR, phase_qR, phase_KR
        )
        self.nR = len(self.translations)

        self.phase_kR = self._build_phase_kR(phase_kR)
        self.phase_qR = self._build_phase_qR(phase_qR)
        self.phase_KR = self._build_phase_KR(phase_KR)

        # Short aliases matching the notation in the derivation notes.
        self.psi = self.psi_kj
        self.beta = self.beta_qnu

        self.omega_qnu = None
        self.x0_qnu = None
        self.p0_qnu = None
        self.hamiltonian = None

        # Compatibility-style names used by existing EPh code paths.
        self.nelec = (1, 0)
        self.nup = 1
        self.ndown = 0
        self.optimized = True
        self.compute_trial_energy = False
        self.energy = None

    def _validate_trial_shapes(self) -> None:
        if self.psi_kj.ndim != 2:
            raise ValueError("psi_kj must have shape (nk, nband).")
        if self.beta_qnu.ndim != 2:
            raise ValueError("beta_qnu must have shape (nq, nmode).")

    def _build_minus_q(self, minus_q: Optional[np.ndarray]) -> np.ndarray:
        if minus_q is None:
            return (-np.arange(self.beta_qnu.shape[0], dtype=np.int64)) % self.beta_qnu.shape[0]

        minus_q = np.asarray(minus_q, dtype=np.int64)
        if minus_q.shape != (self.beta_qnu.shape[0],):
            raise ValueError(
                f"minus_q must have shape {(self.beta_qnu.shape[0],)}, "
                f"got {minus_q.shape}."
            )
        if np.any(minus_q < 0) or np.any(minus_q >= self.beta_qnu.shape[0]):
            raise ValueError("minus_q entries must be valid q-point indices.")
        return minus_q

    def _build_translations(
        self,
        translations: Optional[np.ndarray],
        phase_kR: Optional[np.ndarray],
        phase_qR: Optional[np.ndarray],
        phase_KR: Optional[np.ndarray],
    ) -> np.ndarray:
        if translations is not None:
            return np.asarray(translations)

        if phase_kR is not None:
            return np.arange(np.asarray(phase_kR).shape[1])
        if phase_qR is not None:
            return np.arange(np.asarray(phase_qR).shape[1])
        if phase_KR is not None:
            return np.arange(np.asarray(phase_KR).shape[0])

        return np.arange(self.psi_kj.shape[0])

    def _build_phase_kR(self, phase_kR: Optional[np.ndarray]) -> np.ndarray:
        if phase_kR is None:
            k = np.arange(self.nk)[:, None]
            R = np.asarray(self.translations)[None, :]
            return np.exp(2.0j * np.pi * k * R / self.nk)

        phase_kR = np.asarray(phase_kR, dtype=np.complex128)
        if phase_kR.shape != (self.nk, self.nR):
            raise ValueError(
                f"phase_kR must have shape {(self.nk, self.nR)}, got {phase_kR.shape}."
            )
        return phase_kR

    def _build_phase_qR(self, phase_qR: Optional[np.ndarray]) -> np.ndarray:
        if phase_qR is None:
            q = np.arange(self.nq)[:, None]
            R = np.asarray(self.translations)[None, :]
            return np.exp(2.0j * np.pi * q * R / self.nq)

        phase_qR = np.asarray(phase_qR, dtype=np.complex128)
        if phase_qR.shape != (self.nq, self.nR):
            raise ValueError(
                f"phase_qR must have shape {(self.nq, self.nR)}, got {phase_qR.shape}."
            )
        return phase_qR

    def _build_phase_KR(self, phase_KR: Optional[np.ndarray]) -> np.ndarray:
        if phase_KR is None:
            return np.exp(-1.0j * self.K * np.asarray(self.translations))

        phase_KR = np.asarray(phase_KR, dtype=np.complex128)
        if phase_KR.shape != (self.nR,):
            raise ValueError(f"phase_KR must have shape {(self.nR,)}, got {phase_KR.shape}.")
        return phase_KR

    def build(self, hamiltonian) -> "AbInitioDD2Trial":
        """Attach an ab-initio EPh Hamiltonian and compute coherent centers."""
        self.validate_hamiltonian(hamiltonian)
        self.hamiltonian = hamiltonian
        self.omega_qnu = np.asarray(hamiltonian.omega_qnu, dtype=np.float64)
        self.x0_qnu, self.p0_qnu = self.compute_centers(self.omega_qnu)
        return self

    def validate_hamiltonian(self, hamiltonian) -> None:
        if hamiltonian.eps_kj.shape != self.psi_kj.shape:
            raise ValueError(
                "Hamiltonian eps_kj shape does not match psi_kj: "
                f"{hamiltonian.eps_kj.shape} != {self.psi_kj.shape}."
            )

        if hamiltonian.omega_qnu.shape != self.beta_qnu.shape:
            raise ValueError(
                "Hamiltonian omega_qnu shape does not match beta_qnu: "
                f"{hamiltonian.omega_qnu.shape} != {self.beta_qnu.shape}."
            )

    def compute_centers(self, omega_qnu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute :math:`\bar X_{\mathbf{q}\nu}(0)` and
        :math:`\bar P_{\mathbf{q}\nu}(0)` from ``beta_qnu``.
        """
        omega_qnu = np.asarray(omega_qnu, dtype=np.float64)
        if omega_qnu.shape != self.beta_qnu.shape:
            raise ValueError(
                f"omega_qnu must have shape {self.beta_qnu.shape}, got {omega_qnu.shape}."
            )
        if np.any(omega_qnu < 0.0):
            raise ValueError("omega_qnu must contain non-negative phonon frequencies.")

        beta_minus_conj = np.conj(self.beta_qnu[self.minus_q])
        x0_qnu = (self.beta_qnu + beta_minus_conj) / np.sqrt(2.0 * omega_qnu)
        p0_qnu = -1.0j * np.sqrt(0.5 * omega_qnu) * (self.beta_qnu - beta_minus_conj)
        return x0_qnu, p0_qnu

    def translated_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return translated coherent centers with shape ``(nq, nmode, nR)``."""
        self._require_centers()
        phase_minus_qR = np.conj(self.phase_qR)[:, None, :]
        x_qnu_R = self.x0_qnu[:, :, None] * phase_minus_qR
        p_qnu_R = self.p0_qnu[:, :, None] * phase_minus_qR
        return x_qnu_R, p_qnu_R

    def electronic_amplitudes_flat(self) -> np.ndarray:
        """Return ``psi_kj`` flattened to the Hamiltonian electronic ordering."""
        return self.psi_kj.reshape(self.nbasis)

    def calc_overlap(self, walkers) -> np.ndarray:
        """Refresh and return the walker dD2 overlaps."""
        return walkers.update_overlap(self)

    def calc_electronic_overlap(self, walkers) -> np.ndarray:
        """Refresh and return ``A_el[R]`` for each walker."""
        return walkers.update_electronic_overlap(self)

    def calc_phonon_overlap(self, walkers) -> np.ndarray:
        """Refresh and return ``A_ph[R]`` for each walker."""
        return walkers.update_phonon_overlap(self)

    def calc_phonon_gradient(self, walkers) -> np.ndarray:
        """Return the phonon log derivative used for importance sampling."""
        return walkers.phonon_log_gradient(self)

    def _require_centers(self) -> None:
        if self.x0_qnu is None or self.p0_qnu is None:
            raise RuntimeError("Call trial.build(hamiltonian) before requesting coherent centers.")


AbInitiodD2Trial = AbInitioDD2Trial

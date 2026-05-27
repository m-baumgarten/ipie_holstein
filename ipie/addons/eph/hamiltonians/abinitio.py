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


class AbInitioEPhHamiltonian:
    r"""Container for a band-basis ab-initio electron-phonon Hamiltonian.

    This class stores the ingredients for

    .. math::

        \hat H_{\mathrm{el}} =
        \sum_{\mathbf{k}j}\epsilon_{\mathbf{k}j}
        c^\dagger_{\mathbf{k}j}c_{\mathbf{k}j},

    .. math::

        \hat H_{\mathrm{elph}} =
        \sum_{\mathbf{k}\mathbf{q}mn\nu}
        g^\nu_{mn}(\mathbf{k},\mathbf{q})\sqrt{2\omega_{\mathbf{q}\nu}}\,
        X_{\mathbf{q}\nu}\,
        c^\dagger_{\mathbf{k}+\mathbf{q},m}c_{\mathbf{k}n}.

    The supplied ``g_qnu_kmn`` tensor is used as-is and is assumed to include
    any Brillouin-zone normalization, such as a conventional ``1/sqrt(N)``
    factor from the Fourier transform or low-rank reconstruction.  No
    electron-electron interaction is included.  The dense electron-phonon matrix
    constructor is intentionally provided for debugging and small-system
    prototyping; production propagation can later replace it by a matrix action
    with the same tensor conventions.

    Parameters
    ----------
    eps_kj
        Electronic band energies with shape ``(nk, nband)``.
    g_qnu_kmn
        Electron-phonon vertices with shape
        ``(nq, nmode, nk, nband, nband)``.  The last two indices are
        ``(m, n)`` in :math:`c^\dagger_{\mathbf{k}+\mathbf{q},m}c_{\mathbf{k}n}`.
    omega_qnu
        Phonon frequencies with shape ``(nq, nmode)``.
    k_plus_q
        Optional integer lookup table with shape ``(nk, nq)`` such that
        ``k_plus_q[k, q]`` is the index of ``k + q`` on the electronic grid.
        If omitted, ``nq`` must equal ``nk`` and modular one-dimensional
        indexing is assumed.
    kpoints, qpoints
        Optional metadata arrays. They are stored but not interpreted.
    ecore
        Constant scalar energy shift.
    """

    def __init__(
        self,
        eps_kj: np.ndarray,
        g_qnu_kmn: np.ndarray,
        omega_qnu: np.ndarray,
        k_plus_q: Optional[np.ndarray] = None,
        kpoints: Optional[np.ndarray] = None,
        qpoints: Optional[np.ndarray] = None,
        ecore: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.name = "AbInitioEPh"
        self.verbose = verbose
        self.ecore = ecore

        self.eps_kj = np.asarray(eps_kj)
        self.g_qnu_kmn = np.asarray(g_qnu_kmn, dtype=np.complex128)
        self.omega_qnu = np.asarray(omega_qnu, dtype=np.float64)

        self._validate_core_shapes()

        self.nk, self.nband = self.eps_kj.shape
        self.nq, self.nmode = self.omega_qnu.shape
        self.nbasis = self.nk * self.nband
        self.nphonon_modes = self.nq * self.nmode
        self.N = self.nk

        self.k_plus_q = self._build_k_plus_q(k_plus_q)
        self.kpoints = None if kpoints is None else np.asarray(kpoints)
        self.qpoints = None if qpoints is None else np.asarray(qpoints)

        self.T = self.build_T()
        self.h1e = self.T

        if self.verbose:
            print("# Built AbInitioEPhHamiltonian.")
            print(f"# Number of k-points: {self.nk}")
            print(f"# Number of q-points: {self.nq}")
            print(f"# Number of bands: {self.nband}")
            print(f"# Number of phonon branches: {self.nmode}")

    def _validate_core_shapes(self) -> None:
        if self.eps_kj.ndim != 2:
            raise ValueError("eps_kj must have shape (nk, nband).")

        if self.g_qnu_kmn.ndim != 5:
            raise ValueError("g_qnu_kmn must have shape (nq, nmode, nk, nband, nband).")

        if self.omega_qnu.ndim != 2:
            raise ValueError("omega_qnu must have shape (nq, nmode).")

        nk, nband = self.eps_kj.shape
        nq, nmode = self.omega_qnu.shape
        expected_g_shape = (nq, nmode, nk, nband, nband)
        if self.g_qnu_kmn.shape != expected_g_shape:
            raise ValueError(
                "g_qnu_kmn shape mismatch: expected "
                f"{expected_g_shape}, got {self.g_qnu_kmn.shape}."
            )

        if np.any(self.omega_qnu < 0.0):
            raise ValueError("omega_qnu must contain non-negative phonon frequencies.")

    def _build_k_plus_q(self, k_plus_q: Optional[np.ndarray]) -> np.ndarray:
        if k_plus_q is None:
            if self.nq != self.nk:
                raise ValueError("k_plus_q is required when nq != nk.")
            return (
                np.arange(self.nk, dtype=np.int64)[:, None]
                + np.arange(self.nq, dtype=np.int64)[None, :]
            ) % self.nk

        k_plus_q = np.asarray(k_plus_q, dtype=np.int64)
        if k_plus_q.shape != (self.nk, self.nq):
            raise ValueError(
                f"k_plus_q must have shape {(self.nk, self.nq)}, got {k_plus_q.shape}."
            )

        if np.any(k_plus_q < 0) or np.any(k_plus_q >= self.nk):
            raise ValueError("k_plus_q entries must be valid electronic k-point indices.")

        return k_plus_q

    def build(self) -> "AbInitioEPhHamiltonian":
        """Mirror the addon Hamiltonian API used by the lattice EPh models."""
        self.T = self.build_T()
        self.h1e = self.T
        return self

    def build_T(self) -> np.ndarray:
        """Return spin-independent diagonal one-body electronic matrices."""
        h1 = np.diag(self.eps_kj.reshape(self.nbasis)).astype(np.complex128)
        return np.array([h1.copy(), h1.copy()])

    def build_g(self) -> np.ndarray:
        """Return the stored band-basis electron-phonon vertex tensor."""
        return self.g_qnu_kmn

    def flatten_electronic(self, amplitudes_kj: np.ndarray) -> np.ndarray:
        """Flatten an array with final axes ``(nk, nband)`` to ``(nbasis,)``."""
        amplitudes_kj = np.asarray(amplitudes_kj)
        if amplitudes_kj.shape[-2:] != (self.nk, self.nband):
            raise ValueError(
                "Expected final axes "
                f"{(self.nk, self.nband)}, got {amplitudes_kj.shape[-2:]}."
            )
        return amplitudes_kj.reshape(*amplitudes_kj.shape[:-2], self.nbasis)

    def unflatten_electronic(self, amplitudes_p: np.ndarray) -> np.ndarray:
        """Unflatten an array with final axis ``nbasis`` to ``(nk, nband)``."""
        amplitudes_p = np.asarray(amplitudes_p)
        if amplitudes_p.shape[-1] != self.nbasis:
            raise ValueError(f"Expected final axis {self.nbasis}, got {amplitudes_p.shape[-1]}.")
        return amplitudes_p.reshape(*amplitudes_p.shape[:-1], self.nk, self.nband)

    def flatten_index(self, k: int, band: int) -> int:
        """Return the flattened electronic index for ``(k, band)``."""
        return k * self.nband + band

    def split_index(self, p: int) -> Tuple[int, int]:
        """Return ``(k, band)`` for flattened electronic index ``p``."""
        return divmod(p, self.nband)

    def zero_point_energy(self) -> float:
        r"""Return :math:`E_0 = \frac{1}{2}\sum_{\mathbf{q}\nu}\omega_{\mathbf{q}\nu}`."""
        return 0.5 * float(np.sum(self.omega_qnu))

    def phonon_potential(self, X_qnu: np.ndarray) -> np.ndarray:
        r"""Return :math:`\frac{1}{2}\sum_{\mathbf{q}\nu}\omega^2|X|^2`.

        ``X_qnu`` may be a single configuration with shape ``(nq, nmode)`` or
        a batch whose final axes have that shape.
        """
        X_qnu = np.asarray(X_qnu)
        if X_qnu.shape[-2:] != (self.nq, self.nmode):
            raise ValueError(
                f"Expected final axes {(self.nq, self.nmode)}, got {X_qnu.shape[-2:]}."
            )
        return 0.5 * np.sum((self.omega_qnu**2) * np.abs(X_qnu) ** 2, axis=(-2, -1))

    def construct_eph_matrix(self, X_qnu: np.ndarray) -> np.ndarray:
        r"""Construct the dense one-body e-ph matrix for a phonon configuration.

        The returned matrix has shape ``(nbasis, nbasis)`` and is ordered by the
        flattened electronic index ``p = k * nband + band``.  If a batch of
        phonon configurations is supplied, with shape ``(..., nq, nmode)``, the
        returned array has shape ``(..., nbasis, nbasis)``.
        """
        X_qnu = np.asarray(X_qnu, dtype=np.complex128)
        if X_qnu.shape[-2:] != (self.nq, self.nmode):
            raise ValueError(
                f"Expected final axes {(self.nq, self.nmode)}, got {X_qnu.shape[-2:]}."
            )

        if X_qnu.ndim > 2:
            batch_shape = X_qnu.shape[:-2]
            matrices = [
                self.construct_eph_matrix(X)
                for X in X_qnu.reshape((-1, self.nq, self.nmode))
            ]
            return np.asarray(matrices).reshape(*batch_shape, self.nbasis, self.nbasis)

        eph = np.zeros((self.nbasis, self.nbasis), dtype=np.complex128)
        mode_factors = np.sqrt(2.0 * self.omega_qnu) * X_qnu

        for iq in range(self.nq):
            for inu in range(self.nmode):
                factor = mode_factors[iq, inu]
                if factor == 0.0:
                    continue
                for ik in range(self.nk):
                    kout = self.k_plus_q[ik, iq]
                    row = slice(kout * self.nband, (kout + 1) * self.nband)
                    col = slice(ik * self.nband, (ik + 1) * self.nband)
                    eph[row, col] += factor * self.g_qnu_kmn[iq, inu, ik]

        return eph

    def apply_eph_matrix(self, X_qnu: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Apply the dense debugging e-ph matrix to an electronic vector/matrix."""
        return self.construct_eph_matrix(X_qnu) @ vec


EphHamiltonianAbInitio = AbInitioEPhHamiltonian

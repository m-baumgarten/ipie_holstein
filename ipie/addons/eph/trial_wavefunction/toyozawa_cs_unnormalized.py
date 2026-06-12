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
from typing import Tuple

from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import circ_perm
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp
from ipie.addons.eph.trial_wavefunction.toyozawa import (
    ToyozawaTrial,
    _normalise_by_nonzero_overlap,
)
from ipie.addons.eph.trial_wavefunction.toyozawa_cs import ToyozawaTrialCoherentState

class ToyozawaTrialUnnormalizedCoherentState(ToyozawaTrialCoherentState):  
    def __init__(
        self,
        wavefunction: np.ndarray,
        w0: float,
        num_elec: Tuple[int, int],
        num_basis: int,
        K: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(wavefunction, w0, num_elec, num_basis, K, verbose=verbose)
        self.coherent_state_convention = "unnormalized"
        self._beta_perm_conj = np.asarray(
            [self.beta_shift[perm].conj() for perm in self.perms],
            dtype=np.complex128,
        )

    def calc_energy(self, ham, zero_th=1e-12):
        r"""Computes the variational energy of the trial, i.e.

        .. math::
            E_T = \frac{\langle\Psi_T|\hat{H}|\Psi_T\rangle}{\langle\Psi_T|\Psi_T\rangle}.

        As the Toyozawa trial wavefunction is a superposition of coherent state trials
        the evaluation of :math:`E_T` a naive implementation would scale quadratically
        with the number of sites. Here, we exploit the translational symmetry of the
        wavefunction to obtain linear scaling.

        Parameters
        ----------
        ham:
            Hamiltonian

        Returns
        -------
        etrial : :class:`float`
            Trial energy
        """
        num_energy = 0.0
        num_ph_energy = 0.0
        num_meanfield = np.zeros(ham.nsites, dtype=np.complex128)
        denom = 0.0
        beta0 = self.beta_shift
        for ip, (coeff, perm) in enumerate(zip(self.kcoeffs, self.perms)):
            psia_i = self.psia[perm, :]
            beta_i = beta0[perm]

            if self.ndown > 0:
                psib_i = self.psib[perm, :]
                ov = (
                    np.linalg.det(self.psia.conj().T.dot(psia_i))
                    * np.linalg.det(self.psib.conj().T.dot(psib_i))
                    * np.prod(np.exp(beta0.conj() * beta_i))
                )
            else:
                ov = np.linalg.det(self.psia.conj().T.dot(psia_i)) * np.prod(
                    np.exp(beta0.conj() * beta_i)
                )
            ov *= self.kcoeffs[0].conj() * coeff

            if np.abs(ov) < zero_th:
                continue

            if ip != 0:
                ov = ov * (self.nbasis - ip) * 2
            else:
                ov = ov * self.nbasis

            Ga_i, _, _ = gab_mod_ovlp(self.psia, psia_i)
            if self.ndown > 0:
                Gb_i, _, _ = gab_mod_ovlp(self.psib, psib_i)
            else:
                Gb_i = np.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]

            kinetic = np.sum(ham.T[0] * G_i[0] + ham.T[1] * G_i[1])
            e_ph = ham.w0 * np.sum(beta0.conj() * beta_i)
            e_eph = np.einsum('ijk,ij,k->', ham.g_tensor, G_i[0], beta0.conj() + beta_i)
#            if ip != 0:
#            print('eeph:    ', e_eph)
#            print('Gi calc_energ:   ', G_i[0])
            if self.ndown > 0:
                e_eph += np.einsum('ijk,ij,k->', ham.g_tensor, G_i[1], beta0.conj() + beta_i)
#            rho = ham.g_tensor * (G_i[0] + G_i[1])
#            e_eph = np.sum(np.dot(rho, beta0.conj() + beta_i))

            num_energy += np.real((kinetic + e_ph + e_eph) * ov)
            num_ph_energy += np.real(e_ph * ov)
            num_meanfield += np.real(np.einsum('ijk,ij,k->k', ham.g_tensor, G_i[0], beta0.conj() + beta_i) * ov)
            denom += np.real(ov)

        etrial = num_energy / denom
        etrial_ph = num_ph_energy / denom
        self.mf_eph = num_meanfield / denom
#        print('etrial:', etrial)
        return etrial, etrial_ph

    def calc_phonon_overlap_perms(self, walkers: EPhCSWalkers) -> np.ndarray:
        r""""""
       # print('ph_ovlp I: ', walkers.ph_ovlp[:1, :])
        for ip, perm in enumerate(self.perms):
            # Unnormalized CS
            ph_ov = np.exp(self.beta_shift[perm].conj() * walkers.coherent_state_shift)
            walkers.ph_ovlp[:, ip] = np.prod(ph_ov, axis=1)

        return walkers.ph_ovlp

    def calc_ito_force_bias(
        self,
        walkers: EPhCSWalkers,
        g_tensor_residual_dagger: np.ndarray,
        zero_overlap_threshold: float = 1.0e-14,
    ) -> np.ndarray:
        r"""Return the proper-complex Ito drift from the log trial overlap.

        The returned drift is

            0.5 Re(A - B) + 0.5j Re(i(A + B)),

        where A is the coherent-state log derivative and B is the electronic
        response to the same residual creation generator used by the propagator.
        """
        A, B = self.calc_ito_log_derivatives(
            walkers,
            g_tensor_residual_dagger,
            zero_overlap_threshold=zero_overlap_threshold,
        )
        return 0.5 * (A.conj() - B)

    def calc_ito_log_derivatives(
        self,
        walkers: EPhCSWalkers,
        g_tensor_residual_dagger: np.ndarray,
        zero_overlap_threshold: float = 1.0e-14,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return the ``A`` and ``B`` log-overlap derivatives for Ito gauges."""
        self.calc_overlap_perm(walkers)
        overlap = np.sum(walkers.ovlp_perm, axis=1)

        rho = np.zeros_like(walkers.ovlp_perm, dtype=np.complex128)
        active = np.abs(overlap) > zero_overlap_threshold
        rho[active] = walkers.ovlp_perm[active] / overlap[active, None]

        A = np.einsum("np,pm->nm", rho, self._beta_perm_conj)

        B = np.zeros_like(A)
        for ip, perm in enumerate(self.perms):
            psia_perm = self.psia[perm, :]
            B_perm = self._calc_ito_spin_log_derivative(
                psia_perm, walkers.phia, g_tensor_residual_dagger
            )
            if self.ndown > 0:
                psib_perm = self.psib[perm, :]
                B_perm += self._calc_ito_spin_log_derivative(
                    psib_perm, walkers.phib, g_tensor_residual_dagger
                )
            B += rho[:, ip, None] * B_perm

        return A, B

    @staticmethod
    def _calc_ito_spin_log_derivative(
        trial_orbitals: np.ndarray,
        walker_orbitals: np.ndarray,
        g_tensor_residual_dagger: np.ndarray,
    ) -> np.ndarray:
        if trial_orbitals.shape[1] == 0:
            return np.zeros(
                (walker_orbitals.shape[0], g_tensor_residual_dagger.shape[2]),
                dtype=np.complex128,
            )

        overlap_matrix = np.einsum(
            "ia,nie->nae", trial_orbitals.conj(), walker_orbitals
        )
        try:
            inverse_overlap = np.linalg.inv(overlap_matrix)
        except np.linalg.LinAlgError:
            inverse_overlap = np.linalg.pinv(overlap_matrix)

        generated_orbitals = np.einsum(
            "ijm,nje->niem", g_tensor_residual_dagger, walker_orbitals
        )
        response_matrix = np.einsum(
            "ia,niem->naem", trial_orbitals.conj(), generated_orbitals
        )
        return np.einsum("nab,nbam->nm", inverse_overlap, response_matrix)

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
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial

class ToyozawaTrialCoherentState(ToyozawaTrial):  
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
                    * np.prod(np.exp(-0.5 * (np.abs(beta0)**2 + np.abs(beta_i)**2) + beta0.conj() * beta_i))
                )
            else:
                ov = np.linalg.det(self.psia.conj().T.dot(psia_i)) * np.prod(
                    np.exp(-0.5 * (np.abs(beta0)**2 + np.abs(beta_i)**2) + beta0.conj() * beta_i)
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
            rho = ham.g_tensor * (G_i[0] + G_i[1])
            e_eph = np.sum(np.dot(rho, beta0.conj() + beta_i))

            num_energy += np.real((kinetic + e_ph + e_eph) * ov)
            num_ph_energy += np.real(e_ph * ov)
            num_meanfield += np.real(np.dot(rho, beta0.conj() + beta_i) * ov)
            denom += np.real(ov)

        etrial = num_energy / denom
        etrial_ph = num_ph_energy / denom
        meanfield = num_meanfield / denom
        return etrial, etrial_ph, meanfield

    def calc_phonon_overlap_perms(self, walkers: EPhCSWalkers) -> np.ndarray:
        r""""""
        print('ph_ovlp I: ', walkers.ph_ovlp[:1, :])
        for ip, perm in enumerate(self.perms):
            ph_ov = np.exp(-0.5 * (np.abs(self.beta_shift[perm])**2 + np.abs(walkers.coherent_state_shift)**2 - 2*self.beta_shift[perm].conj() * walkers.coherent_state_shift))
            walkers.ph_ovlp[:, ip] = np.prod(ph_ov, axis=1)

        print('ph_ovlp II: ', walkers.ph_ovlp[:1, :])
        return walkers.ph_ovlp

    def calc_phonon_displacement(self, walkers: EPhCSWalkers, ham) -> np.ndarray:
        r""""""
        displacement = np.zeros((walkers.nwalkers, self.nperms), dtype=xp.complex128)
        for ip, (ovlp, Ga, Gb, perm) in enumerate(zip(walkers.ovlp_perm.T, walkers.Ga_perm.T, walkers.Gb_perm.T, self.perms)):
            displacement[:, ip] = np.einsum('ijk,ijn,nk->n', ham.g_tensor, Ga, self.beta_shift[perm].conj() + walkers.coherent_state_shift)
            if self.ndown > 0:
                displacement[:, ip] += np.einsum('ijk,ijn,nk->n', ham.g_tensor, Gb, self.beta_shift[perm].conj() + walkers.coherent_state_shift)
        
        displacement = np.einsum("np,n->n", displacement, 1 / np.sum(walkers.ovlp_perm, axis=1)) 
        return displacement 

    def calc_harm_osc(self, walkers: EPhCSWalkers) -> np.ndarray:
        r""""""
        harm_osc = np.zeros((walkers.nwalkers, self.nperms), dtype=xp.complex128)
        for ip, (ovlp, perm) in enumerate(zip(walkers.ovlp_perm.T, self.perms)):
            harm_osc[:, ip] = ovlp * np.sum(self.beta_shift[perm].conj() * walkers.coherent_state_shift, axis=1)
        
        harm_osc = np.einsum("np,n->n", harm_osc, 1 / np.sum(walkers.ovlp_perm, axis=1))         
        
        return harm_osc

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
from typing import List, Union
from scipy.optimize import minimize, basinhopping
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.hamiltonians.ssh import AcousticSSHModel, BondSSHModel
import jax
import jax.numpy as npj
import plum

from ipie.addons.eph.trial_wavefunction.variational.variational import Variational
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational, overlap_degeneracy


class dD1Variational(ToyozawaVariational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        K: Union[float, np.ndarray],
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, K, cplx)
        self.shift_params_rows = self.ham.N
#        self.perms = np.array([self.perms[0]])
#        self.Kcoeffs = np.array([1.])

    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]

        num_energy = 0.
        denom = 0.

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip,-ip), axis=(0,1))
            psia_i = c0a[permi, :]

            for jp, (permj, coeffj) in enumerate(zip(self.perms, self.Kcoeffs)):
                shift_j = npj.roll(shift, shift=(-jp,-jp), axis=(0,1))
                # Alternatively shift_j = shift[permj,:][:,permj]
                psia_j = c0a[permj, :]
                
                cs_ovlp = self.cs_overlap(shift_i, shift_j)

                overlap = npj.einsum('ie,ie,i->', psia_i.conj(), psia_j, cs_ovlp.diagonal()) 
                overlap *= coeffi.conj() * coeffj
#                jax.debug.print('ovlp:  {x}', x=(overlap,coeffi,coeffj))
#                jax.debug.print('elec:  {x}', x=(psia_i, psia_j, c0a.shape))
#                jax.debug.print('csovlp:    {x}', x=(cs_ovlp.diagonal()))
#                exit()

                if npj.abs(overlap) < zero_th:
                    continue

                Ga_ij = npj.outer(psia_i.conj(), psia_j)
                if self.sys.ndown > 0:
                    Gb_ij = npj.outer(psib_i.conj(), psib_j)
                else:
                    Gb_ij = npj.zeros_like(Ga_ij)
                G_ij = [Ga_ij, Gb_ij]
   
                projected_energy = self.projected_energy(self.ham, G_ij, shift_i, shift_j)
                num_energy += projected_energy * coeffi.conj() * coeffj #* overlap
                denom += overlap
#                jax.debug.print('energy, ovlp: {x}', x=(num_energy, denom)) 
        energy = num_energy / denom
        return energy.real
    
    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]

        num_energy = 0.
        denom = 0.


        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip,-ip), axis=(0,1))
                # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi, :]
                
            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = npj.einsum('ie,ie,i->', c0a.conj(), psia_i, cs_ovlp.diagonal()) 
            overlap *= self.Kcoeffs[0].conj() * coeffi
#                jax.debug.print('ovlp:  {x}', x=(overlap,coeffi,coeffj))
#                jax.debug.print('elec:  {x}', x=(psia_i, psia_j, c0a.shape))
#                jax.debug.print('csovlp:    {x}', x=(cs_ovlp.diagonal()))
#                exit()

            if npj.abs(overlap) < zero_th:
                continue

            overlap *= overlap_degeneracy(self.ham, ip)

            Ga_i = npj.outer(c0a.conj(), psia_i)
            if self.sys.ndown > 0:
                Gb_i = npj.outer(c0b.conj(), psib_i)
            else:
                Gb_i = npj.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]
   
            projected_energy = self.projected_energy(self.ham, G_i, shift, shift_i)
            num_energy += (projected_energy * overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi).real  #* overlap
            denom += overlap.real
#                jax.debug.print('energy, ovlp: {x}', x=(num_energy, denom)) 
        energy = num_energy / denom
        return energy.real

    def get_args(self):
        return ()

    def cs_overlap(self, shift_i, shift_j):
        cs_ovlp = npj.exp(shift_i.T.conj().dot(shift_j))
        cs_ovlp_norm_i = npj.exp(-0.5 * npj.einsum('ij->j', npj.abs(shift_i) ** 2))
        cs_ovlp_norm_j = npj.exp(-0.5 * npj.einsum('ij->j', npj.abs(shift_j) ** 2))
        cs_ovlp_norm = npj.outer(cs_ovlp_norm_i, cs_ovlp_norm_j)
        cs_ovlp *= cs_ovlp_norm
        return cs_ovlp

    @plum.dispatch
    def projected_energy(self, ham: HolsteinModel, G: list, shift_i, shift_j):

        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        
        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)
        rho = G[0].diagonal() + G[1].diagonal()
        displacement = shift_i.diagonal().conj() + shift_j.diagonal()
        el_ph_contrib = self.ham.g * npj.sum(rho * displacement * cs_ovlp.diagonal()) 
        phonon_contrib = ham.w0 * npj.einsum('ij,j->', shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal())

#        jax.debug.print('energies:  {x}', x=(kinetic, el_ph_contrib, phonon_contrib))
#        jax.debug.print('cs ovlp:   {x}', x=cs_ovlp)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy


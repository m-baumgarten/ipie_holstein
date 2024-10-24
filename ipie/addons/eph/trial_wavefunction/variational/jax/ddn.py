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

import jax.numpy as npj
import jax
from typing import List, Union
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
import plum
from ipie.addons.eph.trial_wavefunction.variational.jax.toyozawa import overlap_degeneracy
from ipie.addons.eph.trial_wavefunction.variational.jax.dd1 import dD1Variational
import numpy as np

class dDnVariational(dD1Variational):
    def __init__(
        self,
        shift_init: npj.ndarray,
        electron_init: npj.ndarray,
        hamiltonian,
        system,
        K: Union[float, npj.ndarray],
        n_rank: int,
        cplx: bool = True,
        freeze_guess: bool = False
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, K, cplx)
        assert n_rank % 2 == 0
        self.freeze_guess = freeze_guess
        if freeze_guess:
            assert shift_init.shape[1] != 2 * n_rank
            self.shift_params_rows = n_rank - shift_init.shape[1]
            self.frozen_ranks = shift_init.copy()
            self.n_frozen = shift_init.shape[1] // 2
            self.make_shift_matrix = self.make_shift_matrix_frozen
            self.shift = np.zeros_like((self.ham.N, self.shift_params_rows), dtype=np.complex128)
        else:
            self.shift_params_rows = n_rank
            self.make_shift_matrix = self.make_shift_matrix_default
    
    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]

        shift = self.make_shift_matrix(shift)

        num_energy = 0.
        denom = 0.

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip,-ip), axis=(0,1))
            psia_i = c0a[permi, :]
                
            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = npj.einsum('ie,ie,i->', c0a.conj(), psia_i, cs_ovlp.diagonal()) 
            overlap *= self.Kcoeffs[0].conj() * coeffi

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
            num_energy += (projected_energy * overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi).real
            denom += overlap.real
        
        energy = num_energy / denom
        return energy.real
    
    def make_shift_matrix_default(self, beta):
#        jax.debug.print('shift: {x}', x=npj.einsum('ij,jk->ik', beta.conj()[:, : self.shift_params_rows//2], beta[:, self.shift_params_rows//2 :].T))
        return npj.einsum('ij,jk->ik', beta.conj()[:, : self.shift_params_rows//2], beta[:, self.shift_params_rows//2 :].T)

#    def make_shift_matrix_lu(self, beta):
#        

    def make_shift_matrix_frozen(self, beta):
#        jax.debug.print('beta shape:    {x}', x=beta.shape)
        new_beta = npj.hstack([self.frozen_ranks[: , : self.n_frozen], beta[:, :self.shift_params_rows // 2], self.frozen_ranks[: , self.n_frozen: ], beta[:, self.shift_params_rows // 2 :]])
        return self.make_shift_matrix_default(new_beta)

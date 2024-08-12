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
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
import jax
import jax.numpy as npj
import plum
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import overlap_degeneracy
from ipie.addons.eph.trial_wavefunction.variational.dd1 import dD1Variational


class GlobalLocalVariational(dD1Variational):
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
        self.shift_params_rows = 2
    
    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]

        shift_tmp = npj.repeat(shift[:,0][:,npj.newaxis], self.ham.N, axis=1)
        shift_tmp -= self.shift_roll(shift[:,1])
        shift = shift_tmp

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
    
    def shift_roll(self, beta) -> np.ndarray:
        circs = beta
        for shift in range(1, len(beta)):
            new_circ = npj.roll(beta, shift)
            circs = npj.vstack([circs, new_circ])
        return circs.T


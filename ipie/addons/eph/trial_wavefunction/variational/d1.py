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


class D1Variational(Variational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)
        self.shift_params_rows = self.ham.N * self.ham.dim

    def objective_function(self, x) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)

        ovlp = self.overlap(shift, c0a, c0b)
        
        Ga = npj.outer(c0a.conj(), c0a)
        if self.sys.ndown > 0:
            Gb = npj.outer(c0b.conj(), c0b)
        else: 
            Gb = npj.zeros_like(Ga, dtype=np.float64)
        G = [Ga, Gb]
      
#        jax.debug.print('elec:  {x}', x=c0a)
#        jax.debug.print('ovlp:  {x}', x=ovlp)
#        exit()
        #Ga = gab(c0a, c0a)
        #if self.sys.ndown > 0:
        #    Gb = gab(c0b, c0b)
        #else:
        #    Gb = npj.zeros_like(Ga, dtype=np.float64)
        #G = [Ga, Gb]
        
        energy = self.variational_energy(self.ham, G, shift) #/ ovlp
#        jax.debug.print('final energy:  {x}', x=energy)
        return energy.real

    def get_args(self):
        return ()

    def overlap(self, shift, c0a, c0b):
#        cs_ovlp = npj.exp(npj.einsum('dij,dji->i', shift.T.conj(), shift))
        el_ovlp_a = npj.abs(c0a) ** 2
        if self.sys.ndown > 0:
            el_ovlp_b = npj.abs(c0b) ** 2
        else:
            el_ovlp_b = 0
        ovlp = el_ovlp_a #* cs_ovlp
        return npj.sum(ovlp)

    @plum.dispatch
    def variational_energy(self, ham: HolsteinModel, G: list, shift):
        ovlp = npj.einsum('ii->', G[0])
        shift = shift[0]
#        shift = npj.diag(shift[0,:,0])

        cs_ovlp = npj.exp(shift.T.conj().dot(shift))
        cs_ovlp_norm = npj.exp(-0.5 * npj.einsum('ij->j', npj.abs(shift) ** 2))
#        cs_ovlp_norm = npj.exp(-0.5 *npj.abs(shift) ** 2)
        cs_ovlp_norm = npj.outer(cs_ovlp_norm, cs_ovlp_norm)
        cs_ovlp *= cs_ovlp_norm
#        jax.debug.print('cs_ovlp:   {x}', x=(cs_ovlp, shift))

        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)
        rho = G[0].diagonal() + G[1].diagonal()
        displacement = shift.diagonal().conj() + shift.diagonal()
#        el_ph_contrib = -self.ham.g * npj.sum(rho * displacement * cs_ovlp.diagonal())
        el_ph_contrib = self.ham.g * npj.sum(rho * displacement)
#        phonon_contrib = ham.w0 * np.sum(shift.diagonal().conj() * shift.diagonal() * cs_ovlp.diagonal())
#        phonon_contrib = ham.w0 * np.sum(shift.diagonal().conj() * shift.diagonal() * G[0].diagonal()) / ovlp 
        phonon_contrib = ham.w0 * npj.einsum('ij,j->', npj.abs(shift)**2, G[0].diagonal())

#        jax.debug.print('energies:  {x}', x=(kinetic, el_ph_contrib, phonon_contrib))
#        jax.debug.print('ovlp:  {x}', x=ovlp)
#        exit()
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy / ovlp



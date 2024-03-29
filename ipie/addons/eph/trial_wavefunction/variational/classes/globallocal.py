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

from ipie.addons.eph.trial_wavefunction.variational.classes.toyozawa import circ_perm, ToyozawaVariational

class GlobalLocalVariational(ToyozawaVariational):
    def __init__(self, shift_init: np.ndarray, electron_init: np.ndarray,
            hamiltonian, system, K = 0.):
        super().__init__(shift_init, electron_init, hamiltonian, system, K)

    def unpack_x(x) -> Tuple:
        """Extracts alpha and beta shift additionally to Slater Determinants"""
        shift = x[0 : 2 * self.ham.nsites].copy()
        shift.reshape((2, self.ham.nsites))
        shift = shift.astype(np.float64)

        c0a = x[2 * self.ham.nsites : (self.sys.nup + 2) * self.ham.nsites].copy()
        c0a = jax.numpy.reshape(c0a, (self.sys.nup, self.ham.nsites)).T
        c0a = c0a.astype(np.float64)

        if self.sys.ndown > 0:
            c0b = x[(self.sys.nup + 2) * self.ham.nsites :].copy()
            c0b = jax.numpy.reshape(c0b, (self.sys.ndown, self.ham.nsites)).T
            c0b = c0b.astype(np.float64)
        else:
            c0b = npj.zeros_like(c0a, dtype=np.float64)

        return shift, c0a, c0b

    def objective_function(self, x, zero_th=1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift_b = shift[0, :]
        shift_a = shift[1, :]

        num_energy = 0.0
        denom = 0.0
        
        coeffs = npj.outer(self.Kcoeffs.conj(), self.Kcoeffs)

        for ip, (perm_i, coeff_i) in enumerate(zip(self.perms, self.Kcoeffs)):

            beta_i = shift_b[npj.array(perm_i)]
            psia_i = c0a[perm_i, :]

            for jp, (perm_j, coeff_j) in enumerate(zip(self.perms, self.Kcoeffs)):
                beta_j = shift_b[npj.array(perm_j)]
                psia_j = c0a[perm_j, :]
                


            # Compute Overlap and determine whether this permutation contributes to E_var
            boson_overlap_tensor = 
            overlap = npj.linalg.det(c0a.T.dot(psia_j)) * npj.prod(
                npj.exp(-0.5 * (shift_b**2 + beta_j**2) + shift_b * beta_j)
            )
            if self.sys.ndown > 0:
                psib_j = c0b[perm, :]
                overlap *= npj.linalg.det(c0b.T.dot(psib_j))
            overlap *= self.Kcoeffs[0] * coeff.conj()
            
            if npj.abs(overlap) < zero_th:
                continue
            
            if ip != 0:
                overlap = overlap.real * (self.ham.nsites-ip) * 2
            else:
                overlap = overlap.real * self.ham.nsites

            # Evaluate Greens functions
            Ga_j = gab(c0a, psia_j)
            if self.sys.ndown > 0:
                Gb_j = gab(c0b, psib_j)
            else:
                Gb_j = npj.zeros_like(Ga_j)
            G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
            projected_energy = self.projected_energy(self.ham, G_j, shift, beta_j)

            
            num_energy += projected_energy * overlap
            denom += overlap

        energy = num_energy / denom
        return energy

    def get_args(self):
        return ()

    @plum.dispatch
    def projected_energy(self, ham: HolsteinModel, G: list, shift, beta_j):
        rho = G[0].diagonal() + G[1].diagonal()
        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        phonon_contrib = ham.w0 * npj.sum(shift * beta_j)
        el_ph_contrib = -ham.g * npj.dot(rho, shift + beta_j)
        projected_energy = kinetic + el_ph_contrib + phonon_contrib
        return projected_energy
 

    @plum.dispatch
    def projected_energy(self, ham: AcousticSSHModel, G: list, shift, beta_j):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        
        X = shift + beta_j
        displ = npj.array(ham.X_connectivity).dot(X)
        displ_mat = npj.diag(displ[:-1], 1)
        displ_mat += displ_mat.T
        if ham.pbc:
            displ_mat = displ_mat.at[0,-1].set(displ[-1])
            displ_mat = displ_mat.at[-1,0].set(displ[-1])

        tmp0 = ham.g_tensor * G[0] * displ_mat
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * displ_mat
        el_ph_contrib = jax.numpy.sum(tmp0)
        
        phonon_contrib = ham.w0 * jax.numpy.sum(shift * beta_j)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def variational_energy(self, ham: BondSSHModel, G: list, shift, beta_j):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        
        X = shift + beta_j
        tmp0 = ham.g_tensor * G[0] * X
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * X
        el_ph_contrib = 2 * jax.numpy.sum(tmp0)
        
        phonon_contrib = ham.w0 * jax.numpy.sum(shift * beta_j)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

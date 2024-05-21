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
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectronHole
import jax
import jax.numpy as npj
import plum

from ipie.addons.eph.trial_wavefunction.variational.el_hole.variational_el_hole import VariationalElectronHole
from ipie.addons.eph.trial_wavefunction.variational.el_hole.toyozawa_el_hole import ToyozawaVariationalElectronHole

class CoherentStateVariationalElectronHole(ToyozawaVariationalElectronHole):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hole_init: np.ndarray,
        hamiltonian,
        system,
        K: float = 0.0,
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hole_init, hamiltonian, system, K, cplx)
        assert K == 0.
        self.perms = np.array([self.perms[0]])
        self.nperms = 1
        self.Kcoeffs = np.array([self.Kcoeffs[0]])


class _CoherentStateVariationalElectronHole(VariationalElectronHole):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hole_init: np.ndarray,
        hamiltonian,
        system,
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hole_init, hamiltonian, system, cplx)
        assert cplx is True

    def objective_function(self, x) -> float:
        """"""
        shift, c0a, c0b, h0a, h0b = self.unpack_x(x)

        Gea = gab(c0a, c0a)
        if self.sys.ndown > 0:
            Geb = gab(c0b, c0b)
        else:
            Geb = npj.zeros_like(Gea, dtype=np.float64)
        Ge = [Gea, Geb]

        Gha = gab(h0a, h0a)
        if self.sys.ndown > 0:
            Ghb = gab(h0b, h0b)
        else:
            Ghb = npj.zeros_like(Gha, dtype=np.float64)
        Gh = [Gha, Ghb]

        energy = self.variational_energy(self.ham, Ge, Gh, shift)
        return energy.real

    def get_args(self):
        return ()

    @plum.dispatch
    def variational_energy(self, ham: ExcitonPhononCavityElectronHole, Ge: list, Gh: list, shift):
        kinetic_e = np.sum(ham.Te[0] * Ge[0] + ham.Te[1] * Ge[1])
        kinetic_h = np.sum(ham.Th[0] * Gh[0] + ham.Th[1] * Gh[1])
        kinetic = kinetic_e + kinetic_h

        el_el_contrib = np.sum(ham.quade[0] * Ge[0] + ham.quade[1] * Ge[1])
        h_h_contrib = np.sum(ham.quadh[0] * Gh[0] + ham.quadh[1] * Gh[1])
        
        # check math for the following!
        el_h_contrib = -2 * npj.einsum('ij,kl->', ham.quade[0] * Ge[0] + ham.quade[1] * Ge[1], ham.quadh[0] * Gh[0] + ham.quadh[1] * Gh[1])
        ##
        quad_contrib = el_el_contrib + el_h_contrib + h_h_contrib

        X = shift + shift.conj()

        disple_mat1 = npj.array(ham.Xe1).dot(X)
        disple_mat1 += disple_mat1.T
        tmpe1 = ham.ge1 * Ge[0] * disple_mat1
        if self.sys.ndown > 0:
            tmpe1 += ham.ge1 * Ge[1] * disple_mat1
 
        disple_mat2 = npj.array(ham.Xe2).dot(X)
        disple_mat2 += disple_mat2.T
        disple_mat2 -= npj.diag(npj.diag(disple_mat2)) / 2
        tmpe2 = ham.ge2 * Ge[0] * disple_mat2
        if self.sys.ndown > 0:
            tmpe2 += ham.ge2 * Ge[1] * disple_mat2

        el_ph_contrib = jax.numpy.sum(tmpe1 + tmpe2)

        displh_mat1 = npj.array(ham.Xh1).dot(X)
        displh_mat1 += displh_mat1.T
        tmph1 = ham.gh1 * Gh[0] * displh_mat1
        if self.sys.ndown > 0:
            tmpe1 += ham.gh1 * Gh[1] * displh_mat1
        displh_mat2 = npj.array(ham.Xh2).dot(X)
        displh_mat2 += displh_mat2.T
        displh_mat2 -= npj.diag(npj.diag(displh_mat2)) / 2
        tmph2 = ham.gh2 * Gh[0] * displh_mat2
        if self.sys.ndown > 0:
            tmph2 += ham.gh2 * Gh[1] * displh_mat2

        el_ph_contrib = jax.numpy.sum(tmpe1 + tmpe2 + tmph1 + tmph2)

        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * shift)
        local_energy = kinetic + el_ph_contrib + phonon_contrib + quad_contrib
        return local_energy
            


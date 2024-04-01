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

from ipie.addons.eph.trial_wavefunction.variational.classes.variational import Variational


class CoherentStateVariational(Variational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)

    def objective_function(self, x) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)

        Ga = gab(c0a, c0a)
        if self.sys.ndown > 0:
            Gb = gab(c0b, c0b)
        else:
            Gb = npj.zeros_like(Ga, dtype=np.float64)
        G = [Ga, Gb]
        energy = self.variational_energy(self.ham, G, shift)
        return energy.real

    def get_args(self):
        return ()

    @plum.dispatch
    def variational_energy(self, ham: HolsteinModel, G: list, shift):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        rho = G[0].diagonal() + G[1].diagonal()
        el_ph_contrib = -self.ham.g * np.sum(rho * 2 * shift.real)
        phonon_contrib = ham.w0 * np.sum(shift.conj() * shift)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    ### not yet complexificated
    @plum.dispatch
    def variational_energy(self, ham: AcousticSSHModel, G: list, shift):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

        displ = npj.array(ham.X_connectivity).dot(shift)
        displ_mat = npj.diag(displ[:-1], 1)
        displ_mat += displ_mat.T
        if ham.pbc:
            displ_mat = displ_mat.at[0, -1].set(displ[-1])
            displ_mat = displ_mat.at[-1, 0].set(displ[-1])

        tmp0 = ham.g_tensor * G[0] * displ_mat
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * displ_mat
        el_ph_contrib = 2 * jax.numpy.sum(tmp0)

        phonon_contrib = ham.w0 * jax.numpy.sum(shift * shift)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def variational_energy(self, ham: BondSSHModel, G: list, shift):
        kinetic = np.sum(hamiltonian.T[0] * G[0] + hamiltonian.T[1] * G[1])

        offdiags = shift[:-1]
        displ = npj.diag(offdiags, -1)
        displ += displ.T
        if ham.pbc:
            displ = displ.at[-1, 0].set(shift[-1])
            displ = displ.at[0, -1].set(shift[-1])

        tmp0 = ham.g_tensor * G[0] * displ
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * displ
        el_ph_contrib = 2 * jax.numpy.sum(tmp0)

        phonon_contrib = ham.w0 * jax.numpy.sum(shift * shift)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

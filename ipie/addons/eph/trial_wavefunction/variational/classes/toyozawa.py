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
from typing import List, Union, Tuple
from scipy.optimize import minimize, basinhopping
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.hamiltonians.ssh import AcousticSSHModel, BondSSHModel
import jax
import jax.numpy as npj
import plum

from ipie.addons.eph.trial_wavefunction.variational.classes.variational import Variational


def circ_perm(lst: np.ndarray) -> np.ndarray:
    """Returns a matrix which rows consist of all possible
    cyclic permutations given an initial array lst.

    Parameters
    ----------
    lst :
        Initial array which is to be cyclically permuted
    """
    circs = lst
    for shift in range(1, len(lst)):
        new_circ = np.roll(lst, -shift)
        circs = np.vstack([circs, new_circ])
    return circs


class ToyozawaVariational(Variational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        K: float = 0.0,
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)
        self.K = K
        self.perms = circ_perm(np.arange(hamiltonian.nsites))
        self.nperms = self.perms.shape[0]
        self.Kcoeffs = np.exp(1j * K * np.arange(hamiltonian.nsites))

    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift_abs = npj.abs(shift)

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):

            beta_i = shift[npj.array(permi)]
            beta_i_abs = npj.abs(beta_i)
            psia_i = c0a[permi, :]

            overlap = npj.linalg.det(c0a.conj().T.dot(psia_i)) * npj.prod(
                npj.exp(-0.5 * (shift_abs**2 + beta_i_abs**2) + shift.conj() * beta_i)
            )
            if self.sys.ndown > 0:
                psib_i = c0b[permi, :]
                overlap *= npj.linalg.det(c0b.conj().T.dot(psib_i))
            overlap *= self.Kcoeffs[0].conj() * coeffi

            if npj.abs(overlap) < zero_th:
                continue

            if ip != 0:
                overlap = overlap * (self.ham.nsites - ip) * 2
            else:
                overlap = overlap * self.ham.nsites

            # Evaluate Greens functions
            Ga_j = gab(c0a, psia_i)
            if self.sys.ndown > 0:
                Gb_j = gab(c0b, psib_i)
            else:
                Gb_j = npj.zeros_like(Ga_j)
            G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
            projected_energy = self.projected_energy(self.ham, G_j, shift, beta_i)

            num_energy += (projected_energy * overlap).real
            denom += overlap.real

        energy = num_energy / denom
        return energy.real

    def get_args(self):
        return ()

    @plum.dispatch
    def projected_energy(self, ham: HolsteinModel, G: list, shift, beta_i):
        rho = G[0].diagonal() + G[1].diagonal()
        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        phonon_contrib = ham.w0 * npj.sum(shift.conj() * beta_i)
        el_ph_contrib = -ham.g * npj.dot(rho, shift.conj() + beta_i)
        projected_energy = kinetic + el_ph_contrib + phonon_contrib
        return projected_energy

    ### not yet complexificated
    @plum.dispatch
    def projected_energy(self, ham: AcousticSSHModel, G: list, shift, beta_i):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

        X = shift + beta_i
        displ = npj.array(ham.X_connectivity).dot(X)
        displ_mat = npj.diag(displ[:-1], 1)
        displ_mat += displ_mat.T
        if ham.pbc:
            displ_mat = displ_mat.at[0, -1].set(displ[-1])
            displ_mat = displ_mat.at[-1, 0].set(displ[-1])

        tmp0 = ham.g_tensor * G[0] * displ_mat
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * displ_mat
        el_ph_contrib = jax.numpy.sum(tmp0)

        phonon_contrib = ham.w0 * jax.numpy.sum(shift * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def variational_energy(self, ham: BondSSHModel, G: list, shift, beta_i):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

        X = shift + beta_i
        tmp0 = ham.g_tensor * G[0] * X
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * X
        el_ph_contrib = 2 * jax.numpy.sum(tmp0)

        phonon_contrib = ham.w0 * jax.numpy.sum(shift * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

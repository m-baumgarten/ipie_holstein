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
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectronHole
import jax
import jax.numpy as npj
import plum

from ipie.addons.eph.trial_wavefunction.variational.el_hole.variational_el_hole import VariationalElectronHole


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


class ToyozawaVariationalElectronHole(VariationalElectronHole):
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
        super().__init__(shift_init, electron_init, hole_init, hamiltonian, system, cplx)
        self.K = K
        self.perms = circ_perm(np.arange(hamiltonian.nsites))
        self.nperms = self.perms.shape[0]
        self.Kcoeffs = np.exp(1j * K * np.arange(hamiltonian.nsites))

    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b, h0a, h0b = self.unpack_x(x)
        shift_abs = npj.abs(shift)

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):

            beta_i = shift[npj.array(permi)]
            beta_i_abs = npj.abs(beta_i)
            psia_i = c0a[permi, :]
            holea_i = h0a[permi, :]

            overlap = npj.linalg.det(c0a.conj().T.dot(psia_i)) * npj.prod(
                npj.exp(-0.5 * (shift_abs**2 + beta_i_abs**2) + shift.conj() * beta_i)
            )
            overlap *= npj.linalg.det(h0a.conj().T.dot(holea_i))
            if self.sys.ndown > 0:
                psib_i = c0b[permi, :]
                holeb_i = h0b[permi, :]
                overlap *= npj.linalg.det(c0b.conj().T.dot(psib_i))
                overlap *= npj.linalg.det(h0b.conj().T.dot(holeb_i))
            overlap *= self.Kcoeffs[0].conj() * coeffi

            if npj.abs(overlap) < zero_th:
                continue

            if ip != 0:
                overlap = overlap * (self.ham.nsites - ip) * 2
            else:
                overlap = overlap * self.ham.nsites

            # Evaluate Greens functions
            Gea_j = gab(c0a, psia_i)
            if self.sys.ndown > 0:
                Geb_j = gab(c0b, psib_i)
            else:
                Geb_j = npj.zeros_like(Gea_j)
            Ge_j = [Gea_j, Geb_j]
            
            Gha_j = gab(h0a, holea_i)
            if self.sys.ndown > 0:
                Ghb_j = gab(h0b, holeb_i)
            else:
                Ghb_j = npj.zeros_like(Gha_j)
            Gh_j = [Gha_j, Ghb_j]

            # Obtain projected energy of permuted soliton on original soliton
            projected_energy = self.projected_energy(self.ham, Ge_j, Gh_j, shift, beta_i)

            num_energy += (projected_energy * overlap).real
            denom += overlap.real

        energy = num_energy / denom
        return energy.real

    def get_args(self):
        return ()

    @plum.dispatch
    def projected_energy(self, ham: ExcitonPhononCavityElectronHole, Ge: list, Gh: list, shift, beta_i):
        kinetic = np.sum(ham.Te[0] * Ge[0] + ham.Te[1] * Ge[1]) 
        kinetic += np.sum(ham.Th[0] * Gh[0] + ham.Th[1] * Gh[1])

        el_el_contrib = np.sum(ham.quade[0] * Ge[0] + ham.quade[1] * Ge[1])
        h_h_contrib = np.sum(ham.quadh[0] * Gh[0] + ham.quadh[1] * Gh[1])
        el_h_contrib = -2 * npj.einsum('ij,kl->', ham.quade[0] * Ge[0] + ham.quade[1] * Ge[1], ham.quadh[0] * Gh[0] + ham.quadh[1] * Gh[1])
        quad_contrib = el_el_contrib + el_h_contrib + h_h_contrib

        el_ph_contrib = npj.einsum('ijk,ij,k->', ham.ge_tensor, Ge[0], shift.conj() + beta_i)
        el_ph_contrib += npj.einsum('ijk,ij,k->', ham.gh_tensor, Gh[0], shift.conj() + beta_i)
        if self.sys.ndown > 0:
            el_ph_contrib += npj.einsum('ijk,ij,k->', ham.ge_tensor, Ge[1], shift.conj() + beta_i)
            el_ph_contrib += npj.einsum('ijk,ij,k->', ham.gh_tensor, Gh[1], shift.conj() + beta_i)

        el_hole_contrib = npj.einsum('ijkl,ij,kl->', ham.elhole_tensor, Ge[0], Gh[0])
        if self.sys.ndown > 0:
            el_hole_contrib += npj.einsum('ijkl,ij,kl->', ham.elhole_tensor, Ge[1], Gh[1])

        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib + quad_contrib + el_hole_contrib
        return local_energy
 

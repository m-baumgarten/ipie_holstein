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
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectron, ExcitonPhononCavityHole
import jax
import jax.numpy as npj
import plum

from ipie.addons.eph.trial_wavefunction.variational.variational import Variational

def circ_perm_1D(sites: Union[int, np.ndarray]) -> np.ndarray:
    sites = np.arange(sites)
    circs = sites
    for shift in range(1, len(sites)):
        new_circ = np.roll(sites, -shift)
        circs = np.vstack([circs, new_circ])
    return circs

def circ_perm(hamiltonian, k) -> np.ndarray:
    """Returns a matrix which rows consist of all possible
    cyclic permutations given an initial array lst.

    Parameters
    ----------
    lst :
        Initial array which is to be cyclically permuted
    """
    nsites = hamiltonian.nsites
    perms = np.zeros((hamiltonian.N, hamiltonian.N), dtype=np.int32)
    lattice = np.arange(hamiltonian.N, dtype=np.int32).reshape(nsites)

    kcoeffs = np.ones(hamiltonian.N, dtype=np.complex128)
    if hamiltonian.dim == 1:
        perms = circ_perm_1D(nsites[0])
        kcoeffs = np.exp(1j * np.arange(hamiltonian.N) * k[0])

    elif hamiltonian.dim == 2:
        perms_x = circ_perm_1D(nsites[0])
        perms_y = circ_perm_1D(nsites[1])
        for xi, perm_x in enumerate(perms_x):
            for yi, perm_y in enumerate(perms_y):
                index = xi * nsites[1] + yi
                kcoeffs[index] = np.exp(1j * (xi * k[0] + yi * k[1]))
                print(kcoeffs)
                perms[index, :] = lattice[perm_x, :][:, perm_y].reshape(hamiltonian.N).astype(np.int32)

    elif hamiltonian.dim == 3:
        perms_x = circ_perm_1D(nsites[0])
        perms_y = circ_perm_1D(nsites[1])
        perms_z = circ_perm_1D(nsites[2])
        for xi, perm_x in enumerate(perms_x):
            for yi, perm_y in enumerate(perms_y):
                for zi, perm_z in enumerate(perms_z):
                    index = xi * nsites[1] * nsites[2] + yi * nsites[2] + zi
                    perms[index, :] = lattice[perm_x, perm_y, perm_z].reshape(hamiltonian.N).astype(np.int32)

    return perms, kcoeffs

def get_kcoeffs(hamiltonian, K):
    """"""
    nsites = hamiltonian.nsites

    if hamiltonian.dim == 1:
        exponent = 1j * K[0] * np.arange(hamiltonian.nsites[0])

    elif hamiltonian.dim == 2:
        coeff_x = K[0] * np.arange(hamiltonian.nsites[0])
        coeff_y = K[1] * np.arange(hamiltonian.nsites[1])
        # TODO do this is numpy.add.outer and then ravel
        exponent = np.array([1j * (cx + cy) for cx in coeff_x for cy in coeff_y])

    elif hamiltonian.dim == 3:
        coeff_x = K[0] * np.arange(hamiltonian.nsites[0])
        coeff_y = K[1] * np.arange(hamiltonian.nsites[1])
        coeff_z = K[2] * np.arange(hamiltonian.nsites[2])
        # TODO do this with numpy.add.outer, ravel, do outer again, and ravel again
        exponent = np.array([1j * (cx + cy + cz) for cx in coeff_x for cy in coeff_y for cz in coeff_z])

    Kcoeffs = np.exp(exponent)
    return Kcoeffs

def overlap_degeneracy(hamiltonian, index):
    if hamiltonian.dim == 1:
        if index != 0:
            degeneracy = (hamiltonian.N - index) * 2
        else:
            degeneracy = hamiltonian.N

    if hamiltonian.dim == 2:
        index_x, index_y = divmod(index, hamiltonian.nsites[1])
        if index_x != 0 or index_y !=0:
            if index_y == 0 or index_x == 0:
                degeneracy = (hamiltonian.nsites[0] - index_x) * (hamiltonian.nsites[1] - index_y) * 2
            else:
                degeneracy = (hamiltonian.nsites[0] - index_x) * (hamiltonian.nsites[1] - index_y) * 4
        else:
            degeneracy = hamiltonian.N

    if hamiltonian.dim == 3:
        index_remainder, index_c = divmod(index, hamiltonian.nsites[2])
        index_x, index_y = divmod(index_remainder, hamiltonian.nsites[1])
        # TODO is this correct?
        if index_x != 0 or index_y !=0 or index_z !=0:
            degeneracy = (hamiltonian.N - index_x) * (hamiltonian.N - index_y) * (hamiltonian.N - index_z) * 2
        else:
            degeneracy = hamiltonian.N

    return degeneracy

#def circ_perm(lst: np.ndarray) -> np.ndarray:
#    """Returns a matrix which rows consist of all possible
#    cyclic permutations given an initial array lst.
#
#    Parameters
#    ----------
#    lst :
#        Initial array which is to be cyclically permuted
#    """
#    circs = lst
#    for shift in range(1, len(lst)):
#        new_circ = np.roll(lst, -shift)
#        circs = np.vstack([circs, new_circ])
#    return circs


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
#        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)
#        self.K = K
#        self.perms = circ_perm(np.arange(hamiltonian.nsites))
#        self.nperms = self.perms.shape[0]
#        self.Kcoeffs = np.exp(1j * K * np.arange(hamiltonian.nsites))
        
        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)
        if isinstance(K, float):
            self.K = np.array([K])
        else:
            self.K = K
        self.perms, kcoeff = circ_perm(hamiltonian, self.K)
        self.nperms = self.perms.shape[0]
        self.Kcoeffs = get_kcoeffs(hamiltonian, self.K)
        assert np.all(kcoeff == self.Kcoeffs)

    def get_args(self):
        return ()
    
    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
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

#            if ip != 0:
#                overlap = overlap * (self.ham.nsites - ip) * 2
#            else:
#                overlap = overlap * self.ham.nsites
            overlap *= overlap_degeneracy(self.ham, ip)

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

    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift, beta_i):
        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        el_ph_contrib = npj.einsum('ijk,ij,k->', ham.g_tensor, G[0], shift.conj() + beta_i)
        if self.sys.ndown > 0:
            el_ph_contrib += npj.einsum('ijk,ij,k->', ham.g_tensor, G[1], shift.conj() + beta_i)
        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def projected_energy(self, ham: Union[ExcitonPhononCavityElectron, ExcitonPhononCavityHole], G: list, shift, beta_i):
        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        ferm_ferm_contrib = np.sum(ham.quad[0] * G[0] + ham.quad[1] * G[1])
        el_ph_contrib = npj.einsum('ijk,ij,k->', ham.g_tensor, G[0], shift.conj() + beta_i)
        if self.sys.ndown > 0:
            el_ph_contrib += npj.einsum('ijk,ij,k->', ham.g_tensor, G[1], shift.conj() + beta_i)
        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib + ferm_ferm_contrib
        return local_energy

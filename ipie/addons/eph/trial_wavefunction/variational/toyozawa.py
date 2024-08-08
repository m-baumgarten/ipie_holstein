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
        exponent = np.array([1j * (cx + cy) for cx in coeff_x for cy in coeff_y])

    elif hamiltonian.dim == 3:
        coeff_x = K[0] * np.arange(hamiltonian.nsites[0])
        coeff_y = K[1] * np.arange(hamiltonian.nsites[1])
        coeff_z = K[2] * np.arange(hamiltonian.nsites[2])
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
        if index_x != 0 or index_y !=0 or index_z !=0:
            degeneracy = (hamiltonian.N - index_x) * (hamiltonian.N - index_y) * (hamiltonian.N - index_z) * 2
        else:
            degeneracy = hamiltonian.N 

    return degeneracy


class ToyozawaVariational(Variational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        K: Union[float, np.ndarray],
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)
        if isinstance(K, float):
#            print(K)
            self.K = np.array([K])
#            print('ey')
        else:
            self.K = K
#        print(self.K)
#        exit()
        self.perms, kcoeff = circ_perm(hamiltonian, self.K)
#        print(self.perms)
        self.nperms = self.perms.shape[0]
        self.Kcoeffs = get_kcoeffs(hamiltonian, self.K)
        assert np.all(kcoeff == self.Kcoeffs)

    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
        shift_abs = npj.abs(shift)

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
#            jax.debug.print('perms_ {x}', x=[permi, self.perms, type(permi)])
            beta_i = shift[permi]
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
#        jax.debug.print('energy:    {x}', x=energy)
        return energy.real
    
    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift_abs = npj.abs(shift)
#        jax.debug.print('shift {x}', x=shift.shape)
        shift = npj.squeeze(shift)

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            beta_i = shift[permi]
            beta_i_abs = npj.abs(beta_i)
            psia_i = c0a[permi, :]

            for jp, (permj, coeffj) in enumerate(zip(self.perms, self.Kcoeffs)):
                beta_j = shift[permj]
                beta_j_abs = npj.abs(beta_j)
                psia_j = c0a[permj, :]

                overlap = npj.linalg.det(psia_i.conj().T.dot(psia_j)) * npj.prod(
                    npj.exp(-0.5 * (beta_i_abs**2 + beta_j_abs**2) + beta_i.conj() * beta_j)
                )
                if self.sys.ndown > 0:
                    psib_i = c0b[permi, :]
                    psib_j = c0b[permj, :]
                    overlap *= npj.linalg.det(psib_i.conj().T.dot(psib_j))
                overlap *= coeffi.conj() * coeffj

                if npj.abs(overlap) < zero_th:
                    continue

#                if ip != 0:
#                    overlap = overlap * (self.ham.nsites - ip) * 2
#                else:
#                    overlap = overlap * self.ham.nsites

            # Evaluate Greens functions
                Ga_ij = gab(psia_i, psia_j)
                if self.sys.ndown > 0:
                    Gb_ij = gab(psib_i, psib_j)
                else:
                    Gb_ij = npj.zeros_like(Ga_ij)
                G_ij = [Ga_ij, Gb_ij]

            # Obtain projected energy of permuted soliton on original soliton
                projected_energy = self.projected_energy(self.ham, G_ij, beta_i, beta_j)
    
#                num_energy += (projected_energy * overlap).real
#                denom += overlap.real
                num_energy += projected_energy * overlap
                denom += overlap
#                jax.debug.print('perms:     {x}', x=(ip,jp))
#                jax.debug.print('energy_    {x}', x=projected_energy * overlap)
                jax.debug.print('overlap:   {x}', x=[ip,jp,overlap])
        energy = num_energy / denom
#        npj.save('ovlp.npy', ovlp)
#        npj.save('energy.npy', energy)
        jax.debug.print('energy:    {x}', x=energy)       
        return energy.real

    def get_args(self):
        return ()

    @plum.dispatch
    def projected_energy(self, ham: HolsteinModel, G: list, shift, beta_i):
        rho = G[0].diagonal() + G[1].diagonal()
        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        phonon_contrib = ham.w0 * npj.sum(shift.conj() * beta_i)
        el_ph_contrib = -ham.g * npj.sum(npj.dot(rho, shift.conj() + beta_i))
        projected_energy = kinetic + el_ph_contrib + phonon_contrib
        return projected_energy

    @plum.dispatch
    def projected_energy(self, ham: AcousticSSHModel, G: list, shift, beta_i):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

        X = shift.conj() + beta_i
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

        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def variational_energy(self, ham: BondSSHModel, G: list, shift, beta_i):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

        X = shift.conj() + beta_i
        tmp0 = ham.g_tensor * G[0] * X
        if self.sys.ndown > 0:
            tmp0 += ham.g_tensor * G[1] * X
        el_ph_contrib = 2 * jax.numpy.sum(tmp0)

        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

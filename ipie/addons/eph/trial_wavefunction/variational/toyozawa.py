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
        
#        self.perms = [self.perms[0]]
#        self.nperms = 1
#        self.Kcoeffs = np.array([1.])
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
            
#            print(f'energy I {ip}:    ', (projected_energy * overlap).real, overlap.real)

#        print('energ I:  ', num_energy, denom)
        energy = num_energy / denom
        return energy.real

    def d_xAx_d_x(self, A, x):
        return (A + A.conj().T).dot(x)

    def _gradient(self, x, *args) -> np.ndarray:
        """For GenericEPhModel"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        c0a = np.squeeze(c0a)
        shift_abs = np.abs(shift)

        
        shift_grad_real = np.zeros_like(shift)
        shift_grad_imag = np.zeros_like(shift)
        psia_grad_real = np.zeros_like(c0a)
        psia_grad_imag = np.zeros_like(c0a)
     
        shift_grad_real_ovlp = np.zeros_like(shift)
        shift_grad_imag_ovlp = np.zeros_like(shift)
        psia_grad_real_ovlp = np.zeros_like(c0a)
        psia_grad_imag_ovlp = np.zeros_like(c0a)

        #TODO get total energy and overlap
        ovlp = 0.
        energy = 0.

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            fac_i = overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi
            
            beta_i = shift[permi]
            psia_i = c0a[permi] # [permi, :]
            
            perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            perm_mat_symm = perm_mat + perm_mat.conj().T
            perm_mat_asym = perm_mat - perm_mat.conj().T

            Ga_i = np.outer(c0a.conj(), psia_i)
            occ = np.sum(Ga_i.diagonal())
            
            kin = np.sum(Ga_i * self.ham.T[0])
            kin_perm = self.ham.T[0].dot(perm_mat)
            el_ph_c = np.einsum('ijk,ij->k', self.ham.g_tensor, Ga_i)
            el_ph_c_perm = np.einsum('k,km->m', el_ph_c, perm_mat)
            
            cs_ovlp = np.exp(np.dot(shift.conj(), beta_i) - np.sum(shift_abs**2)) #verify dot works here
            ovlp_i = occ * cs_ovlp
            tot_energy = (kin + el_ph_c.dot(shift.conj() + beta_i) + occ * shift.conj().dot(beta_i)) * cs_ovlp

            d_cs_ovlp_r = cs_ovlp * (perm_mat_symm.dot(shift.real) + 1j * perm_mat_asym.dot(shift.imag) - 2 * shift.real)
            d_cs_ovlp_i = cs_ovlp * (perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real) - 2 * shift.imag)
            

            # shift_grad_real contribs
            kin_contrib = kin * d_cs_ovlp_r
            el_ph_contrib = d_cs_ovlp_r * np.sum(el_ph_c * (shift.conj() + beta_i)) + cs_ovlp * (el_ph_c + el_ph_c_perm)
            boson_contrib = d_cs_ovlp_r * np.sum(shift.conj() * beta_i) + cs_ovlp * (perm_mat_symm.dot(shift.real) + 1j * perm_mat_asym.dot(shift.imag))
            boson_contrib *= occ * self.ham.w0
            ovlp_contrib = d_cs_ovlp_r * occ
            sgr = (kin_contrib + el_ph_contrib + boson_contrib)
            sgr_ovlp = ovlp_contrib 

            # shift_grad_imag contribs
            kin_contrib = kin * d_cs_ovlp_i
            el_ph_contrib = d_cs_ovlp_i * np.sum(el_ph_c * (shift.conj() + beta_i)) +  1j * cs_ovlp * (el_ph_c_perm - el_ph_c)
            boson_contrib = d_cs_ovlp_i * np.sum(shift.conj() * beta_i) + cs_ovlp * (perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real))
            boson_contrib *= occ * self.ham.w0
            ovlp_contrib = d_cs_ovlp_i * occ
            sgi = (kin_contrib + el_ph_contrib + boson_contrib)
            sgi_ovlp = ovlp_contrib


    # BUG HERE TODO, in both diffs wrt imag and real -> bug in cs_ovlp?

            # psia_grad_real contribs
            g_contracted = np.einsum('ijk,k->ij', self.ham.g_tensor, shift.conj() + beta_i).dot(perm_mat) # TODO FIX
            
            kin_contrib = cs_ovlp * ((kin_perm + kin_perm.conj().T).dot(c0a.real) + 1j * (kin_perm - kin_perm.conj().T).dot(c0a.imag))
            el_ph_contrib = cs_ovlp * ((g_contracted + g_contracted.T).dot(c0a.real) + 1j * (g_contracted - g_contracted.T).dot(c0a.imag))
            boson_contrib = self.ham.w0 * cs_ovlp * shift.conj().dot(beta_i) * (perm_mat_symm.dot(c0a.real) + 1j * perm_mat_asym.dot(c0a.imag))
            ovlp_contrib = cs_ovlp * (perm_mat_symm.dot(c0a.real) + 1j * perm_mat_asym.dot(c0a.imag))
            pgr = (kin_contrib + el_ph_contrib + boson_contrib)
            pgr_ovlp = ovlp_contrib

            # psia_grad_imag contribs
            kin_contrib = cs_ovlp * ((kin_perm + kin_perm.conj().T).dot(c0a.imag) - 1j * (kin_perm - kin_perm.conj().T).dot(c0a.real))
            el_ph_contrib = cs_ovlp * ((g_contracted + g_contracted.T).dot(c0a.imag) - 1j * (g_contracted - g_contracted.T).dot(c0a.real))
            boson_contrib = self.ham.w0 * cs_ovlp * shift.conj().dot(beta_i) * (perm_mat_symm.dot(c0a.imag) - 1j * perm_mat_asym.dot(c0a.real))
            ovlp_contrib = cs_ovlp * (perm_mat_symm.dot(c0a.imag) - 1j * perm_mat_asym.dot(c0a.real))
            pgi = (kin_contrib + el_ph_contrib + boson_contrib)
            pgi_ovlp = ovlp_contrib

            # Accumulate
            shift_grad_real += (fac_i * sgr).real
            shift_grad_imag += (fac_i * sgi).real
            psia_grad_real += (fac_i * pgr).real
            psia_grad_imag += (fac_i * pgi).real
            
            shift_grad_real_ovlp += (fac_i * sgr_ovlp).real
            shift_grad_imag_ovlp += (fac_i * sgi_ovlp).real
            psia_grad_real_ovlp += (fac_i * pgr_ovlp).real
            psia_grad_imag_ovlp += (fac_i * pgi_ovlp).real

#            print(f'raw overlap {ip}: ', ovlp_i, cs_ovlp)
            energy += (fac_i * cs_ovlp * (kin + np.sum(np.einsum('ijk,k->ij', self.ham.g_tensor, shift.conj() + beta_i) * Ga_i) + self.ham.w0 * occ * shift.conj().dot(beta_i))).real
            ovlp += (fac_i * ovlp_i).real
        
#        print('energy:  ', energy, ovlp)

        dx_energy = np.hstack([shift_grad_real, shift_grad_imag, psia_grad_real, psia_grad_imag]).astype(np.float64)        
        dx_ovlp = np.hstack([shift_grad_real_ovlp, shift_grad_imag_ovlp, psia_grad_real_ovlp, psia_grad_imag_ovlp]).astype(np.float64)
#        print('dx energy and ovlp', np.hstack([shift_grad_real, shift_grad_imag, psia_grad_real, psia_grad_imag]), np.hstack([shift_grad_real_ovlp, shift_grad_imag_ovlp, psia_grad_real_ovlp, psia_grad_imag_ovlp]))
        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp ** 2
#        print('my grad: ', dx)
#        exit()
#        super().gradient(x)
        return dx

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

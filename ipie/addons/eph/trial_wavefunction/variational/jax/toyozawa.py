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
#import plum

from ipie.addons.eph.trial_wavefunction.variational.jax.variational import Variational
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational as dd2

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
                    perms[index, :] = lattice[perm_x, :, :][:, perm_y, :][:, :, perm_z].reshape(hamiltonian.N).astype(np.int32)
                    kcoeffs[index] = np.exp(1j * (xi * k[0] + yi * k[1] + zi * k[2]))

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

def overlap_degeneracy_1D(hamiltonian, index, nsites):
    if index != 0:
        degeneracy = (nsites - index) * 2
    else:
        degeneracy = nsites
    return degeneracy


def overlap_degeneracy(hamiltonian, index):
    if hamiltonian.dim == 1:
        degeneracy = overlap_degeneracy_1D(hamiltonian, index, hamiltonian.nsites[0])

    if hamiltonian.dim == 2:
#        index_x, index_y = divmod(index, hamiltonian.nsites[1])
#        degeneracy = overlap_degeneracy_1D(hamiltonian, index_x, hamiltonian.nsites[0]) * overlap_degeneracy_1D(hamiltonian, index_y, hamiltonian.nsites[1])
        index_x, index_y = divmod(index, hamiltonian.nsites[1])
        if index_x == 0 and index_y == 0:
            degeneracy = hamiltonian.N
        elif index_x == 0 and index_y != 0:
            degeneracy = hamiltonian.nsites[0] * overlap_degeneracy_1D(hamiltonian, index_y, hamiltonian.nsites[1])
        elif index_x != 0 and index_y == 0:
            degeneracy = hamiltonian.nsites[1] * overlap_degeneracy_1D(hamiltonian, index_x, hamiltonian.nsites[0])
        elif index_x != 0 and index_y != 0:
            degeneracy = 2 * (hamiltonian.nsites[1] - index_y) * (hamiltonian.nsites[0] - index_x)

#        degeneracy = hamiltonian.N - index_y * hamiltonian.nsites[0]
#        if index_y != 0:
#           degeneracy *= 2

#        degeneracy = hamiltonian.N
#        if index_x == 0:
#            degeneracy = hamiltonian.N - hamiltonian.nsites[0] * index_y
#        else:
#            degeneracy = hamiltonian.N - hamiltonian.nsites[1] * index_x 

    if hamiltonian.dim == 3:
        index_remainder, index_z = divmod(index, hamiltonian.nsites[2])
        index_x, index_y = divmod(index_remainder, hamiltonian.nsites[1])
        # TODO is this correct?
        degeneracy = overlap_degeneracy_1D(hamiltonian, index_x, hamiltonian.nsites[0]) * overlap_degeneracy_1D(hamiltonian, index_y, hamiltonian.nsites[1]) * overlap_degeneracy_1D(hamiltonian, index_y, hamiltonian.nsites[2])
#        if index_x != 0 or index_y != 0 or index_z != 0:
#            degeneracy = (
#                (hamiltonian.N - index_x)
#                * (hamiltonian.N - index_y)
#                * (hamiltonian.N - index_z)
#                * 2
#            )
#        else:
#            degeneracy = hamiltonian.N

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
        self.analytical = dd2(shift_init, electron_init, hamiltonian, system, K, cplx)

#        self.perms = [self.perms[0]]
#        self.nperms = 1
#        self.Kcoeffs = np.array([1.])
        assert np.all(kcoeff == self.Kcoeffs)

    def get_args(self):
        return ()
   
    def objective_function_2D(self, x) -> float:
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
        shift_abs = npj.abs(shift)

#        num_energy = 0.0
#        denom = 0.0

        en = npj.zeros((self.ham.N), dtype=npj.complex128)
        ov = npj.zeros((self.ham.N), dtype=npj.complex128)
        co = npj.zeros((self.ham.N, self.ham.N), dtype=npj.complex128)
        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):

            beta_i = shift[npj.array(permi)]
            beta_i_abs = npj.abs(beta_i)
            psia_i = c0a[permi, :]

            cs_ovlp = self.cs_overlap(shift, beta_i)
            overlap = npj.linalg.det(c0a.conj().T.dot(psia_i)) * cs_ovlp #npj.prod(
#                npj.exp(-0.5 * (shift_abs**2 + beta_i_abs**2) + shift.conj() * beta_i)
#            )

            if self.sys.ndown > 0:
                psib_i = c0b[permi, :]
                overlap *= npj.linalg.det(c0b.conj().T.dot(psib_i))
            
            ov = ov.at[ip].set(overlap)
#            overlap *= self.Kcoeffs[0].conj() * coeffi
#            co = co.at[0,ip].set(self.Kcoeffs[0].conj() * coeffi)

#            overlap *= overlap_degeneracy(self.ham, ip)
#            overlap *= self.ham.N

            # Evaluate Greens functions
            Ga_j = gab(c0a, psia_i)
            if self.sys.ndown > 0:
                Gb_j = gab(c0b, psib_i)
            else:
                Gb_j = npj.zeros_like(Ga_j)
            G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
#            jax.debug.print('ovlp {x}', x=(ip,overlap, overlap_degeneracy(self.ham, ip)))
            projected_energy = self.projected_energy(self.ham, G_j, shift, beta_i)
#            num_energy += (projected_energy * overlap).real
#            denom += overlap.real

#            jax.debug.print('en: {x}', x=(ip, projected_energy, self.Kcoeffs[0].conj() * coeffi, self.ham.N)) #overlap_degeneracy(self.ham, ip)))
            en = en.at[ip].set(projected_energy)
        
        for ip, coeffi in enumerate(self.Kcoeffs):
            for jp, coeffj in enumerate(self.Kcoeffs):
                co = co.at[jp,ip].set(coeffj.conj() * coeffi)
       
        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
#            new = en[0,:] #anitperm
#            en = en.at[ip,:].set(new)
#            ov = ov.at[ip,:].set(ov[0,:])
            co = co.at[ip,:].set(co[ip,permi])
#            if ip == 3:
#                jax.debug.print('new:   {x}', x=(new, en[3,:]))
#                exit()
#        energy_2 = npj.sum(en * ov * co, axis=(0,1)) / npj.sum(ov * co, axis=(0,1))
        numer = npj.einsum('i,i,ji->',en, ov, co)        
        denom = npj.einsum('i,ji->', ov, co)
#        numer = 1.
        energy = numer / denom
        return energy.real

#        if True:
#            energy = num_energy / denom
#            return energy.real,en,ov,co,energy_2

#        for ip_x, (permi_x, coeffi_x) in enumerate(zip(self.perms[self.ham.nsites[1]::self.ham.nsites[1]], self.Kcoeffs[self.ham.nsites[1]::self.ham.nsites[1]])):
#            beta_i = shift[npj.array(permi_x)]
#            beta_i_abs = npj.abs(beta_i)
#            psia_i = c0a[permi_x, :]
#
#            for jp_y, (permj_y, coeffj_y) in enumerate(zip(self.perms[1:self.ham.nsites[1]], self.Kcoeffs[1:self.ham.nsites[1]])):
#                beta_j = shift[npj.array(permj_y)]
#                beta_j_abs = npj.abs(beta_j)
#                psia_j = c0a[permj_y, :]
#                    
#                cs_ovlp = self.cs_overlap(beta_j, beta_i)
#                overlap = npj.linalg.det(psia_j.conj().T.dot(psia_i)) * cs_ovlp #* npj.prod(npj.exp(beta_j.conj() * beta_i))
#
#                overlap *= coeffj_y.conj() * coeffi_x
#                dgen = 2 * (self.ham.nsites[1] - 1 - jp_y) * (self.ham.nsites[0] - 1 - ip_x)
#                overlap *= dgen
#                # Evaluate Greens functions
#                Ga_j = gab(psia_j, psia_i)
#                Gb_j = npj.zeros_like(Ga_j)
#                G_j = [Ga_j, Gb_j]
#        
#                # Obtain projected energy of permuted soliton on original soliton
#                projected_energy = self.projected_energy(self.ham, G_j, beta_j, beta_i)
#                num_energy += (projected_energy * overlap).real
#                denom += overlap.real
#
#                jax.debug.print('en: {x}', x=(ip_x, jp_y, projected_energy, coeffj_y.conj() * coeffi_x, dgen))    
#
#        energy = num_energy / denom
#        return energy.real
#

    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
        shift_abs = npj.abs(shift)

        num_energy = 0.0
        denom = 0.0
        ov = npj.zeros((self.ham.N), dtype=npj.complex128)
        en = npj.zeros((self.ham.N), dtype=npj.complex128)
        co = npj.zeros((self.ham.N), dtype=npj.complex128)
        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            
            beta_i = shift[npj.array(permi)]
            beta_i_abs = npj.abs(beta_i)
            psia_i = c0a[permi, :]

            cs_ovlp = self.cs_overlap(shift, beta_i)
            overlap = npj.linalg.det(c0a.conj().T.dot(psia_i)) * cs_ovlp #npj.prod(
#                npj.exp(-0.5 * (shift_abs**2 + beta_i_abs**2) + shift.conj() * beta_i)
#            )
            ov = ov.at[ip].set((overlap))
            if self.sys.ndown > 0:
                psib_i = c0b[permi, :]
                overlap *= npj.linalg.det(c0b.conj().T.dot(psib_i))
            overlap *= self.Kcoeffs[0].conj() * coeffi

            co = co.at[ip].set((coeffi))
            overlap *= overlap_degeneracy(self.ham, ip)

            # Evaluate Greens functions
            Ga_j = gab(c0a, psia_i)
            if self.sys.ndown > 0:
                Gb_j = gab(c0b, psib_i)
            else:
                Gb_j = npj.zeros_like(Ga_j)
            G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
#            jax.debug.print('ovlp {x}', x=(ip,overlap, overlap_degeneracy(self.ham, ip)))
            projected_energy = self.projected_energy(self.ham, G_j, shift, beta_i)
            num_energy += (projected_energy * overlap).real
            denom += overlap.real
            en = en.at[ip].set((projected_energy))
#            jax.debug.print('ov: {x}', x=(ip, overlap))
#            jax.debug.print('en: {x}', x=(ip, projected_energy, overlap_degeneracy(self.ham, ip)))
#        jax.debug.print('denom: {x}', x=denom)
#        energy = npj.sum(en * ov * self.Kcoeffs) / npj.sum(ov * self.Kcoeffs)
        jax.debug.print('lin en mat:    {x}', x=(en))
        jax.debug.print('lin ov mat:    {x}', x=(ov))
        jax.debug.print('lin co mat:    {x}', x=(self.Kcoeffs))
        
#        for ip, perm in enumerate(self.perms):
#            en = en.at[ip, :].set(en[0, :][perm])
#            ov = ov.at[ip, :].set(ov[0, :][perm])
#            co = co.at[ip, :].set(co[0, :][perm])
#        energy = npj.sum(en * ov * co) / npj.sum(ov * co)
        energy = num_energy / denom
        return energy.real

    def cs_overlap(self, shift_i, shift_j) -> float:
        cs_ovlp_log = npj.sum(
            -0.5 * (npj.abs(shift_i) ** 2 + npj.abs(shift_j) ** 2) + shift_i.conj() * shift_j
        )
        return npj.exp(cs_ovlp_log)


    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
#        jax.debug.print('coeffs:    {x}', x=(self.Kcoeffs))
        shift = npj.squeeze(shift) # get rid of number params col dimension
        shift_abs = npj.abs(shift)

        num_energy = 0.0
        denom = 0.0

        ov = npj.zeros((self.ham.N, self.ham.N), dtype=npj.complex128)
        en = npj.zeros((self.ham.N, self.ham.N), dtype=npj.complex128)
        co = npj.zeros((self.ham.N, self.ham.N), dtype=npj.complex128)
        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):

            beta_i = shift[npj.array(permi)]
            beta_i_abs = npj.abs(beta_i)
            psia_i = c0a[permi, :]

            for jp, (permj, coeffj) in enumerate(zip(self.perms, self.Kcoeffs)):

                beta_j = shift[npj.array(permj)]
                beta_j_abs = npj.abs(beta_j)
                psia_j = c0a[permj, :]

                cs_ovlp = self.cs_overlap(beta_j, beta_i)
                overlap = npj.linalg.det(psia_j.conj().T.dot(psia_i)) * cs_ovlp #* npj.prod(npj.exp(beta_j.conj() * beta_i))

                ov = ov.at[jp,ip].set((overlap))

                overlap *= coeffj.conj() * coeffi

                co = co.at[jp,ip].set((coeffj.conj() * coeffi))

                # Evaluate Greens functions
#                Ga_j = gab(psia_j, psia_i)
                Ga_j = npj.outer(psia_j.conj(), psia_i) / npj.sum(psia_j.conj() * psia_i, axis=(0,1))
                Gb_j = npj.zeros_like(Ga_j)
                G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
                projected_energy = self.projected_energy(self.ham, G_j, beta_j, beta_i)
#                jax.debug.print('ovlp {x}', x=(jp,ip,projected_energy))
                num_energy += projected_energy * overlap
                denom += overlap
                en = en.at[jp, ip].set((projected_energy))
#                if jp == 3:
#                    jax.debug.print('ov jp ip with coeff: {x}', x=(jp, ip, projected_energy, coeffj.conj() * coeffi))

        occ, freq = npj.unique(npj.round(ov.ravel(),13), return_counts=True)
        occ_en, freq_en = npj.unique(npj.round(en.ravel(),13), return_counts=True)
        occ_co, freq_co = npj.unique(npj.round(co.ravel(),13), return_counts=True)
        occfreq = npj.vstack([occ, freq]).T
        occfreq_en = npj.vstack([occ_en, freq_en]).T
        occfreq_co = npj.vstack([occ_co, freq_co]).T

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            new = en[ip,permi] #anitperm
            en = en.at[ip,:].set(new)
            ov = ov.at[ip,:].set(ov[ip,permi])
            co = co.at[ip,:].set(co[ip,permi])


#        isherm = npj.abs(en - en.T.conj())
#        jax.debug.print("is herm:   {x}", x=isherm)
#        jax.debug.print('unique ov:  {x}', x=occfreq)
#        jax.debug.print('unique co:  {x}', x=occfreq_co)
#        jax.debug.print("unique en: {x}", x=occfreq_en)
        energy = num_energy / denom
      
#        en_test = npj.sum(en * ov * co, axis=(0,1)) / npj.sum(ov * co, axis=(0,1))
#        jax.debug.print("energy diff mat:   {x}", x=(npj.abs(en_test - energy)))
#        en_raw = npj.sum(en[:5] * ov[:5] * co[:5]) / npj.sum(ov[:5] * co[:5])

#        energy_mat = npj.sum(en * ov * co) / npj.sum(ov * co)
        #if self.ham.dim == 2:
        en3  = self.objective_function_2D(x)
        #else:
        #    en2 = self._objective_function(x)
 #       jax.debug.print('numer: {x}', x=num_energy)
        
#        jax.debug.print('en mat:    {x}', x=(en))
#        jax.debug.print('en mat 2d:    {x}', x=(en_mat2))
#        jax.debug.print('ov mat:    {x}', x=(ov))
#        jax.debug.print('co mat:    {x}', x=(co))
        jax.debug.print("diff:  {x}", x=npj.abs(en3 - energy))
#        jax.debug.print('diff: {x}', x=(npj.abs(en2 - energy), en2, energy, num_energy, denom, npj.max(npj.abs(en_mat2 - en)), npj.max(npj.abs(ovmat2 - ov)), npj.abs(comat2.T - co), npj.abs(en3 - energy)))
#        jax.debug.print('phases: {x}',x=(comat2[0,15], ,co[0,15]))
#        jax.debug.print('diff mat {x}', x=(npj.abs(energy_mat - energy)))
#        jax.debug.print('diff row {x}', x=(en_raw - energy))
        return energy.real

#    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift, beta_i):
        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        tmp = npj.einsum('ijk,ij,k->i', ham.g_tensor, G[0], shift.conj() + beta_i)
#        jax.debug.print("contribs:  {x}", x=tmp)
        el_ph_contrib = npj.einsum('ijk,ij,k->', ham.g_tensor, G[0], shift.conj() + beta_i)
        if self.sys.ndown > 0:
            el_ph_contrib += npj.einsum('ijk,ij,k->', ham.g_tensor, G[1], shift.conj() + beta_i)
        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib
#        jax.debug.print('energy contrib:  {x}', x=(kinetic, el_ph_contrib, phonon_contrib))
#        jax.debug.print('energy:  {x}', x=local_energy)
        return local_energy

#    @plum.dispatch
#    def projected_energy(self, ham: Union[ExcitonPhononCavityElectron, ExcitonPhononCavityHole], G: list, shift, beta_i):
#        kinetic = npj.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
#        ferm_ferm_contrib = np.sum(ham.quad[0] * G[0] + ham.quad[1] * G[1])
#        el_ph_contrib = npj.einsum('ijk,ij,k->', ham.g_tensor, G[0], shift.conj() + beta_i)
#        if self.sys.ndown > 0:
#            el_ph_contrib += npj.einsum('ijk,ij,k->', ham.g_tensor, G[1], shift.conj() + beta_i)
#        phonon_contrib = ham.w0 * jax.numpy.sum(shift.conj() * beta_i)
#        local_energy = kinetic + el_ph_contrib + phonon_contrib + ferm_ferm_contrib
#        return local_energy

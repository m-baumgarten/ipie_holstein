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
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import (
    ExcitonPhononCavityElectron,
    ExcitonPhononCavityHole,
)
#import plum

from ipie.addons.eph.trial_wavefunction.variational.variational import Variational
from numba import jit
from ipie.addons.eph.hamiltonians.ssh import OpticalSSHModel
from ipie.systems import Generic

def circ_perm_1D(sites: Union[int, np.ndarray]) -> np.ndarray:
    """Build the cyclic-permutation table used to assemble the Toyozawa /
    K-projected trial.

    Convention: row ip is the index list such that
        psia_translated[k] = c0a[perms[ip][k]] = c0a[(k - ip) mod N],
    i.e. it implements the *forward* translation T_{+ip}: site i -> i + ip.

    Combined with kcoeffs[ip] = exp(+i K ip) elsewhere in the trial, this
    builds
        |Psi_K> = sum_{ip} e^{i K ip} T_{+ip} |alpha, beta>
    so that "K" is the physical momentum quantum number.

    Historical note: a previous version used np.roll(sites, -shift), which
    produced T_{-ip} and made the trial labelled "K" actually represent
    |Psi_{-K}>. For 1D Holstein this was harmless (E(K)=E(-K) by inversion)
    but it broke wavefunction-level identifications and would have given
    wrong energies for non-inversion-symmetric models. See
    test_variational_fock_ed.py for the regression that pins this down.
    """
    sites = np.arange(sites)
    circs = sites
    for shift in range(1, len(sites)):
        new_circ = np.roll(sites, shift)
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
                perms[index, :] = (
                    lattice[perm_x, :][:, perm_y].reshape(hamiltonian.N).astype(np.int32)
                )

    elif hamiltonian.dim == 3:
        perms_x = circ_perm_1D(nsites[0])
        perms_y = circ_perm_1D(nsites[1])
        perms_z = circ_perm_1D(nsites[2])
        for xi, perm_x in enumerate(perms_x):
            for yi, perm_y in enumerate(perms_y):
                for zi, perm_z in enumerate(perms_z):
                    index = xi * nsites[1] * nsites[2] + yi * nsites[2] + zi
                    perms[index, :] = (
                            lattice[perm_x, :, :][:, perm_y, :][:, :,perm_z].reshape(hamiltonian.N).astype(np.int32)
                    )
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
        exponent = np.array(
            [1j * (cx + cy + cz) for cx in coeff_x for cy in coeff_y for cz in coeff_z]
        )

    Kcoeffs = np.exp(exponent)
    print("kcoeffs: ", Kcoeffs)
    print("K", K)
    #exit()
    return Kcoeffs


def overlap_degeneracy_1D(hamiltonian, index):
    if index != 0:
        degeneracy = (hamiltonian.N - index) * 2
    else:
        degeneracy = hamiltonian.N
    return degeneracy


def overlap_degeneracy(hamiltonian, index):
    if hamiltonian.dim == 1:
        degeneracy = overlap_degeneracy_1D(hamiltonian, index)

    if hamiltonian.dim == 2:
        index_x, index_y = divmod(index, hamiltonian.nsites[1])
        if index_x != 0 or index_y != 0:
            if index_y == 0 or index_x == 0:
                degeneracy = (
                    (hamiltonian.nsites[0] - index_x) * (hamiltonian.nsites[1] - index_y) * 2
                )
            else:
                degeneracy = (
                    (hamiltonian.nsites[0] - index_x) * (hamiltonian.nsites[1] - index_y) * 4
                )
        else:
            degeneracy = hamiltonian.N

    if hamiltonian.dim == 3:
        index_remainder, index_c = divmod(index, hamiltonian.nsites[2])
        index_x, index_y = divmod(index_remainder, hamiltonian.nsites[1])
        # TODO is this correct?
        if index_x != 0 or index_y != 0 or index_z != 0:
            degeneracy = (
                (hamiltonian.N - index_x)
                * (hamiltonian.N - index_y)
                * (hamiltonian.N - index_z)
                * 2
            )
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
        super().__init__(shift_init, electron_init, hamiltonian, system, cplx)
        if isinstance(K, float):
            self.K = np.array([K])
        else:
            self.K = K
        self.perms, kcoeff = circ_perm(hamiltonian, self.K)
        self.nperms = self.perms.shape[0]
        self.Kcoeffs = get_kcoeffs(hamiltonian, self.K)
        assert np.all(kcoeff == self.Kcoeffs)

        # Single-sum K-projection weight. With perms[ip] = T_{+ip} (see
        # circ_perm_1D), the energy is
        #     E_K = sum_ip e^{i K ip} <psi_0|H|psi_ip> / sum_ip e^{i K ip} <psi_0|psi_ip>,
        # so the per-ip weight is simply kcoeffs[ip]. The overall N-fold
        # multiplicity cancels in the numerator/denominator ratio. The
        # historical `coeff_perm`/`m_phases` machinery was tied to the
        # earlier (wrong-K) perm convention and became degenerate after
        # the convention fix.
        self.m_phases = self.Kcoeffs.copy()

    def get_args(self):
        return ()

    def cs_overlap(self, shift_i: np.ndarray, shift_j: np.ndarray) -> float:
        cs_ovlp_log = np.sum(
            -0.5 * (np.abs(shift_i) ** 2 + np.abs(shift_j) ** 2) + shift_i.conj() * shift_j,
        )
        return np.exp(cs_ovlp_log)

    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        shift_abs = np.abs(shift)

        num_energy = 0.0
        denom = 0.0

        
        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            fac_i = self.m_phases[ip] 
#            print("fac_i    ", ip, np.sum(coeff_perm[:,ip]), permi)
            beta_i = shift[np.array(permi)]
            beta_i_abs = np.abs(beta_i)
            psia_i = c0a[permi, :]

            cs_ovlp = self.cs_overlap(shift, beta_i)
            overlap = np.linalg.det(c0a.conj().T.dot(psia_i)) * cs_ovlp

            if self.sys.ndown > 0:
                psib_i = c0b[permi, :]
                overlap *= np.linalg.det(c0b.conj().T.dot(psib_i))
#            overlap *= self.Kcoeffs[0].conj() * coeffi

#            if np.abs(overlap) < zero_th:
#                continue

#            overlap *= overlap_degeneracy(self.ham, ip)

            # Evaluate Greens functions
            Ga_j = gab(c0a, psia_i)
            if self.sys.ndown > 0:
                Gb_j = gab(c0b, psib_i)
            else:
                Gb_j = np.zeros_like(Ga_j)
            G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
            projected_energy = self.projected_energy(self.ham, G_j, shift, beta_i)
            num_energy += (projected_energy * overlap * fac_i).real
            denom += (overlap * fac_i).real

#        num_energy = 1
#        exit()
        energy = num_energy / denom

        print("energy diff: ", self.objective_function_cubic(x) - energy.real)

        return energy.real

    def objective_function_cubic(self, x, zero_th: float = 1e-12) -> float:
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        shift_abs = np.abs(shift)

        num_energy = 0.0
        denom = 0.0


        betas = np.zeros((self.ham.N, self.ham.N), dtype=np.complex128)
        psis = np.zeros((self.ham.N, self.ham.N), dtype=np.complex128)
        cs_ovlps = np.zeros((self.ham.N), dtype=np.complex128)
        for ip, permi in enumerate(self.perms):
            betas[:, ip] = shift[np.array(permi)]
            psis[:, ip] = np.squeeze(c0a[permi]) #, :]
            cs_ovlps[ip] = self.cs_overlap(shift, betas[:, ip])


        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            fac_i = self.m_phases[ip]
#            print("fac_i    ", ip, np.sum(coeff_perm[:,ip]), permi)
            beta_i = betas[:, ip]
            beta_i_abs = np.abs(beta_i)
            psia_i = psis[:, ip][:,None]

            cs_ovlp = cs_ovlps[ip] #self.cs_overlap(shift, beta_i)
            overlap = np.linalg.det(c0a.conj().T.dot(psia_i)) * cs_ovlp

            if self.sys.ndown > 0:
                psib_i = c0b[permi, :]
                overlap *= np.linalg.det(c0b.conj().T.dot(psib_i))

            # Evaluate Greens functions
            Ga_j = gab(c0a, psia_i)
            if self.sys.ndown > 0:
                Gb_j = gab(c0b, psib_i)
            else:
                Gb_j = np.zeros_like(Ga_j)
            G_j = [Ga_j, Gb_j]

            # Obtain projected energy of permuted soliton on original soliton
            projected_energy = self.projected_energy_cubic(self.ham, G_j, shift, beta_i)
            num_energy += (projected_energy * overlap * fac_i).real
            denom += (overlap * fac_i).real
    
        # part I: elph shift with perm
        num_energy += np.einsum("jk,ijk,i->", (psis * (self.m_phases * cs_ovlps)).dot(betas.T), self.ham.g_tensor, np.squeeze(c0a).conj()).real

        # part II: elph shift with perm
        num_energy += np.einsum("jn,k,ijk,i->", (psis * (self.m_phases * cs_ovlps)), shift.conj(), self.ham.g_tensor, np.squeeze(c0a).conj()).real

        energy = num_energy / denom
        return energy.real



    def projected_energy_cubic(self, ham: GenericEPhModel, G: list, shift, beta_i):
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        phonon_contrib = ham.w0 * np.sum(shift.conj() * beta_i)
        local_energy = kinetic + phonon_contrib
        return local_energy


    def gradient(self, x, *args) -> np.ndarray:
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

        # TODO get total energy and overlap
        ovlp = 0.0
        energy = 0.0

        # Single-sum K-projection weight (same as in objective_function;
        # see comment at __init__).
        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            fac_i = self.m_phases[ip]

            beta_i = shift[permi]
            psia_i = c0a[permi]  # [permi, :]

            #perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            perm_mat = np.eye(self.ham.N)[permi, :]
            perm_mat_symm = perm_mat + perm_mat.conj().T
            perm_mat_asym = perm_mat - perm_mat.conj().T

            Ga_i = np.outer(c0a.conj(), psia_i)
            occ = np.sum(Ga_i.diagonal())

            kin = np.sum(Ga_i * self.ham.T[0])
            kin_perm = self.ham.T[0].dot(perm_mat)
            el_ph_c = np.einsum("ijk,ij->k", self.ham.g_tensor, Ga_i)
            el_ph_c_perm = np.einsum("k,km->m", el_ph_c, perm_mat)

#            cs_ovlp = np.exp(
#                np.dot(shift.conj(), beta_i) - np.sum(shift_abs**2)
#            )  # verify dot works here
            cs_ovlp = self.cs_overlap(shift, beta_i)
            ovlp_i = occ * cs_ovlp

            d_cs_ovlp_r = cs_ovlp * (
                perm_mat_symm.dot(shift.real) + 1j * perm_mat_asym.dot(shift.imag) - 2 * shift.real
            )
            d_cs_ovlp_i = cs_ovlp * (
                perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real) - 2 * shift.imag
            )

            # shift_grad_real contribs
            kin_contrib = kin * d_cs_ovlp_r
            el_ph_contrib = d_cs_ovlp_r * np.sum(el_ph_c * (shift.conj() + beta_i)) + cs_ovlp * (
                el_ph_c + el_ph_c_perm
            )
            boson_contrib = d_cs_ovlp_r * np.sum(shift.conj() * beta_i) + cs_ovlp * (
                perm_mat_symm.dot(shift.real) + 1j * perm_mat_asym.dot(shift.imag)
            )
            boson_contrib *= occ * self.ham.w0
            ovlp_contrib = d_cs_ovlp_r * occ
            sgr = kin_contrib + el_ph_contrib + boson_contrib
            sgr_ovlp = ovlp_contrib

            # shift_grad_imag contribs
            kin_contrib = kin * d_cs_ovlp_i
            el_ph_contrib = d_cs_ovlp_i * np.sum(
                el_ph_c * (shift.conj() + beta_i)
            ) + 1j * cs_ovlp * (el_ph_c_perm - el_ph_c)
            boson_contrib = d_cs_ovlp_i * np.sum(shift.conj() * beta_i) * self.ham.w0
            # + cs_ovlp * (perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real))
            boson_contrib += (
                cs_ovlp
                * (perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real))
                * self.ham.w0
            )
            boson_contrib *= occ
            ovlp_contrib = d_cs_ovlp_i * occ
            sgi = kin_contrib + el_ph_contrib + boson_contrib
            sgi_ovlp = ovlp_contrib

            # BUG HERE TODO, in both diffs wrt imag and real -> bug in cs_ovlp?

            # psia_grad_real contribs
            g_contracted = np.einsum("ijk,k->ij", self.ham.g_tensor, shift.conj() + beta_i).dot(
                perm_mat
            )  # TODO FIX

            kin_contrib = cs_ovlp * (
                (kin_perm + kin_perm.conj().T).dot(c0a.real)
                + 1j * (kin_perm - kin_perm.conj().T).dot(c0a.imag)
            )
            el_ph_contrib = cs_ovlp * (
                (g_contracted + g_contracted.T).dot(c0a.real)
                + 1j * (g_contracted - g_contracted.T).dot(c0a.imag)
            )
            boson_contrib = (
                self.ham.w0
                * cs_ovlp
                * shift.conj().dot(beta_i)
                * (perm_mat_symm.dot(c0a.real) + 1j * perm_mat_asym.dot(c0a.imag))
            )
            ovlp_contrib = cs_ovlp * (
                perm_mat_symm.dot(c0a.real) + 1j * perm_mat_asym.dot(c0a.imag)
            )
            pgr = kin_contrib + el_ph_contrib + boson_contrib
            pgr_ovlp = ovlp_contrib

            # psia_grad_imag contribs
            kin_contrib = cs_ovlp * (
                (kin_perm + kin_perm.conj().T).dot(c0a.imag)
                - 1j * (kin_perm - kin_perm.conj().T).dot(c0a.real)
            )
            el_ph_contrib = cs_ovlp * (
                (g_contracted + g_contracted.T).dot(c0a.imag)
                - 1j * (g_contracted - g_contracted.T).dot(c0a.real)
            )
            boson_contrib = (
                self.ham.w0
                * cs_ovlp
                * shift.conj().dot(beta_i)
                * (perm_mat_symm.dot(c0a.imag) - 1j * perm_mat_asym.dot(c0a.real))
            )
            ovlp_contrib = cs_ovlp * (
                perm_mat_symm.dot(c0a.imag) - 1j * perm_mat_asym.dot(c0a.real)
            )
            pgi = kin_contrib + el_ph_contrib + boson_contrib
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
            energy += (
                fac_i
                * cs_ovlp
                * (
                    kin
                    + np.sum(
                        np.einsum("ijk,k->ij", self.ham.g_tensor, shift.conj() + beta_i) * Ga_i
                    )
                    + self.ham.w0 * occ * shift.conj().dot(beta_i)
                )
            ).real
            ovlp += (fac_i * ovlp_i).real

#        print('energy in grad:  ', energy / ovlp)
#        print("energy from obj: ", self.objective_function(x))        

        dx_energy = np.hstack(
            [shift_grad_real, shift_grad_imag, psia_grad_real, psia_grad_imag]
        ).real.astype(np.float64)
        dx_ovlp = np.hstack(
            [shift_grad_real_ovlp, shift_grad_imag_ovlp, psia_grad_real_ovlp, psia_grad_imag_ovlp]
        ).real.astype(np.float64)
        #        print('dx energy and ovlp', np.hstack([shift_grad_real, shift_grad_imag, psia_grad_real, psia_grad_imag]), np.hstack([shift_grad_real_ovlp, shift_grad_imag_ovlp, psia_grad_real_ovlp, psia_grad_imag_ovlp]))
        
        #TODO remove, only for debugging!!
#        dx_energy = np.zeros_like(dx_energy)
#        energy = 1
        
        
        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp**2
        #        print('my grad: ', dx)
        #        exit()
        #        super().gradient(x)
        return dx

#    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift, beta_i):
#        print('shapes:  ', shift.shape, beta_i.shape)
        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
        el_ph_contrib = np.einsum(
            "ijk,ij,k->", ham.g_tensor, G[0], shift.conj() + beta_i
        )  # p == 1 for toyozawa
        phonon_contrib = ham.w0 * np.sum(shift.conj() * beta_i)
        local_energy = kinetic + el_ph_contrib + phonon_contrib  # + ferm_ferm_contrib
#        print("kinetic, elph, phonon:   ", kinetic, el_ph_contrib, phonon_contrib)
        return local_energy


    def gradient_cubic(self, x, *args) -> np.ndarray:
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

        # TODO get total energy and overlap
        ovlp = 0.0
        energy = 0.0



        # elph terms:
        ## shift real:
        for ip, permi in enumerate(self.perms):
            perm_mat = np.eye(self.ham.N)[permi, :]
            perm_mat_symm = perm_mat + perm_mat.conj().T
            perm_mat_asym = perm_mat - perm_mat.conj().T


        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            fac_i = self.m_phases[ip] #overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi
#            fac_i = 1.

            beta_i = shift[permi]
            psia_i = c0a[permi]  # [permi, :]

            #perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            perm_mat = np.eye(self.ham.N)[permi, :]
            perm_mat_symm = perm_mat + perm_mat.conj().T
            perm_mat_asym = perm_mat - perm_mat.conj().T

            Ga_i = np.outer(c0a.conj(), psia_i)
            occ = np.sum(Ga_i.diagonal())

            kin = np.sum(Ga_i * self.ham.T[0])
            kin_perm = self.ham.T[0].dot(perm_mat)
            el_ph_c = np.einsum("ijk,ij->k", self.ham.g_tensor, Ga_i)
            el_ph_c_perm = np.einsum("k,km->m", el_ph_c, perm_mat)

#            cs_ovlp = np.exp(
#                np.dot(shift.conj(), beta_i) - np.sum(shift_abs**2)
#            )  # verify dot works here
            cs_ovlp = self.cs_overlap(shift, beta_i)
            ovlp_i = occ * cs_ovlp

            d_cs_ovlp_r = cs_ovlp * (
                perm_mat_symm.dot(shift.real) + 1j * perm_mat_asym.dot(shift.imag) - 2 * shift.real
            )
            d_cs_ovlp_i = cs_ovlp * (
                perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real) - 2 * shift.imag
            )

            # shift_grad_real contribs
            kin_contrib = kin * d_cs_ovlp_r
            el_ph_contrib = d_cs_ovlp_r * np.sum(el_ph_c * (shift.conj() + beta_i)) + cs_ovlp * (
                el_ph_c + el_ph_c_perm
            )
            boson_contrib = d_cs_ovlp_r * np.sum(shift.conj() * beta_i) + cs_ovlp * (
                perm_mat_symm.dot(shift.real) + 1j * perm_mat_asym.dot(shift.imag)
            )
            boson_contrib *= occ * self.ham.w0
            ovlp_contrib = d_cs_ovlp_r * occ
            sgr = kin_contrib + el_ph_contrib + boson_contrib
            sgr_ovlp = ovlp_contrib

            # shift_grad_imag contribs
            kin_contrib = kin * d_cs_ovlp_i
            el_ph_contrib = d_cs_ovlp_i * np.sum(
                el_ph_c * (shift.conj() + beta_i)
            ) + 1j * cs_ovlp * (el_ph_c_perm - el_ph_c)
            boson_contrib = d_cs_ovlp_i * np.sum(shift.conj() * beta_i) * self.ham.w0
            # + cs_ovlp * (perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real))
            boson_contrib += (
                cs_ovlp
                * (perm_mat_symm.dot(shift.imag) - 1j * perm_mat_asym.dot(shift.real))
                * self.ham.w0
            )
            boson_contrib *= occ
            ovlp_contrib = d_cs_ovlp_i * occ
            sgi = kin_contrib + el_ph_contrib + boson_contrib
            sgi_ovlp = ovlp_contrib

            # BUG HERE TODO, in both diffs wrt imag and real -> bug in cs_ovlp?

            # psia_grad_real contribs
            g_contracted = np.einsum("ijk,k->ij", self.ham.g_tensor, shift.conj() + beta_i).dot(
                perm_mat
            )  # TODO FIX

            kin_contrib = cs_ovlp * (
                (kin_perm + kin_perm.conj().T).dot(c0a.real)
                + 1j * (kin_perm - kin_perm.conj().T).dot(c0a.imag)
            )
            el_ph_contrib = cs_ovlp * (
                (g_contracted + g_contracted.T).dot(c0a.real)
                + 1j * (g_contracted - g_contracted.T).dot(c0a.imag)
            )
            boson_contrib = (
                self.ham.w0
                * cs_ovlp
                * shift.conj().dot(beta_i)
                * (perm_mat_symm.dot(c0a.real) + 1j * perm_mat_asym.dot(c0a.imag))
            )
            ovlp_contrib = cs_ovlp * (
                perm_mat_symm.dot(c0a.real) + 1j * perm_mat_asym.dot(c0a.imag)
            )
            pgr = kin_contrib + el_ph_contrib + boson_contrib
            pgr_ovlp = ovlp_contrib

            # psia_grad_imag contribs
            kin_contrib = cs_ovlp * (
                (kin_perm + kin_perm.conj().T).dot(c0a.imag)
                - 1j * (kin_perm - kin_perm.conj().T).dot(c0a.real)
            )
            el_ph_contrib = cs_ovlp * (
                (g_contracted + g_contracted.T).dot(c0a.imag)
                - 1j * (g_contracted - g_contracted.T).dot(c0a.real)
            )
            boson_contrib = (
                self.ham.w0
                * cs_ovlp
                * shift.conj().dot(beta_i)
                * (perm_mat_symm.dot(c0a.imag) - 1j * perm_mat_asym.dot(c0a.real))
            )
            ovlp_contrib = cs_ovlp * (
                perm_mat_symm.dot(c0a.imag) - 1j * perm_mat_asym.dot(c0a.real)
            )
            pgi = kin_contrib + el_ph_contrib + boson_contrib
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
            energy += (
                fac_i
                * cs_ovlp
                * (
                    kin
                    + np.sum(
                        np.einsum("ijk,k->ij", self.ham.g_tensor, shift.conj() + beta_i) * Ga_i
                    )
                    + self.ham.w0 * occ * shift.conj().dot(beta_i)
                )
            ).real
            ovlp += (fac_i * ovlp_i).real

#        print('energy in grad:  ', energy / ovlp)
#        print("energy from obj: ", self.objective_function(x))        

        dx_energy = np.hstack(
            [shift_grad_real, shift_grad_imag, psia_grad_real, psia_grad_imag]
        ).real.astype(np.float64)
        dx_ovlp = np.hstack(
            [shift_grad_real_ovlp, shift_grad_imag_ovlp, psia_grad_real_ovlp, psia_grad_imag_ovlp]
        ).real.astype(np.float64)
        #        print('dx energy and ovlp', np.hstack([shift_grad_real, shift_grad_imag, psia_grad_real, psia_grad_imag]), np.hstack([shift_grad_real_ovlp, shift_grad_imag_ovlp, psia_grad_real_ovlp, psia_grad_imag_ovlp]))
        
        #TODO remove, only for debugging!!
#        dx_energy = np.zeros_like(dx_energy)
#        energy = 1
        
        
        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp**2
        #        print('my grad: ', dx)
        #        exit()
        #        super().gradient(x)
        return dx



def main():
    # System Parameters
    nup = 1
    ndown = 0
    nelec = (nup, ndown)

    # Hamiltonian Parameters
    g = 0.7
    t = 1.
    w0 = 0.5
    nsites = 10
    pbc = True

    system = Generic(nelec)
    ham = OpticalSSHModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
    ham.build()

    initial_electron = np.random.random((nsites, nup + ndown)) + 1j * np.random.random((nsites, nup + ndown))
    initial_phonons = np.random.normal(size=(nsites)) + 1j * np.random.normal(size=(nsites))

    dd1_obj = ToyozawaVariational(initial_phonons, initial_electron, ham, system, cplx=True, K=0.)
    x = dd1_obj.pack_x()
    dd1_obj.objective_function(x)

if __name__ == '__main__':
    main()

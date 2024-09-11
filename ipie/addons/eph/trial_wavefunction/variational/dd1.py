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

from typing import List, Union

import numpy as np
import plum

from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectron
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import (
    overlap_degeneracy,
    ToyozawaVariational,
)

from line_profiler import LineProfiler
from ipie.addons.eph.hamiltonians.dispersive_phonons import DispersivePhononModel
from numba import jit

@jit(nopython=True)
def get_elph_tensors(g_tensor: np.ndarray, Ga_i: np.ndarray, shift: np.ndarray, beta_i: np.ndarray, cs_ovlp: np.ndarray, perm_index: np.ndarray):
    elph_ij_cs = g_tensor * cs_ovlp[..., None]
    g_contracted = elph_ij_cs * shift.conj().T[:, None, :] 
    g_contracted += elph_ij_cs * beta_i.T[None, ...]
    g_contracted = np.sum(g_contracted, axis=2)
    econtr = g_contracted * Ga_i
    g_contracted = g_contracted[:, perm_index]
    elph_ij_cs *= Ga_i[...,None]
#    print(elph_ij_cs.strides, Ga_i[...,None].strides)
    return elph_ij_cs, econtr, g_contracted

@jit(nopython=True)
def cs_overlap_jit(shift_i, shift_j) -> np.ndarray:
    cs_ovlp = np.exp(shift_i.T.conj().dot(shift_j))
    cs_ovlp_norm_i = np.exp(-0.5 * np.sum(np.abs(shift_i) ** 2, axis=0))
    cs_ovlp_norm_j = np.exp(-0.5 * np.sum(np.abs(shift_j) ** 2, axis=0))
    cs_ovlp_norm = np.outer(cs_ovlp_norm_i, cs_ovlp_norm_j)
    cs_ovlp *= cs_ovlp_norm
    return cs_ovlp

@jit(nopython=True)
def elph_contrib_shift(shift: np.ndarray, beta_i: np.ndarray, elph_contracted: np.ndarray, perm_index: np.ndarray):
    el_ph_contrib = beta_i.dot(elph_contracted.T)
    el_ph_contrib += shift.conj().dot(elph_contracted)[perm_index, :][:, perm_index]
    el_ph_contrib -= shift.real * (np.sum(elph_contracted, axis=1) + np.sum(elph_contracted ,axis=0)[perm_index])
    return el_ph_contrib

@jit(nopython=True)
def kin_contrib_shift(shift: np.ndarray, beta_i: np.ndarray, kin_tensor: np.ndarray, permi: int, perm_index: np.ndarray):
    kin_contrib = beta_i.dot(kin_tensor.T)
    kin_contrib += shift.conj().dot(kin_tensor)[perm_index,:][:,perm_index]
    kin_contrib -= shift.real * (np.sum(kin_tensor, axis=1) + np.sum(kin_tensor, axis=0)[perm_index])
    return kin_contrib

@jit(nopython=True)
def boson_contrib_shift(shift: np.ndarray, beta_i: np.ndarray, Ga_shifts: np.ndarray, perm_index: np.ndarray, permi):
    boson_contrib = ovlp_contrib_shift(shift, beta_i, Ga_shifts * np.sum(shift.conj() * beta_i, axis=0), perm_index)

    boson_contrib += (beta_i * Ga_shifts)
    boson_contrib += (shift.conj() * Ga_shifts)[perm_index, :][:, perm_index]
    return boson_contrib

@jit(nopython=True)
def ovlp_contrib_shift(shift: np.ndarray, beta_i: np.ndarray, Ga_shifts: np.ndarray, perm_index: np.ndarray):
    ovlp_contrib = (beta_i - shift.real) * Ga_shifts
    ovlp_contrib += (shift.conj() * Ga_shifts)[perm_index, :][:, perm_index]
    ovlp_contrib -= shift.real * Ga_shifts[perm_index]
    return ovlp_contrib

def d_tensor_elec(c0a: np.ndarray, tensor: np.ndarray):
    contrib = (tensor + tensor.T).dot(c0a.real) 
    contrib += 1j * (tensor - tensor.T).dot(c0a.imag)
    return contrib


class dD1Variational(ToyozawaVariational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        K: Union[float, np.ndarray],
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, K, cplx)
        self.shift_params_rows = self.ham.N
    
    def objective_function(self, x, zero_th: float = 1e-12) -> float:
#    def objective_function(self, x) -> float: 
        """"""
#        zero_th = 1e-12
#        print(x, zero_th)
#        exit()
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]
        c0a = c0a[:,0]

        num_energy = 0.0
        denom = 0.0
        
        #denom = np.zeros_like(self.Kcoeffs)
        #num_energy = 

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = np.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi]

            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = np.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())
            overlap *= self.Kcoeffs[0].conj() * coeffi
#            print(overlap, zero_th)

            if np.abs(overlap) < zero_th:
                continue

            overlap *= overlap_degeneracy(self.ham, ip)

            Ga_i = np.outer(c0a.conj(), psia_i)
            if self.sys.ndown > 0:
                Gb_i = np.outer(c0b.conj(), psib_i)
            else:
                Gb_i = np.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]

            projected_energy = self.projected_energy(self.ham, G_i, shift, shift_i)
            num_energy += (
                projected_energy
                * overlap_degeneracy(self.ham, ip)
                * self.Kcoeffs[0].conj()
                * coeffi
            ).real
            denom += overlap.real
            num = (
                projected_energy
                * overlap_degeneracy(self.ham, ip)
                * self.Kcoeffs[0].conj()
                * coeffi
            ).real
#            print('engery & denom:  ', num, overlap.real)
#            print(f'energy & ovlp {ip}', projected_energy, overlap, overlap_degeneracy(self.ham, ip))
#        print('num_energy:  ', num_energy)
#        print('denom:   ', denom)
        energy = num_energy / denom
        # TODO
#        print(energy)
#        exit()
        return energy.real

    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]
        c0a = c0a[:,0]

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = np.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi]

            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = np.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())
            overlap *= self.Kcoeffs[0].conj() * coeffi

#            if np.abs(overlap) < zero_th:
#                continue

            overlap *= overlap_degeneracy(self.ham, ip)

            Ga_i = np.outer(c0a.conj(), psia_i)
            if self.sys.ndown > 0:
                Gb_i = np.outer(c0b.conj(), psib_i)
            else:
                Gb_i = np.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]

            projected_energy = self.projected_energy(self.ham, G_i, shift, shift_i)
            num_energy += (
                projected_energy
                * overlap_degeneracy(self.ham, ip)
                * self.Kcoeffs[0].conj()
                * coeffi
            ).real
            denom += overlap.real
            print(f'energy & ovlp {ip}', projected_energy, overlap, overlap_degeneracy(self.ham, ip))
        print('num_energy:  ', num_energy)
        print('denom:   ', denom)
        energy = num_energy / denom
        # TODO
        print(energy)
        exit()
        return energy.real
    
    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
#        print('shift:   ', shift)
#        print('c0a: ', c0a)
#        print('c0b: ', c0b)
 #       exit()

        shift = shift[0]
        c0a = c0a[:,0]

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = np.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi]
            
            for jp, (permj, coeffj) in enumerate(zip(self.perms, self.Kcoeffs)):
                shift_j = np.roll(shift, shift=(-jp, -jp), axis=(0, 1))
                psia_j = c0a[permj]

                cs_ovlp = self.cs_overlap(shift_j, shift_i)

                overlap = np.sum(psia_j.conj() * psia_i * cs_ovlp.diagonal())
                overlap *= coeffj.conj() * coeffi

#                if np.abs(overlap) < zero_th:
#                    continue

                Ga_i = np.outer(psia_j.conj(), psia_i)
                if self.sys.ndown > 0:
                    Gb_i = np.outer(c0b.conj(), psib_i)
                else:
                    Gb_i = np.zeros_like(Ga_i)
                G_i = [Ga_i, Gb_i]

                projected_energy = self.projected_energy(self.ham, G_i, shift_j, shift_i)
                num_energy += (
                    projected_energy
                    * coeffj.conj()
                    * coeffi
                )
                denom += overlap
                print(f'energy & ovlp {ip} {jp}', projected_energy, overlap)
        print('num_energy:  ', num_energy)
        print('denom:   ', denom)

        energy = num_energy / denom
        # TODO
        print('energy:    ', energy)
        exit()
        return energy.real

    def get_args(self):
        return ()

    def cs_overlap(self, shift_i, shift_j):
        return cs_overlap_jit(shift_i, shift_j)

    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift_i, shift_j):
 
        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        Gcsa = G[0] * cs_ovlp

        kinetic = np.sum(ham.T[0] * Gcsa)

        el_ph_contrib = np.sum(ham.g_tensor * Gcsa[...,None] * shift_i.conj().T[:,None,:])   
        el_ph_contrib += np.sum(ham.g_tensor * Gcsa[...,None] * shift_j.T[None,...])

        if self.sys.ndown > 0:
            Gcsb = G[1] * cs_ovlp
            kinetic += np.sum(ham.T[1] * Gcsb)
            el_ph_contrib += np.sum( ham.g_tensor * Gcsb[...,None] * shift_i.conj().T[:,None,:])
            el_ph_contrib += np.sum( ham.g_tensor * Gcsb[...,None] * shift_j.T[None,...])

        phonon_contrib = ham.w0 * np.sum((shift_i.conj() * shift_j).dot(Gcsa.diagonal()))

        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def projected_energy(self, ham: DispersivePhononModel, G: list, shift_i, shift_j):
        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        kinetic = np.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)

        el_ph_contrib = np.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += np.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = np.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
            el_ph_contrib += np.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)

        phonon_contrib = ham.w0 * np.einsum('ij,j->', shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal())
        phonon_contrib += np.einsum('ij,in,jn,n->', ham.ph_tensor, shift_i.conj(), shift_j, G[0].diagonal() * cs_ovlp.diagonal())
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def projected_energy(self, ham: ExcitonPhononCavityElectron, G: list, shift_i, shift_j):
        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        kinetic = np.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)

        # for 1e- this is just one body operator
        ferm_ferm_contrib = np.sum((ham.quad[0] * G[0] + ham.quad[1] * G[1]) * cs_ovlp)

        el_ph_contrib = np.einsum("ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += np.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = np.einsum(
                "ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp
            )
            el_ph_contrib += np.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)

        phonon_contrib = ham.w0 * np.einsum(
            "ij,j->", shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal()
        )

        local_energy = kinetic + el_ph_contrib + phonon_contrib + ferm_ferm_contrib
        return local_energy


    #@jit(nopython=True)
    def gradient(self, x, *args) -> np.ndarray:
        """For GenericEPhModel"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        c0a = np.squeeze(c0a)

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

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            fac_i = overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi
            
            perm_index = np.roll(np.arange(self.ham.N), shift=ip)
            
            beta_i = np.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            psia_i = c0a[permi]  # [permi, :]

            # TODO Could store these matrices

            Ga_i = np.outer(c0a.conj(), psia_i)
            cs_ovlp = self.cs_overlap(shift, beta_i)

            # Auxiliary Tensors
            kin_tensor = Ga_i * self.ham.T[0]
            kin = np.sum(kin_tensor * cs_ovlp)
            kin_csovlp = kin_tensor * cs_ovlp
            kin_perm = (cs_ovlp * self.ham.T[0])[:, perm_index]
            
#            lp = LineProfiler()
#            lp_wrapper = lp(get_elph_tensors)
#            for _ in range(2):
#                lp_wrapper(self.ham.g_tensor, Ga_i, shift, beta_i, cs_ovlp, perm_index)
#                lp.print_stats()
#            exit()

            elph_ij_cs, econtr, g_contracted = get_elph_tensors(self.ham.g_tensor, Ga_i, shift, beta_i, cs_ovlp, perm_index)
            w_contracted = self.ham.w0 * perm_mat.T * (np.sum(shift.conj() * beta_i, axis=0) * cs_ovlp.diagonal())
            ovlp_contracted = perm_mat.T * cs_ovlp.diagonal()

            ovlp_i = np.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())

            # shift_grad_real contribs
            # perm inverse
            kin_contrib = kin_contrib_shift(shift, beta_i, kin_csovlp, ip, perm_index)
            
            el_ph_contrib = elph_contrib_shift(shift, beta_i, econtr, perm_index)
            el_ph_contrib += np.sum(elph_ij_cs, axis=1).T
            el_ph_contrib += np.sum(elph_ij_cs, axis=0).T[perm_index, :][:, perm_index]

            Ga_shifts = Ga_i.diagonal() * cs_ovlp.diagonal()
            ovlp_contrib = ovlp_contrib_shift(shift, beta_i, Ga_shifts, perm_index)
            boson_contrib = boson_contrib_shift(shift, beta_i, Ga_shifts, perm_index, permi)
            boson_contrib *= self.ham.w0

            sgr = kin_contrib + el_ph_contrib + boson_contrib
            sgr_ovlp = ovlp_contrib

            # shift_grad_imag contribs
            kin_contrib = kin_contrib_shift(-1j * shift, -1j * beta_i, kin_csovlp, ip, perm_index)
            
            el_ph_contrib = elph_contrib_shift(-1j * shift, -1j * beta_i, econtr, perm_index)
            el_ph_contrib -= 1j * np.sum(elph_ij_cs, axis=1).T
            el_ph_contrib += 1j * np.sum(elph_ij_cs, axis=0).T[perm_index, :][:, perm_index]

            ovlp_contrib = ovlp_contrib_shift(-1j * shift,-1j * beta_i, Ga_shifts, perm_index)
            boson_contrib = boson_contrib_shift(-1j * shift, -1j * beta_i, Ga_shifts, perm_index, permi)
            boson_contrib *= self.ham.w0

            sgi = kin_contrib + el_ph_contrib + boson_contrib
            sgi_ovlp = ovlp_contrib

            # psia_grad_real contribs
            kin_contrib = d_tensor_elec(c0a, kin_perm)
            el_ph_contrib = d_tensor_elec(c0a, g_contracted)
            boson_contrib = d_tensor_elec(c0a.conj(), w_contracted)
            ovlp_contrib = d_tensor_elec(c0a.conj(), ovlp_contracted)

            pgr = kin_contrib + el_ph_contrib + boson_contrib
            pgr_ovlp = ovlp_contrib

            # psia_grad_imag contribs
            kin_contrib = d_tensor_elec(-1j * c0a, kin_perm)
            el_ph_contrib = d_tensor_elec(-1j * c0a, g_contracted)
            boson_contrib = d_tensor_elec(1j * c0a.conj(), w_contracted)
            ovlp_contrib = d_tensor_elec(1j * c0a.conj(), ovlp_contracted)

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

            tmp = self.ham.g_tensor * (Ga_i * cs_ovlp)[..., None]
            eph_energy = np.sum(tmp * shift.conj().T[:, None, :])
            eph_energy += np.sum(tmp * beta_i.T[None, ...])

            ph_energy = self.ham.w0 * np.sum((shift.conj() * beta_i).dot(Ga_shifts))
            energy += (fac_i * (kin + eph_energy + ph_energy)).real
            ovlp += (fac_i * ovlp_i).real

        shift_grad = (shift_grad_real + 1j * shift_grad_imag).ravel()
        psia_grad = (psia_grad_real + 1j * psia_grad_imag).ravel()
        shift_grad_ovlp = (shift_grad_real_ovlp + 1j * shift_grad_imag_ovlp).ravel()
        psia_grad_ovlp = (psia_grad_real_ovlp + 1j * psia_grad_imag_ovlp).ravel()

        dx_energy = self.pack_x(shift_grad, psia_grad)
        dx_ovlp = self.pack_x(shift_grad_ovlp, psia_grad_ovlp)

        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp**2
#        print('my grad: ', dx)
#        print('diff:    ', np.max(np.abs(dx - np.load('/n/home01/mbaumgarten/Software/ipie/examples/14-1d_holstein/variational_test/dd1_testing/gnumeric.npy'))))
#        exit()
        return dx

    def _gradient(self, x, *args) -> np.ndarray:
        shift, c0a, c0b = self.unpack_x(x)
        shift_grad, psia_grad, energy, ovlp = gradient_jit(shift, c0a, c0b, ham.ham_jit, self.perms, self.Kcoeffs)

        dx_energy = self.pack_x(shift_grad, psia_grad)
        dx_ovlp = self.pack_x(shift_grad_ovlp, psia_grad_ovlp)

        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp**2
        return dx

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

#@jit(nopython=True)
#def cs_overlap_jit(shift_i, shift_j) -> np.ndarray:
#    cs_ovlp = np.exp(shift_i.T.conj().dot(shift_j))
#    cs_ovlp_norm_i = np.exp(-0.5 * np.sum(np.abs(shift_i) ** 2, axis=0))
#    cs_ovlp_norm_j = np.exp(-0.5 * np.sum(np.abs(shift_j) ** 2, axis=0))
#    cs_ovlp_norm = np.outer(cs_ovlp_norm_i, cs_ovlp_norm_j)
#    cs_ovlp *= cs_ovlp_norm
#    return cs_ovlp

@jit(nopython=True)
def cs_overlap_jit(shift_i, shift_j) -> np.ndarray:
    cs_ovlp = shift_i.T.conj().dot(shift_j)
    cs_ovlp_norm_i = -0.5 * npj.einsum("ij->j", npj.abs(shift_i) ** 2)
    cs_ovlp_norm_j = -0.5 * npj.einsum("ij->j", npj.abs(shift_j) ** 2)
    cs_ovlp_norm = np.add.outer(cs_ovlp_norm_i, cs_ovlp_norm_j)
    cs_ovlp += cs_ovlp_norm
    cs_ovlp = np.exp(cs_ovlp)
    return cs_ovlp


@jit(nopython=True)
def elph_contrib_shift(shift: np.ndarray, beta_i: np.ndarray, elph_contracted: np.ndarray, perm_index: np.ndarray):
    el_ph_contrib = beta_i.dot(elph_contracted.T)
    el_ph_contrib += shift.conj().dot(elph_contracted)[perm_index, :][:, perm_index]
    el_ph_contrib -= shift.real * (np.sum(elph_contracted, axis=1) + np.sum(elph_contracted ,axis=0)[perm_index])
    return el_ph_contrib

@jit(nopython=True)
def kin_contrib_shift(shift: np.ndarray, beta_i: np.ndarray, kin_tensor: np.ndarray, perm_index: np.ndarray):
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

#@jit(nopython=True)
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
        energy = num_energy / denom
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
#        exit()
        return energy.real

    def get_args(self):
        return ()

    def cs_overlap(self, shift_i, shift_j):
        return cs_overlap_jit(shift_i, shift_j)

    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift_i, shift_j):
 
        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        Gcsa = G[0] * cs_ovlp
#        print('cs_ovlp in csa:  ', cs_ovlp)
#        print('Ga_i:    ', G[0])
#        print('Gcsa:    ', Gcsa)
        kinetic = np.sum(ham.T[0] * Gcsa)

        el_ph_contrib = np.sum(ham.g_tensor * Gcsa[...,None] * shift_i.conj().T[:,None,:])   
        el_ph_contrib += np.sum(ham.g_tensor * Gcsa[...,None] * shift_j.T[None,...])

        if self.sys.ndown > 0:
            Gcsb = G[1] * cs_ovlp
            kinetic += np.sum(ham.T[1] * Gcsb)
            el_ph_contrib += np.sum( ham.g_tensor * Gcsb[...,None] * shift_i.conj().T[:,None,:])
            el_ph_contrib += np.sum( ham.g_tensor * Gcsb[...,None] * shift_j.T[None,...])

        phonon_contrib = ham.w0 * np.sum((shift_i.conj() * shift_j).dot(Gcsa.diagonal()))

#        print('kinectic, el_ph, ph: ', kinetic, el_ph_contrib, phonon_contrib)
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
 #   def projected_energy(self, ham: ExcitonPhononCavityElectron, G: list, shift_i, shift_j):
 #       cs_ovlp = self.cs_overlap(shift_i, shift_j)
 #       kinetic = np.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)

        # for 1e- this is just one body operator
#        ferm_ferm_contrib = np.sum((ham.quad[0] * G[0] + ham.quad[1] * G[1]) * cs_ovlp)

  #      el_ph_contrib = np.einsum("ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
  #      el_ph_contrib += np.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)
  #      if self.sys.ndown > 0:
  #          el_ph_contrib = np.einsum(
  #              "ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp
  #          )
  #          el_ph_contrib += np.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)

  #      phonon_contrib = ham.w0 * np.einsum(
  #          "ij,j->", shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal()
  #      )

#        local_energy = kinetic + el_ph_contrib + phonon_contrib + ferm_ferm_contrib
   #     local_energy = kinetic + el_ph_contrib + phonon_contrib
   #     return local_energy


    #@jit(nopython=True)
    def gradient_comps(self, x, *args) -> np.ndarray:
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
            
            Ga_i = np.outer(c0a.conj(), psia_i) # NOTE
            cs_ovlp = self.cs_overlap(shift, beta_i)

            # Auxiliary Tensors
            kin_tensor = Ga_i * self.ham.T[0]
            kin = np.sum(kin_tensor * cs_ovlp)
            kin_csovlp = kin_tensor * cs_ovlp
            kin_perm = (cs_ovlp * self.ham.T[0])[:, perm_index]

            elph_ij_cs, econtr, g_contracted = get_elph_tensors(self.ham.g_tensor, Ga_i, shift, beta_i, cs_ovlp, perm_index)
            w_contracted = self.ham.w0 * perm_mat.T * (np.sum(shift.conj() * beta_i, axis=0) * cs_ovlp.diagonal())
            ovlp_contracted = perm_mat.T * cs_ovlp.diagonal()

            ovlp_i = np.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())

            # shift_grad_real contribs
            # perm inverse
            Ga_shifts = Ga_i.diagonal() * cs_ovlp.diagonal()
            kin_contrib = kin_contrib_shift(shift, beta_i, kin_csovlp, perm_index)

            el_ph_contrib = elph_contrib_shift(shift, beta_i, econtr, perm_index)
            el_ph_contrib += np.sum(elph_ij_cs, axis=1).T
            el_ph_contrib += np.sum(elph_ij_cs, axis=0).T[perm_index, :][:, perm_index]

            ovlp_contrib = ovlp_contrib_shift(shift, beta_i, Ga_shifts, perm_index)
            boson_contrib = boson_contrib_shift(shift, beta_i, Ga_shifts, perm_index, permi)
            boson_contrib *= self.ham.w0

            sgr = kin_contrib + el_ph_contrib + boson_contrib
            sgr_ovlp = ovlp_contrib

            # shift_grad_imag contribs
            kin_contrib = kin_contrib_shift(-1j * shift, -1j * beta_i, kin_csovlp, perm_index)
            
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
        return dx_energy, dx_ovlp, energy, ovlp 
#        return dx


    def hess_diag(self, x, *args) -> np.ndarray:
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        c0a = np.squeeze(c0a)
        shift_abs = np.abs(shift)

        shift_hess_real = np.zeros_like(shift)
        shift_hess_imag = np.zeros_like(shift)
        psia_hess_real = np.zeros_like(c0a)
        psia_hess_imag = np.zeros_like(c0a)

        shift_hess_real_ovlp = np.zeros_like(shift)
        shift_hess_imag_ovlp = np.zeros_like(shift)
        psia_hess_real_ovlp = np.zeros_like(c0a)
        psia_hess_imag_ovlp = np.zeros_like(c0a)

        # TODO get total energy and overlap
        ovlp = 0.0
        energy = 0.0

        kin_en_hess = np.zeros_like(x, dtype=np.float64)
        elph_en_hess = np.zeros_like(x, dtype=np.float64)
        boson_en_hess = np.zeros_like(x, dtype=np.float64)
        ovlp_en_hess = np.zeros_like(x, dtype=np.float64)

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            perm_mat_symm = perm_mat + perm_mat.T
            perm_mat_asym = perm_mat - perm_mat.T

            fac_i = overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi

            # beta_i = shift[permi]
            beta_i = np.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            psia_i = c0a[permi]  # [permi, :]

            Ga_i = np.outer(c0a.conj(), psia_i)
            occ = np.sum(Ga_i.diagonal())
            cs_ovlp = self.cs_overlap(shift, beta_i)

            # Auxiliary Tensors
            kin_tensor = Ga_i * self.ham.T[0]
            kin = np.sum(kin_tensor * cs_ovlp)

            ovlp_i = np.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())

            elph_ij_contracted = np.einsum("ijk,ij->ijk", self.ham.g_tensor, Ga_i)
            
            # shift hess real
            kin_contrib = self.contract_ij_peij_r(kin_tensor, shift, beta_i, perm_mat, cs_ovlp, 1.)

            elph_contrib = 2 * np.einsum('ejp,pe,ej->pe', elph_ij_contracted, perm_mat.T.dot(shift.conj()), perm_mat.T * cs_ovlp)
            elph_contrib += 2 * np.einsum('ijk,kp,ej,pi,je,ij->pe', elph_ij_contracted, perm_mat, perm_mat.T, perm_mat.T.dot(shift.conj()), perm_mat, cs_ovlp)
            elph_contrib += self.elph_d2_r(elph_ij_contracted, shift, beta_i, perm_mat, cs_ovlp, 1.)
            elph_contrib += self.contract_ij_peij_r(np.sum(elph_ij_contracted * shift.conj().T[:, None, :], axis=2), shift, beta_i, perm_mat, cs_ovlp, 1.)                         #10
            elph_contrib += self.contract_ij_peij_r(np.sum(elph_ij_contracted * beta_i.T[None, :, :], axis=2), shift, beta_i, perm_mat, cs_ovlp, 1.)                               #11

            wij = np.eye(self.ham.N)
            cs_ga = Ga_i * cs_ovlp
            boson_contrib = self.contract_ij_peij_r(np.diag(Ga_i.diagonal() * np.sum(shift.conj() * beta_i, axis=0) ), shift, beta_i, perm_mat, cs_ovlp, 1.) * self.ham.w0
            boson_contrib += self.boson_d2_r(cs_ga, wij, shift, beta_i, perm_mat, 1.)
            ovlp_contrib = self.contract_ij_peij_r(np.diag(Ga_i.diagonal()), shift, beta_i, perm_mat, cs_ovlp, 1.)

            shift_hess_real += (fac_i * (kin_contrib + elph_contrib + boson_contrib)).real
            shift_hess_real_ovlp += (fac_i * ovlp_contrib).real

            # shift hess imag
            kin_contrib = self.contract_ij_peij_r(kin_tensor, shift, beta_i, perm_mat, cs_ovlp, -1j)

            elph_contrib = -2 *  np.einsum('ijk,kp,ej,pi,je,ij->pe', elph_ij_contracted, perm_mat, perm_mat.T, perm_mat.T.dot(shift.conj()), perm_mat, cs_ovlp)                             #6
            
            elph_contrib += 2 * np.einsum('ejp,pe,ej->pe', elph_ij_contracted, perm_mat.T.dot(shift.conj()), perm_mat.T * cs_ovlp)                                                 #2
            elph_contrib += self.elph_d2_i(elph_ij_contracted, shift, beta_i, perm_mat, cs_ovlp, -1j)
            elph_contrib += self.contract_ij_peij_r(np.sum(elph_ij_contracted * shift.conj().T[:, None, :], axis=2), shift, beta_i, perm_mat, cs_ovlp, -1j)
            elph_contrib += self.contract_ij_peij_r(np.sum(elph_ij_contracted * beta_i.T[None, :, :], axis=2), shift, beta_i, perm_mat, cs_ovlp, -1j)

            boson_contrib = self.contract_ij_peij_r(np.diag(Ga_i.diagonal() * shift.conj().T.dot(wij).dot(beta_i).diagonal() ), shift, beta_i, perm_mat, cs_ovlp, -1j) * self.ham.w0
            boson_contrib += self.boson_d2_i(cs_ga, wij, shift, beta_i, perm_mat, 1.)

            ovlp_contrib = self.contract_ij_peij_r(np.diag(Ga_i.diagonal()), shift, beta_i, perm_mat, cs_ovlp, -1j)
            
            shift_hess_imag += (fac_i * (kin_contrib + elph_contrib + boson_contrib)).real
            shift_hess_imag_ovlp += (fac_i * ovlp_contrib).real

            # helper tensors for elec hess
            g_contracted = np.einsum("ijk,ki->ij", self.ham.g_tensor, shift.conj())
            g_contracted += np.einsum("ijk,kj->ij", self.ham.g_tensor, beta_i)
            g_contracted = g_contracted * cs_ovlp
            g_contracted = g_contracted.dot(perm_mat)
            w_contracted = self.ham.w0 * np.einsum(
                "ij,j,jn->jn", shift.conj() * beta_i, cs_ovlp.diagonal(), perm_mat
            )
            ovlp_contracted = np.einsum("i,ij->ij", cs_ovlp.diagonal(), perm_mat)

            kin_perm = (cs_ovlp * self.ham.T[0]).dot(perm_mat)

            # elec hess real
            kin_contrib = 2 * kin_perm.diagonal()
            elph_contrib = 2 * g_contracted.diagonal()
            boson_contrib = 2 * w_contracted.diagonal()
            ovlp_contrib = 2 * ovlp_contracted.diagonal()
            psia_hess_real += (fac_i * (kin_contrib + elph_contrib + boson_contrib)).real   
            psia_hess_real_ovlp += (fac_i * ovlp_contrib).real

            # elec hess imag
            kin_contrib = 2 * kin_perm.diagonal()
            elph_contrib = 2 * g_contracted.diagonal()
            boson_contrib = 2 * w_contracted.diagonal()
            ovlp_contrib = 2 * ovlp_contracted.diagonal()
            psia_hess_imag += (fac_i * (kin_contrib + elph_contrib + boson_contrib)).real
            psia_hess_imag_ovlp += (fac_i * ovlp_contrib).real

            en = self.projected_energy(self.ham, [Ga_i, np.zeros_like(Ga_i)], shift, beta_i)
            energy += (fac_i * en).real
            ovlp += (fac_i * ovlp_i).real

        shift_hess = (shift_hess_real + 1j * shift_hess_imag).ravel()
        psia_hess = (psia_hess_real + 1j * psia_hess_imag).ravel()
        shift_hess_ovlp = (shift_hess_real_ovlp + 1j * shift_hess_imag_ovlp).ravel()
        psia_hess_ovlp = (psia_hess_real_ovlp + 1j * psia_hess_imag_ovlp).ravel()

        d2x_energy = self.pack_x(shift_hess, psia_hess)
        d2x_ovlp = self.pack_x(shift_hess_ovlp, psia_hess_ovlp)

        # Gather
        dx_energy, dx_ovlp, energy_grad, ovlp_grad = self.gradient_comps(x, *args)
        assert np.allclose(energy_grad, energy)
        assert np.allclose(ovlp_grad, ovlp)
        H = (d2x_energy / ovlp) - energy * d2x_ovlp / ovlp**2 - 2 * dx_ovlp * (dx_energy * ovlp - energy * dx_ovlp) / (ovlp**3)

        return H


    def gradient(self, x, *args) -> np.ndarray:
        dx_energy, dx_ovlp, energy, ovlp = self.gradient_comps(x, *args)
        return dx_energy / ovlp - dx_ovlp * energy / ovlp**2

    def contract_ij_peij_r(self, tensor: np.ndarray, shift: np.ndarray, beta_i: np.ndarray, perm_mat: np.ndarray, cs_ovlp: np.ndarray, fac) -> np.ndarray:
        """Contracts indices ij without building 4 index object"""
        # Counter:
        #   4 - perm_mat.T.dot(beta_i.real) 
        #   4 - perm_mat.T.dot(shift.conj())
        #   4 - 
        shift_new = shift.copy() * fac
        beta_i_new = beta_i.copy() * fac

        cs_ovlp_new = cs_ovlp.copy() *  tensor

        d2_r = (beta_i_new ** 2).dot(cs_ovlp_new.T)                                                                         # 1
       

        d2_r -= 2 * (beta_i_new * perm_mat.T.dot(beta_i_new.real)).dot(perm_mat * cs_ovlp_new.T)                                # 2
        d2_r -= 2 * perm_mat.T.dot(shift_new.conj()) * shift_new.real * np.sum(perm_mat.T * cs_ovlp_new, axis=1)                # 3
        d2_r -= 2 * (perm_mat.T.dot(shift_new.conj()).dot(cs_ovlp_new) * perm_mat.T.dot(beta_i_new.real)).dot(perm_mat **2)     # 4
        d2_r += 2 * shift_new.real * perm_mat.T.dot(beta_i_new.real).dot(perm_mat * cs_ovlp_new.T)                              # 5
        d2_r += 2 * np.einsum('p,e->pe', perm_mat.diagonal(), np.sum(perm_mat.T * cs_ovlp_new, axis=1))                 # 6
        d2_r += 2 * beta_i_new.dot(perm_mat * cs_ovlp_new.T) * perm_mat.T.dot(shift_new.conj())                                 # 7
        d2_r -= 2 * beta_i_new.dot(cs_ovlp_new.T) * shift_new.real                                                              # 8
        
        d2_r -= np.einsum('e,p->pe', np.sum(cs_ovlp_new, axis=1), np.ones(self.ham.N))                                  # 9
        d2_r -= np.einsum('p,e->pe', np.sum(perm_mat, axis=0), np.sum(perm_mat.T.dot(cs_ovlp_new.T), axis=1))           # 10
        d2_r += (perm_mat.T.dot(shift_new.conj()) ** 2).dot(cs_ovlp_new).dot( perm_mat ** 2)                                # 11
        d2_r += (shift_new.real**2) * np.sum(cs_ovlp_new, axis=1)                                                           # 12
        d2_r += ((perm_mat.T.dot(beta_i_new.real) ** 2) * np.sum(cs_ovlp_new,axis=0)).dot(perm_mat ** 2)                    # 13

        return d2_r

    def elph_d2_r(self, elph_ij_contracted: np.ndarray, shift: np.ndarray, beta_i: np.ndarray, perm_mat: np.ndarray, cs_ovlp: np.ndarray,  fac):
        
       
        shift_new = shift.copy() * fac
        beta_i_new = beta_i.copy() * fac

        elph_contrib = np.einsum('ejp,pj,ej->pe', elph_ij_contracted, beta_i_new, cs_ovlp) # * fac
        elph_contrib -= np.einsum('ejp,pe,ej->pe', elph_ij_contracted, shift_new.real, cs_ovlp) # * fac
        elph_contrib -= np.einsum('ejp,pj,ej->pe', elph_ij_contracted, perm_mat.T.dot(beta_i_new.real), perm_mat.T * cs_ovlp) # * fac

        elph_contrib += np.einsum('ejk,kp,ej,pj,ej->pe', elph_ij_contracted, perm_mat, perm_mat.T, beta_i_new, cs_ovlp) # * fac
        elph_contrib -= np.einsum('ejk,kp,ej,pe,ej->pe', elph_ij_contracted, perm_mat, perm_mat.T, shift_new.real, cs_ovlp)
        elph_contrib -= np.einsum('ijk,kp,pj,ej,ij->pe', elph_ij_contracted, perm_mat, perm_mat.T.dot(beta_i_new.real), perm_mat.T ** 2, cs_ovlp)

        elph_contrib *= 2.

        return elph_contrib

    def elph_d2_i(self,elph_ij_contracted: np.ndarray, shift: np.ndarray, beta_i: np.ndarray, perm_mat: np.ndarray, cs_ovlp: np.ndarray, fac):
        shift_new = shift.copy() * fac
        beta_i_new = beta_i.copy() * fac

        elph_contrib = np.einsum('ejp,pj,ej->pe', elph_ij_contracted, beta_i_new, cs_ovlp) * fac
        elph_contrib -= np.einsum('ejp,pe,ej->pe', elph_ij_contracted, shift_new.real, cs_ovlp) * fac
        elph_contrib -= np.einsum('ejp,pj,ej,ej->pe', elph_ij_contracted, shift_new.real.dot(perm_mat.T), perm_mat.T, cs_ovlp, optimize=True) * fac

        elph_contrib += np.einsum('ejk,kp,ej,pj,ej->pe', elph_ij_contracted, perm_mat, perm_mat.T, beta_i_new, cs_ovlp) * (-fac)            #5
        elph_contrib -= np.einsum('ejk,kp,ej,pe,ej->pe', elph_ij_contracted, perm_mat, perm_mat.T, shift_new.real, cs_ovlp) *(-fac) 
        elph_contrib -= np.einsum('ijk,kp,ej,pj,ij->pe', elph_ij_contracted, perm_mat, perm_mat.T ** 2, shift_new.real.dot(perm_mat.T), cs_ovlp, optimize=True) * (-fac)

        elph_contrib *= 2.
        return elph_contrib

    def boson_d2_r(self, tensor: np.ndarray, wij: np.ndarray, shift: np.ndarray, beta_i: np.ndarray, perm_mat: np.ndarray, fac):
        beta_i_new = beta_i.copy() * fac
        shift_new = shift.copy() * fac
        
        boson_contrib = 2 * (wij.dot(beta_i) * tensor.diagonal()) * (beta_i - shift.real)
        boson_contrib += 2 * (wij.dot(beta_i) * (tensor * perm_mat.T).diagonal()) * perm_mat.T.dot(shift.conj() - beta_i.real)
        boson_contrib += 2 * shift.conj().T.dot(wij).dot(perm_mat).T * (tensor * perm_mat.T).diagonal() * (beta_i - shift.real)       
        boson_contrib += 2 * ((shift.conj().T.dot(wij).dot(perm_mat) * (shift.conj().T - beta_i.real.T).dot(perm_mat)).T * tensor.diagonal()).dot(perm_mat ** 2)
        boson_contrib += 2 * np.einsum("p,e->pe", perm_mat.diagonal(), (tensor * perm_mat.T).diagonal(), optimize=True)
        boson_contrib *= self.ham.w0
        return boson_contrib

    def boson_d2_i(self, tensor: np.ndarray, wij: np.ndarray, shift: np.ndarray, beta_i: np.ndarray, perm_mat: np.ndarray, fac):
        boson_contrib = -2 * wij.dot(beta_i) * beta_i * tensor.diagonal()
        boson_contrib += 2 * (wij.dot(beta_i) * perm_mat.T.dot(shift.conj())) * (tensor.diagonal() * perm_mat.diagonal())
        boson_contrib += 1j * 2 * (wij.dot(beta_i) * shift.imag) * tensor.diagonal()
        boson_contrib += 1j * 2 * (wij.dot(beta_i) * shift.imag.dot(perm_mat.T)) * (tensor.diagonal() * perm_mat.T.diagonal())

        boson_contrib += 2 * (beta_i.T * shift.conj().T.dot(wij).dot(perm_mat)).T * (tensor * perm_mat.T).diagonal()
        boson_contrib -= 2 * ((shift.conj().T.dot(wij).dot(perm_mat)).T * perm_mat.T.dot(shift.conj()) * tensor.diagonal()).dot(perm_mat ** 2)
        boson_contrib -= 1j * 2 * shift.conj().T.dot(wij).dot(perm_mat).T * shift.imag * (tensor * perm_mat).diagonal()
        boson_contrib -= 1j * 2 * ((shift.conj().T.dot(wij).dot(perm_mat) * shift.imag.dot(perm_mat.T).T).T * tensor.diagonal()).dot(perm_mat**2)
        boson_contrib += 2 * np.einsum("p,e->pe", wij.dot(perm_mat).diagonal(), (tensor * perm_mat.T).diagonal(), optimize=True) # perm_mat.T should go over ej for general ph tensor # this is d2 of beta_0* beta_i
        boson_contrib *= self.ham.w0
        return boson_contrib


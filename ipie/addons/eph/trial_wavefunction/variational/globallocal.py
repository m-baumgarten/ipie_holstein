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
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
import plum
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import overlap_degeneracy
from ipie.addons.eph.trial_wavefunction.variational.dd1 import (
    dD1Variational,
    get_elph_tensors,
    elph_contrib_shift,
    cs_overlap_jit,
    kin_contrib_shift,
    boson_contrib_shift,
    ovlp_contrib_shift,
    d_tensor_elec,
)

class GlobalLocalVariational(dD1Variational):
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
        self.shift_params_rows = 2
    
    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]

        shift_tmp = np.repeat(shift[:,0][:,np.newaxis], self.ham.N, axis=1)
        shift_tmp -= self.shift_roll(shift[:,1])
        shift = shift_tmp

        num_energy = 0.
        denom = 0.


        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = np.roll(shift, shift=(-ip,-ip), axis=(0,1))
            psia_i = c0a[permi, :]
                
            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = np.einsum('ie,ie,i->', c0a.conj(), psia_i, cs_ovlp.diagonal()) 
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
            num_energy += (projected_energy * overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi).real
            denom += overlap.real
        
        energy = num_energy / denom
        return energy.real
    
    def shift_roll(self, beta) -> np.ndarray:
        circs = beta
        for shift in range(1, len(beta)):
            new_circ = np.roll(beta, shift)
            circs = np.vstack([circs, new_circ])
        return circs.T

    def gradient(self, x, *args) -> np.ndarray:
        """For GenericEPhModel"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        
        shift_tmp = np.repeat(shift[:,0][:,np.newaxis], self.ham.N, axis=1)
        shift_tmp -= self.shift_roll(shift[:,1])
        shift = shift_tmp
        
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

        print(shift_grad)
        print(shift_grad_ovlp)
        exit()
        dx_energy = self.pack_x(shift_grad, psia_grad)
        dx_ovlp = self.pack_x(shift_grad_ovlp, psia_grad_ovlp)

        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp**2
#        print('my grad: ', dx)
#        print('diff:    ', np.max(np.abs(dx - np.load('/n/home01/mbaumgarten/Software/ipie/examples/14-1d_holstein/variational_test/dd1_testing/gnumeric.npy'))))
#        exit()
        return dx


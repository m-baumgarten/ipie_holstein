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
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectron
import jax
import jax.numpy as npj
import plum
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational, overlap_degeneracy

from ipie.addons.eph.hamiltonians.dispersive_phonons import DispersivePhononModel

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

        num_energy = 0.
        denom = 0.


        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip,-ip), axis=(0,1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi, :]
                
            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = npj.einsum('ie,ie,i->', c0a.conj(), psia_i, cs_ovlp.diagonal()) 
            overlap *= self.Kcoeffs[0].conj() * coeffi

            if npj.abs(overlap) < zero_th:
                continue

            overlap *= overlap_degeneracy(self.ham, ip)

            Ga_i = npj.outer(c0a.conj(), psia_i)
            if self.sys.ndown > 0:
                Gb_i = npj.outer(c0b.conj(), psib_i)
            else:
                Gb_i = npj.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]
   
            projected_energy = self.projected_energy(self.ham, G_i, shift, shift_i)
            num_energy += (projected_energy * overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi).real
            denom += overlap.real
        
        energy = num_energy / denom
# TODO        
        return energy.real

    def get_args(self):
        return ()

    def cs_overlap(self, shift_i, shift_j):
        cs_ovlp = npj.exp(shift_i.T.conj().dot(shift_j))
        cs_ovlp_norm_i = npj.exp(-0.5 * npj.einsum('ij->j', npj.abs(shift_i) ** 2))
        cs_ovlp_norm_j = npj.exp(-0.5 * npj.einsum('ij->j', npj.abs(shift_j) ** 2))
        cs_ovlp_norm = npj.outer(cs_ovlp_norm_i, cs_ovlp_norm_j)
        cs_ovlp *= cs_ovlp_norm
        return cs_ovlp

    def shift_perm_tmp(self, x, nperm, i, j):
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        beta_i = npj.roll(shift, shift=(-nperm,-nperm), axis=(0,1))
        perm_mat = npj.roll(np.eye(self.ham.N), shift=-nperm, axis=0)

        ovlp = npj.einsum('ki,km,mn,nj->ij', shift.conj(), perm_mat, shift, perm_mat.T)
        return ovlp[i,j]

    def cs_ovlp_tmp(self, x, nperm, i, j):
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
        beta_i = npj.roll(shift, shift=(-nperm,-nperm), axis=(0,1))
        return self.cs_overlap(shift, beta_i)[i,j]

    def elph_tmp(self, x, nperm, i, j):
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
        beta_i = npj.roll(shift, shift=(-nperm,-nperm), axis=(0,1))
        psia_i = c0a[self.perms[nperm]]
        cs_ovlp = self.cs_overlap(shift, beta_i)

        Ga_i = npj.outer(c0a.conj(), psia_i)

        el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', self.ham.g_tensor, Ga_i, shift.conj(), cs_ovlp)
        el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', self.ham.g_tensor, Ga_i, beta_i, cs_ovlp)
        return el_ph_contrib.real

    def double_perm_tmp(self, x, nperm, i, j):
        shift, c0a, c0b = self.unpack_x(x)
        shift = npj.squeeze(shift)
        beta_i = npj.roll(shift, shift=(-nperm,-nperm), axis=(0,1))
        return npj.einsum('ki,kj->ij',shift.conj(), beta_i)[i,j] 

    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift_i, shift_j):

        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        
        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)
       
        el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
            el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        
        phonon_contrib = ham.w0 * npj.einsum('ij,j->', shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal())

        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy
    
    @plum.dispatch
    def projected_energy(self, ham: DispersivePhononModel, G: list, shift_i, shift_j):
        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)
        
        el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
            el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        
        phonon_contrib = ham.w0 * npj.einsum('ij,j->', shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal())
        phonon_contrib += npj.einsum('ij,in,jn,n->', ham.ph_tensor, shift_i.conj(), shift_j, G[0].diagonal() * cs_ovlp.diagonal())
        local_energy = kinetic + el_ph_contrib + phonon_contrib
        return local_energy

    @plum.dispatch
    def projected_energy(self, ham: ExcitonPhononCavityElectron, G: list, shift_i, shift_j):
        cs_ovlp = self.cs_overlap(shift_i, shift_j)
        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)
        
        # for 1e- this is just one body operator
        ferm_ferm_contrib = npj.sum((ham.quad[0] * G[0] + ham.quad[1] * G[1]) * cs_ovlp)

        el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = npj.einsum('ijk,ij,ki,ij->', ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
            el_ph_contrib += npj.einsum('ijk,ij,kj,ij->', ham.g_tensor, G[0], shift_j, cs_ovlp)
        
        phonon_contrib = ham.w0 * npj.einsum('ij,j->', shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal()) 

        local_energy = kinetic + el_ph_contrib + phonon_contrib + ferm_ferm_contrib
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

        #TODO get total energy and overlap
        ovlp = 0.
        energy = 0.

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            perm_mat_symm = perm_mat + perm_mat.conj().T
            perm_mat_asym = perm_mat - perm_mat.conj().T
            
            fac_i = overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi
            
            #beta_i = shift[permi]
            beta_i = np.roll(shift, shift=(-ip,-ip), axis=(0,1))
            psia_i = c0a[permi] # [permi, :]

            # TODO Could store these matrices
            
            Ga_i = np.outer(c0a.conj(), psia_i)
            occ = np.sum(Ga_i.diagonal())
            cs_ovlp = self.cs_overlap(shift, beta_i)
            
            # Auxiliary Tensors
            kin_tensor = Ga_i * self.ham.T[0]
            kin = np.sum(kin_tensor * cs_ovlp)
            
            ovlp_i = np.einsum('i,i,i->', c0a.conj(), psia_i, cs_ovlp.diagonal())

            d_shift_perm_r = np.einsum('ei,pj->peij', np.eye(self.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
            d_shift_perm_r += np.einsum('pi,je->peij', perm_mat.T.dot(shift.conj()), perm_mat)
#            d_shift_perm_r = np.zeros_like(d_shift_perm_r)

            d_shift_perm_i = -1j * np.einsum('ei,pj->peij', np.eye(self.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
            d_shift_perm_i += 1j * np.einsum('pi,je->peij', perm_mat.T.dot(shift.conj()), perm_mat)
#            d_shift_perm_i = -1j * np.einsum('ei,pm,mn,nj->peij',np.eye(self.ham.N), perm_mat, shift,perm_mat.T)
#            d_shift_perm_i += 1j * np.einsum('ki,kp,ej->peij', shift.conj(),perm_mat, perm_mat.T)


            d_cs_ovlp_r = np.einsum('peij,ij->peij', d_shift_perm_r, cs_ovlp)
            d_cs_ovlp_r -= np.einsum('ei,pi,ij->peij', np.eye(self.ham.N), shift.real, cs_ovlp)
            d_cs_ovlp_r -= 0.5 * np.einsum('pb,ej,jb,ij->peij', shift, perm_mat.T, perm_mat, cs_ovlp)
            d_cs_ovlp_r -= 0.5 * np.einsum('pb,bj,je,ij->peij',shift.conj(), perm_mat.T, perm_mat, cs_ovlp)
#            d_cs_ovlp_r -= np.einsum('pb,bj,je,ij->peij', shift.real, perm_mat.T, perm_mat, cs_ovlp)

            d_cs_ovlp_i = np.einsum('peij,ij->peij', d_shift_perm_i, cs_ovlp)
            d_cs_ovlp_i -= np.einsum('ei,pi,ij->peij', np.eye(self.ham.N), shift.imag, cs_ovlp)
            d_cs_ovlp_i += 1j * 0.5 * np.einsum('pb,ej,jb,ij->peij', shift, perm_mat.T, perm_mat, cs_ovlp)
            d_cs_ovlp_i -= 1j * 0.5 * np.einsum('pb,bj,je,ij->peij',shift.conj(), perm_mat.T, perm_mat, cs_ovlp) # TODO
#            d_cs_ovlp_r -= np.einsum('pb,bj,je,ij->peij', shift.imag, perm_mat.T, perm_mat, cs_ovlp)


            # shift_grad_real contribs
            kin_contrib = np.einsum('ij,peij->pe', kin_tensor, d_cs_ovlp_r)
            
            elph_ij_contracted = np.einsum('ijk,ij->ijk', self.ham.g_tensor, Ga_i)
            el_ph_contrib = np.einsum('ijk,ki,peij->pe', elph_ij_contracted, shift.conj(), d_cs_ovlp_r)
            el_ph_contrib += np.einsum('ijk,kj,peij->pe', elph_ij_contracted, beta_i, d_cs_ovlp_r) 
            el_ph_contrib += np.einsum('ejp,ej->pe', elph_ij_contracted , cs_ovlp)
            el_ph_contrib += np.einsum('ijk,ij,kp,ej->pe', elph_ij_contracted, cs_ovlp, perm_mat, perm_mat.T)

            
            boson_contrib = np.einsum('ii,ki,peii->pe', Ga_i, shift.conj() * beta_i, d_cs_ovlp_r)
            boson_contrib += np.einsum('ii,peii,ii->pe', Ga_i, d_shift_perm_r, cs_ovlp)
            boson_contrib *= self.ham.w0

            ovlp_contrib = np.einsum('i,peii->pe', Ga_i.diagonal(), d_cs_ovlp_r)
            sgr = (kin_contrib + el_ph_contrib + boson_contrib)
            sgr_ovlp = ovlp_contrib 

            # shift_grad_imag contribs
            kin_contrib = np.einsum('ij,peij->pe', kin_tensor, d_cs_ovlp_i)
            
            el_ph_contrib = np.einsum('ijk,ki,peij->pe', elph_ij_contracted, shift.conj(), d_cs_ovlp_i) # TODO contract over k already to 2 index object
            el_ph_contrib += np.einsum('ijk,kj,peij->pe', elph_ij_contracted, beta_i, d_cs_ovlp_i)  
            el_ph_contrib -= 1j * np.einsum('ejp,ej->pe', elph_ij_contracted, cs_ovlp)
            el_ph_contrib += 1j * np.einsum('ijk,ij,kp,ej->pe', elph_ij_contracted, cs_ovlp, perm_mat, perm_mat.T) 
            
            boson_contrib = np.einsum('ii,ki,peii->pe', Ga_i, shift.conj() * beta_i, d_cs_ovlp_i)
            boson_contrib += np.einsum('ii,peii,ii->pe', Ga_i, d_shift_perm_i, cs_ovlp)
            boson_contrib *= self.ham.w0

            ovlp_contrib = np.einsum('i,peii->pe', Ga_i.diagonal(), d_cs_ovlp_i)
            sgi = (kin_contrib + el_ph_contrib + boson_contrib)
            sgi_ovlp = ovlp_contrib


            # psia_grad_real contribs
            g_contracted = np.einsum('ijk,ki->ij', self.ham.g_tensor, shift.conj())
            g_contracted += np.einsum('ijk,kj->ij', self.ham.g_tensor, beta_i)
            g_contracted = g_contracted * cs_ovlp
            g_contracted = g_contracted.dot(perm_mat)
            w_contracted = self.ham.w0 * np.einsum('ij,j,jn->jn', shift.conj() * beta_i, cs_ovlp.diagonal(), perm_mat)
            ovlp_contracted = np.einsum('i,ij->ij', cs_ovlp.diagonal(), perm_mat)

            kin_perm = (cs_ovlp * self.ham.T[0]).dot(perm_mat)

            kin_contrib = (kin_perm + kin_perm.T).dot(c0a.real) + 1j * (kin_perm - kin_perm.T).dot(c0a.imag)
            el_ph_contrib = (g_contracted + g_contracted.T).dot(c0a.real) + 1j * (g_contracted - g_contracted.T).dot(c0a.imag)
            boson_contrib = (w_contracted + w_contracted.T).dot(c0a.real) + 1j * (w_contracted - w_contracted.T).dot(c0a.imag)
            
            ovlp_contrib = (ovlp_contracted + ovlp_contracted.T).dot(c0a.real) + 1j * (ovlp_contracted - ovlp_contracted.T).dot(c0a.imag)
            pgr = (kin_contrib + el_ph_contrib + boson_contrib)
            pgr_ovlp = ovlp_contrib

            # psia_grad_imag contribs
            kin_contrib = (kin_perm + kin_perm.T).dot(c0a.imag) - 1j * (kin_perm - kin_perm.T).dot(c0a.real)
            el_ph_contrib = (g_contracted + g_contracted.T).dot(c0a.imag) - 1j * (g_contracted - g_contracted.T).dot(c0a.real)
            boson_contrib = (w_contracted + w_contracted.T).dot(c0a.imag) - 1j * (w_contracted - w_contracted.T).dot(c0a.real)

            ovlp_contrib = (ovlp_contracted + ovlp_contracted.T).dot(c0a.imag) - 1j * (ovlp_contracted - ovlp_contracted.T).dot(c0a.real)
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

            eph_energy = np.einsum('ijk,ij,ki,ij->', self.ham.g_tensor, Ga_i, shift.conj(), cs_ovlp)
            eph_energy += np.einsum('ijk,ij,kj,ij->', self.ham.g_tensor, Ga_i, beta_i, cs_ovlp)
            ph_energy = self.ham.w0 * npj.einsum('ij,j->', shift.conj() * beta_i, cs_ovlp.diagonal() * Ga_i.diagonal()) 
            energy += (fac_i * (kin + eph_energy + ph_energy)).real
            ovlp += (fac_i * ovlp_i).real
        
        shift_grad = (shift_grad_real + 1j * shift_grad_imag).ravel()
        psia_grad = (psia_grad_real + 1j * psia_grad_imag).ravel()
        shift_grad_ovlp = (shift_grad_real_ovlp + 1j * shift_grad_imag_ovlp).ravel()
        psia_grad_ovlp = (psia_grad_real_ovlp + 1j * psia_grad_imag_ovlp).ravel()

        dx_energy = self.pack_x(shift_grad, psia_grad)
        dx_ovlp = self.pack_x(shift_grad_ovlp, psia_grad_ovlp)

        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp ** 2
        print('my grad: ', dx)
        super().gradient(x)
        exit()
        return dx


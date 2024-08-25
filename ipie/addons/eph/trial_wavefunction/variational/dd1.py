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

import jax
import jax.numpy as npj
import numpy as np
import plum

from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectron
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import (
    overlap_degeneracy,
    ToyozawaVariational,
)

from ipie.addons.eph.hamiltonians.dispersive_phonons import DispersivePhononModel
from numba import jit

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

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi, :]

            cs_ovlp = self.cs_overlap(shift, shift_i)

            overlap = npj.einsum("ie,ie,i->", c0a.conj(), psia_i, cs_ovlp.diagonal())
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
            num_energy += (
                projected_energy
                * overlap_degeneracy(self.ham, ip)
                * self.Kcoeffs[0].conj()
                * coeffi
            ).real
            denom += overlap.real

        energy = num_energy / denom
        # TODO
        return energy.real

    def get_args(self):
        return ()

    def cs_overlap(self, shift_i, shift_j):
        cs_ovlp = npj.exp(shift_i.T.conj().dot(shift_j))
        cs_ovlp_norm_i = npj.exp(-0.5 * npj.einsum("ij->j", npj.abs(shift_i) ** 2))
        cs_ovlp_norm_j = npj.exp(-0.5 * npj.einsum("ij->j", npj.abs(shift_j) ** 2))
        cs_ovlp_norm = npj.outer(cs_ovlp_norm_i, cs_ovlp_norm_j)
        cs_ovlp *= cs_ovlp_norm
        return cs_ovlp

    @plum.dispatch
    def projected_energy(self, ham: GenericEPhModel, G: list, shift_i, shift_j):

        cs_ovlp = self.cs_overlap(shift_i, shift_j)

        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)

        el_ph_contrib = npj.einsum("ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = npj.einsum(
                "ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp
            )
            el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)

        phonon_contrib = ham.w0 * npj.einsum(
            "ij,j->", shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal()
        )

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

        el_ph_contrib = npj.einsum("ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
        el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)
        if self.sys.ndown > 0:
            el_ph_contrib = npj.einsum(
                "ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp
            )
            el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)

        phonon_contrib = ham.w0 * npj.einsum(
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

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            perm_mat = np.roll(np.eye(self.ham.N), shift=-ip, axis=0)
            perm_mat_symm = perm_mat + perm_mat.conj().T
            perm_mat_asym = perm_mat - perm_mat.conj().T

            fac_i = overlap_degeneracy(self.ham, ip) * self.Kcoeffs[0].conj() * coeffi

            beta_i = np.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            psia_i = c0a[permi]  # [permi, :]

            # TODO Could store these matrices

            Ga_i = np.outer(c0a.conj(), psia_i)
            occ = np.sum(Ga_i.diagonal())
            cs_ovlp = self.cs_overlap(shift, beta_i)

            # Auxiliary Tensors
            kin_tensor = Ga_i * self.ham.T[0]
            kin = np.sum(kin_tensor * cs_ovlp)

            ovlp_i = np.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())

            # shift_grad_real contribs
            kin_csovlp = kin_tensor * cs_ovlp
            kin_contrib = beta_i.dot(kin_csovlp.T)
            kin_contrib += perm_mat.T.dot(shift.conj()).dot(kin_csovlp).dot(perm_mat) # TODO this rolls both indices of shift.conj().dot(kin_csovlp)
            kin_contrib -= shift.real * np.sum(kin_csovlp, axis=1)
#            kin_contrib -= (shift.real.dot(perm_mat.T) * np.sum(kin_csovlp, axis=0)).dot(perm_mat) # TODO this rolls sum kin_csovlp in both indices
            kin_contrib -= shift.real * np.roll(np.sum(kin_csovlp, axis=0), shift=ip)

            elph_ij_contracted = self.ham.g_tensor * Ga_i[..., None]
            elph_ij_contr_shift_cs = np.sum(elph_ij_contracted * shift.conj().T[:, None, :], axis=2) * cs_ovlp
            el_ph_contrib = beta_i.dot(elph_ij_contr_shift_cs.T)
            el_ph_contrib += perm_mat.T.dot(shift.conj()).dot(elph_ij_contr_shift_cs).dot(perm_mat) # NOTE If there is translational invariance, are all terms of H translationally invariant?
            el_ph_contrib -= shift.real * np.sum(elph_ij_contr_shift_cs, axis=1)
            el_ph_contrib -= (shift.real.dot(perm_mat.T) * np.sum(elph_ij_contr_shift_cs ,axis=0)).dot(perm_mat)

            elph_ij_contracted_beta_cs = np.sum(elph_ij_contracted * beta_i.T[None, ...], axis=2) * cs_ovlp
            el_ph_contrib += beta_i.dot(elph_ij_contracted_beta_cs.T)
            el_ph_contrib += perm_mat.T.dot(shift.conj()).dot(elph_ij_contracted_beta_cs).dot(perm_mat)
            el_ph_contrib -= shift.real * np.sum(elph_ij_contracted_beta_cs, axis=1)           
            el_ph_contrib -= (shift.real.dot(perm_mat.T) * np.sum(elph_ij_contracted_beta_cs,axis=0)).dot(perm_mat)

            el_ph_contrib += np.sum(elph_ij_contracted * cs_ovlp[...,None], axis=1).T
            el_ph_contrib += np.sum((elph_ij_contracted * cs_ovlp[...,None]).dot(perm_mat).swapaxes(1,2).dot(perm_mat), axis=0)

            Ga_shifts = Ga_i.diagonal() * cs_ovlp.diagonal()
            ovlp_contrib = beta_i * Ga_shifts
            ovlp_contrib += (perm_mat.T.dot(shift.conj()) * Ga_shifts).dot(perm_mat)
            ovlp_contrib -= shift.real * Ga_shifts
            ovlp_contrib -= (shift.real.dot(perm_mat.T) * Ga_shifts).dot(perm_mat)

            Ga_shifts *= np.sum(shift.conj() * beta_i, axis=0)
            boson_contrib = beta_i * Ga_shifts
            boson_contrib += (perm_mat.T.dot(shift.conj()) * Ga_shifts).dot(perm_mat)
            boson_contrib -= shift.real * Ga_shifts
            boson_contrib -= (shift.real.dot(perm_mat.T) * Ga_shifts).dot(perm_mat)
            boson_contrib += (beta_i * Ga_i.diagonal() * cs_ovlp.diagonal())
            boson_contrib += (perm_mat.T.dot(shift.conj()) * Ga_i.diagonal() * cs_ovlp.diagonal()).dot(perm_mat)
            boson_contrib *= self.ham.w0

            sgr = kin_contrib + el_ph_contrib + boson_contrib
            sgr_ovlp = ovlp_contrib

            # shift_grad_imag contribs
            kin_contrib = -1j * beta_i.dot(kin_csovlp.T)
            kin_contrib += 1j * perm_mat.T.dot(shift.conj()).dot(kin_csovlp).dot(perm_mat)
            kin_contrib -= shift.imag * np.sum(kin_csovlp, axis=1)
            kin_contrib -= (shift.imag.dot(perm_mat.T) * np.sum(kin_csovlp, axis=0)).dot(perm_mat)

            el_ph_contrib = -1j * beta_i.dot(elph_ij_contr_shift_cs.T)
            el_ph_contrib += 1j * perm_mat.T.dot(shift.conj()).dot(elph_ij_contr_shift_cs).dot(perm_mat)
            el_ph_contrib -= shift.imag * np.sum(elph_ij_contr_shift_cs, axis=1)
            el_ph_contrib -= (shift.imag.dot(perm_mat.T) * np.sum(elph_ij_contr_shift_cs, axis=0)).dot(perm_mat)

            el_ph_contrib += -1j * beta_i.dot(elph_ij_contracted_beta_cs.T)
            el_ph_contrib += 1j * perm_mat.T.dot(shift.conj()).dot(elph_ij_contracted_beta_cs).dot(perm_mat)
            el_ph_contrib -= shift.imag * np.sum(elph_ij_contracted_beta_cs, axis=1)
            el_ph_contrib -= (shift.imag.dot(perm_mat.T) * np.sum(elph_ij_contracted_beta_cs,axis=0)).dot(perm_mat)


            el_ph_contrib -= 1j * np.sum(elph_ij_contracted * cs_ovlp[...,None], axis=1).T
            el_ph_contrib += 1j *np.sum((elph_ij_contracted * cs_ovlp[...,None]).dot(perm_mat).swapaxes(1,2).dot(perm_mat), axis=0)

            Ga_shifts = Ga_i.diagonal() * cs_ovlp.diagonal()
            ovlp_contrib = -1j * beta_i * Ga_shifts
            ovlp_contrib += 1j *(perm_mat.T.dot(shift.conj()) * Ga_shifts).dot(perm_mat)
            ovlp_contrib -= shift.imag * Ga_shifts
            ovlp_contrib -= (shift.imag.dot(perm_mat.T) * Ga_shifts).dot(perm_mat)
            
            Ga_shifts *= np.sum(shift.conj() * beta_i, axis=0)
            boson_contrib = -1j * beta_i * Ga_shifts
            boson_contrib += 1j * (perm_mat.T.dot(shift.conj()) * Ga_shifts).dot(perm_mat)
            boson_contrib -= shift.imag * Ga_shifts
            boson_contrib -= (shift.imag.dot(perm_mat.T) * Ga_shifts).dot(perm_mat)


            boson_contrib -= 1j * (beta_i * Ga_i.diagonal() * cs_ovlp.diagonal())
            boson_contrib += 1j * (perm_mat.T.dot(shift.conj()) * Ga_i.diagonal() * cs_ovlp.diagonal()).dot(perm_mat)
            boson_contrib *= self.ham.w0

            sgi = kin_contrib + el_ph_contrib + boson_contrib
            sgi_ovlp = ovlp_contrib

            # psia_grad_real contribs
            g_contracted = np.sum(self.ham.g_tensor * shift.conj().T[:, None, :], axis=2)
            g_contracted += np.sum(self.ham.g_tensor * beta_i.T[None, ...], axis=2)
            g_contracted *= cs_ovlp
            g_contracted = g_contracted.dot(perm_mat)
            w_contracted = self.ham.w0 * perm_mat.T * (np.sum(shift.conj() * beta_i, axis=0) * cs_ovlp.diagonal())
            ovlp_contracted = perm_mat.T * cs_ovlp.diagonal()

            kin_perm = (cs_ovlp * self.ham.T[0]).dot(perm_mat)

            kin_contrib = (kin_perm + kin_perm.T).dot(c0a.real) + 1j * (kin_perm - kin_perm.T).dot(
                c0a.imag
            )
            el_ph_contrib = (g_contracted + g_contracted.T).dot(c0a.real) + 1j * (
                g_contracted - g_contracted.T
            ).dot(c0a.imag)
            boson_contrib = (w_contracted + w_contracted.T).dot(c0a.real) + 1j * (
                w_contracted.T - w_contracted
            ).dot(c0a.imag)

            ovlp_contrib = (ovlp_contracted + ovlp_contracted.T).dot(c0a.real) + 1j * (
                ovlp_contracted.T - ovlp_contracted
            ).dot(c0a.imag)
            pgr = kin_contrib + el_ph_contrib + boson_contrib
            pgr_ovlp = ovlp_contrib

            # psia_grad_imag contribs
            kin_contrib = (kin_perm + kin_perm.T).dot(c0a.imag) - 1j * (kin_perm - kin_perm.T).dot(
                c0a.real
            )
            el_ph_contrib = (g_contracted + g_contracted.T).dot(c0a.imag) - 1j * (
                g_contracted - g_contracted.T
            ).dot(c0a.real)
            boson_contrib = (w_contracted + w_contracted.T).dot(c0a.imag) - 1j * (
                w_contracted.T - w_contracted
            ).dot(c0a.real)

            ovlp_contrib = (ovlp_contracted + ovlp_contracted.T).dot(c0a.imag) - 1j * (
                ovlp_contracted.T - ovlp_contracted
            ).dot(c0a.real)
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

            eph_energy = np.sum(self.ham.g_tensor * (Ga_i * cs_ovlp)[..., None] * shift.conj().T[:, None, :])
            eph_energy += np.sum(self.ham.g_tensor * (Ga_i * cs_ovlp)[..., None] * beta_i.T[None, ...])
            ph_energy = self.ham.w0 * np.sum((shift.conj() * beta_i).dot(cs_ovlp.diagonal() * Ga_i.diagonal()))
            energy += (fac_i * (kin + eph_energy + ph_energy)).real
            ovlp += (fac_i * ovlp_i).real

        shift_grad = (shift_grad_real + 1j * shift_grad_imag).ravel()
        psia_grad = (psia_grad_real + 1j * psia_grad_imag).ravel()
        shift_grad_ovlp = (shift_grad_real_ovlp + 1j * shift_grad_imag_ovlp).ravel()
        psia_grad_ovlp = (psia_grad_real_ovlp + 1j * psia_grad_imag_ovlp).ravel()

        dx_energy = self.pack_x(shift_grad, psia_grad)
        dx_ovlp = self.pack_x(shift_grad_ovlp, psia_grad_ovlp)

        dx = dx_energy / ovlp - dx_ovlp * energy / ovlp**2
        print('my grad: ', dx)
        return dx

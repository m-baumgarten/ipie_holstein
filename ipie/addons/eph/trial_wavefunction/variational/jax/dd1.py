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
from ipie.addons.eph.trial_wavefunction.variational.jax.toyozawa import (
    overlap_degeneracy,
    ToyozawaVariational,
)

from ipie.addons.eph.hamiltonians.dispersive_phonons import DispersivePhononModel
from ipie.addons.eph.trial_wavefunction.variational.dd1 import dD1Variational as dd1

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
        self.dd1 = dd1(shift_init, electron_init, hamiltonian, system, K, cplx)

    def objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = shift[0]
        c0a = c0a[:,0]

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi]

            cs_ovlp = self.cs_overlap(shift, shift_i)

#            overlap = np.einsum("ie,ie,i->", c0a.conj(), psia_i, cs_ovlp.diagonal())
            overlap = npj.sum(c0a.conj() * psia_i * cs_ovlp.diagonal())
            overlap *= self.Kcoeffs[0].conj() * coeffi

#            if npj.abs(overlap) < zero_th:
#                continue

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
#            print(f'energy & ovlp {ip}', projected_energy, overlap, overlap_degeneracy(self.ham, ip))
#            jax.debug.print('energy & ovlp {x}', x = (ip, projected_energy, overlap, overlap_degeneracy(self.ham, ip)))
#        print('num_energy:  ', num_energy)
#        print('denom:   ', denom)
        energy = num_energy / denom
        # TODO
#        jax.debug.print('energy:    {x}', x=(energy, num_energy, denom))
#        exit()
        return energy.real

    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
#        print('shift:   ', shift)
#        print('c0a: ', c0a)
#        print('c0b: ', c0b)
#        exit()

        shift = shift[0]
        c0a = c0a[:,0]

        num_energy = 0.0
        denom = 0.0

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi]

            for jp, (permj, coeffj) in enumerate(zip(self.perms, self.Kcoeffs)):
                shift_j = npj.roll(shift, shift=(-jp, -jp), axis=(0, 1))
                psia_j = c0a[permj]

                cs_ovlp = self.cs_overlap(shift_j, shift_i)
#                jax.debug.print('cs ovlp:   {x}', x=cs_ovlp)

                overlap = npj.sum(psia_j.conj() * psia_i * cs_ovlp.diagonal())
                overlap *= coeffj.conj() * coeffi

#                if npj.abs(overlap) < zero_th:
#                    continue

                Ga_i = npj.outer(psia_j.conj(), psia_i)
                if self.sys.ndown > 0:
                    Gb_i = npj.outer(c0b.conj(), psib_i)
                else:
                    Gb_i = npj.zeros_like(Ga_i)
                G_i = [Ga_i, Gb_i]

                projected_energy = self.projected_energy(self.ham, G_i, shift_j, shift_i)
                num_energy += (
                    projected_energy
                    * coeffj.conj()
                    * coeffi
                )
                denom += overlap
#                jax.debug.print('energy & ovlp {x}', x = (ip, jp, projected_energy, overlap))
#        print('num_energy:  ', num_energy)
#        print('denom:   ', denom)
        energy = num_energy / denom
        # TODO
#        jax.debug.print('energy:    {x}', x=(energy, num_energy, denom))
        exit()
        return energy.real
    
    def _objective_function(self, x, zero_th: float = 1e-12) -> float:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
#        print('shift:   ', shift)
#        print('c0a: ', c0a)
#        print('c0b: ', c0b)
#        exit()

        shift = shift[0]
        c0a = c0a[:,0]

        num_energy = 0.0
        denom = 0.0

        num_energy = npj.zeros((self.nperms, self.nperms), npj.complex128)
        denom = npj.zeros_like(num_energy, dtype=npj.complex128)

        for ip, (permi, coeffi) in enumerate(zip(self.perms, self.Kcoeffs)):
            shift_i = npj.roll(shift, shift=(-ip, -ip), axis=(0, 1))
            # Alternatively shift_j = shift[permj,:][:,permj]
            psia_i = c0a[permi]

            for jp, (permj, coeffj) in enumerate(zip(self.perms, self.Kcoeffs)):
                shift_j = npj.roll(shift, shift=(-jp, -jp), axis=(0, 1))
                psia_j = c0a[permj]

                cs_ovlp = self.cs_overlap(shift_j, shift_i)
#                jax.debug.print('cs ovlp:   {x}', x=cs_ovlp)

                log_factor = npj.log(coeffj.conj()) + npj.log(coeffi)

                overlap = npj.log(npj.sum(psia_j.conj() * psia_i * cs_ovlp.diagonal()))
                overlap += log_factor

#                if npj.abs(overlap) < zero_th:
#                    continue

                Ga_i = npj.outer(psia_j.conj(), psia_i)
                if self.sys.ndown > 0:
                    Gb_i = npj.outer(c0b.conj(), psib_i)
                else:
                    Gb_i = npj.zeros_like(Ga_i)
                G_i = [Ga_i, Gb_i]

                projected_energy = self.projected_energy(self.ham, G_i, shift_j, shift_i)
                num_energy = num_energy.at[jp, ip].set((npj.log(projected_energy) + log_factor))
#                num_energy += (
#                    projected_energy
#                    * coeffj.conj()
#                    * coeffi
#                )
#                denom += overlap
                denom = denom.at[jp, ip].set(overlap)
#                jax.debug.print('engery & denom:  {x}', x=(npj.exp(num_energy[jp,ip]), npj.exp(denom[jp,ip])))

#                jax.debug.print('energy & ovlp {x}', x = (ip, jp, projected_energy, overlap))
#        print('num_energy:  ', num_energy)
#        print('denom:   ', denom)
        max_factor_abs = max(npj.max(npj.abs(num_energy)), npj.max(npj.abs(denom)))   
#        jax.debug.print('max fac: {x}', x=(max_factor_abs, num_energy, npj.abs(num_energy), denom, npj.abs(denom))) 
        energy = npj.sum(npj.exp(num_energy - max_factor_abs)) / npj.sum(npj.exp(denom - max_factor_abs))
        
#        energy = num_energy / denom
        # TODO
#        jax.debug.print('energy:    {x}', x=(energy, num_energy, denom))
#        exit()
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
#        jax.debug.print('contribs:  {x}', x=(kinetic, el_ph_contrib, phonon_contrib))
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
        jax.debug.print('ea:')
        return local_energy

#    @plum.dispatch
#    def projected_energy(self, ham: ExcitonPhononCavityElectron, G: list, shift_i, shift_j):
#        cs_ovlp = self.cs_overlap(shift_i, shift_j)
#        kinetic = npj.sum((ham.T[0] * G[0] + ham.T[1] * G[1]) * cs_ovlp)
#
        # for 1e- this is just one body operator
##        ferm_ferm_contrib = npj.sum((ham.quad[0] * G[0] + ham.quad[1] * G[1]) * cs_ovlp)
#
#        el_ph_contrib = npj.einsum("ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp)
#        el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)
#        if self.sys.ndown > 0:
#            el_ph_contrib = npj.einsum(
#                "ijk,ij,ki,ij->", ham.g_tensor, G[0], shift_i.conj(), cs_ovlp
#            )
#            el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", ham.g_tensor, G[0], shift_j, cs_ovlp)
#
#        phonon_contrib = ham.w0 * npj.einsum(
#            "ij,j->", shift_i.conj() * shift_j, cs_ovlp.diagonal() * G[0].diagonal()
#        )
#
#        local_energy = kinetic + el_ph_contrib + phonon_contrib + ferm_ferm_contrib
#        return local_energy


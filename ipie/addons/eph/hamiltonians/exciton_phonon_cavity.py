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

import numpy
from typing import Sequence
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.utils.backend import arraylib as xp


class ExcitonPhononCavityElectron(HolsteinModel):
    """Class for exciton phonon cavity model carrying elph tensor and system parameters"""

    def __init__(
        self,
        g: float,
        ge: float,
        te: float,
        w0: float,
        wtilde: float,
        je: float,
        nsites: int,
        pbc: bool,
    ):
        super().__init__(g=g, t=te, w0=w0, nsites=nsites, pbc=pbc)
        assert pbc
        self.ge = ge
        self.te = self.t
        self.wtilde = wtilde
        self.je = je
        self.c = 0.5 * (g * je / wtilde) ** 2
        self.quad_c = -self.c * 2 * wtilde

        # TODO are these the correct constants?
        self.m = 1 / self.w0
        self.Xconst = numpy.sqrt(2.0 * self.m * self.w0)

    def build(self) -> None:
        super().build()
        self.quad = self.build_quadratic()

    def build_quadratic(self) -> Sequence[numpy.array]:
        """"""
        quad = numpy.diag(numpy.ones(self.nsites - 2), 2)
        quad += numpy.diag(numpy.ones(self.nsites - 2), -2)
        quad -= 2 * numpy.eye(self.nsites)

        if self.pbc:
            quad[0, -2] = quad[-2, 0] = 1.0
            quad[1, -1] = quad[-1, 1] = 1.0

        quad *= -((self.g * self.je) ** 2) / self.wtilde

        quad = [quad.copy(), quad.copy()]
        return quad

    def build_g(self) -> numpy.ndarray:
        """"""
        g_tensor = numpy.zeros((self.nsites, self.nsites, self.nsites), dtype=numpy.complex128)
        for site in range(self.nsites):
            g_tensor[site, site, site] = (1 - self.c)
            g_tensor[site, site, (site+1) % self.nsites] = - 0.5 * self.c
            g_tensor[site, site, (site-1) % self.nsites] = - 0.5 * self.c
            
            g_tensor[site, (site+2) % self.nsites, site] = 0.25 * self.c
            g_tensor[(site+2) % self.nsites, site, site] = 0.25 * self.c
            g_tensor[site, (site+2) % self.nsites, (site+2) % self.nsites] = 0.25 * self.c
            g_tensor[(site+2) % self.nsites, site, (site+2) % self.nsites] = 0.25 * self.c
            g_tensor[site, (site+2) % self.nsites, (site+1) % self.nsites] = 0.5 * self.c
            g_tensor[(site+2) % self.nsites, site, (site+1) % self.nsites] = 0.5 * self.c
        
        g_tensor *= -self.ge * self.Xconst
        return g_tensor

class ExcitonPhononCavityHole(ExcitonPhononCavityElectron):
    def __init__(
        self,
        g: float,
        ge: float,
        te: float,
        w0: float,
        wtilde: float,
        je: float,
        nsites: int,
        pbc: bool,
    ):
        super().__init__(g, ge, te, w0, wtilde, je, nsites, pbc)


class ExcitonPhononCavityElectronHole(ExcitonPhononCavityElectron):
    def __init__(
        self,
        g: float,
        ge: float,
        gh: float,
        te: float,
        th: float,
        w0: float,
        wtilde: float,
        je: float,
        jh: float,
        u0: float,
        u1: float,
        L: float,
        e: float,
        nsites: int,
        pbc: bool,
    ):
        super().__init__(g, ge, te, w0, wtilde, je, nsites, pbc)
        assert u0 >= u1 >= 0
        self.ce = self.c
        self.ch = 0.5 * (g * jh / wtilde)**2 
        self.ceh = self.je * jh * (g / wtilde)**2
        self.gh = gh
        self.th = th
        self.jh = jh
        self.u0 = u0
        self.u1 = u1
        self.L = L
        self.e = e

    def build_quadratic(self) -> Sequence[numpy.array]:
        """"""
        quade = super().build_quadratic()
        quadh = [quade[0].copy() * (self.jh / self.je)**2, quade[1].copy() * (self.jh / self.je)**2]

        return quade, quadh
    
    def build_T(self) -> Sequence[numpy.ndarray]:
        """Constructs electronic hopping matrix."""
        Te = super().build_T()
        Th = [Te[0].copy() * (self.th / self.te), Te[1].copy() * (self.th / self.te)]
        return Te, Th

    def build_g(self) -> Sequence[numpy.ndarray]:
        """"""
        g_tensor_e = super().build_g()
        
        g_tensor_h = numpy.zeros((self.nsites, self.nsites, self.nsites), dtype=numpy.complex128)
        for site in range(self.nsites):
            # should check whether we need += TODO
            g_tensor_h[site, site, site] += (1 - self.ch)
            g_tensor_h[site, site, (site+1) % self.nsites] += - 0.5 * self.ch
            g_tensor_h[site, site, (site-1) % self.nsites] += - 0.5 * self.ch
            
            g_tensor_h[site, (site+2) % self.nsites, site] += 0.25 * self.ch
            g_tensor_h[(site+2) % self.nsites, site, site] += 0.25 * self.ch
            g_tensor_h[site, (site+2) % self.nsites, (site+2) % self.nsites] += 0.25 * self.ch
            g_tensor_h[(site+2) % self.nsites, site, (site+2) % self.nsites] += 0.25 * self.ch
            g_tensor_h[site, (site+2) % self.nsites, (site+1) % self.nsites] += 0.5 * self.ch
            g_tensor_h[(site+2) % self.nsites, site, (site+1) % self.nsites] += 0.5 * self.ch
        g_tensor_h *= -self.gh * self.Xconst

        return g_tensor_e, g_tensor_h

    def build_elhole_interaction(self) -> numpy.ndarray:
        coulomb_site = numpy.zeros((self.nsites, self.nsites), dtype=numpy.complex128)
        for site_i in range(self.nsites):
            for site_j in range(site_i+1, self.nsites):
                coulomb_site[site_i, site_j] = self.u1 * 1 / numpy.min([numpy.abs(site_i - site_j), numpy.abs(site_i - site_j + self.nsites)])
        coulomb_site += coulomb_site.T
        coulomb_site += numpy.eye(self.nsites) * self.u0

        elhole_tensor = numpy.zeros((self.nsites, self.nsites, self.nsites, self.nsites), dtype=numpy.complex128)
        for site_i in range(self.nsites):
            sp1_i = (site_i+1) % self.nsites
            sp2_i = (site_i+2) % self.nsites
            sm1_i = (site_i-1) % self.nsites
            sm2_i = (site_i-2) % self.nsites
            

            for site_j in range(self.nsites):
                sp1_j = (site_j+1) % self.nsites
                sp2_j = (site_j+2) % self.nsites
                sm1_j = (site_j-1) % self.nsites
                sm2_j = (site_j-2) % self.nsites
                    
                
                # Contribution from (I)
                elhole_tensor[site_i, site_i, site_j, site_j] += coulomb_site[site_i, site_j]

                # Contribution from (II)
                elhole_tensor[site_i, site_i, site_j, site_j] -= 1. * self.ce * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, sp2_i, site_j, site_j] -= -0.25 * self.ce * coulomb_site[site_i, sm2_j]
                elhole_tensor[site_i, sm2_i, site_j, site_j] -= -0.25 * self.ce * coulomb_site[site_i, sp2_j]
                elhole_tensor[site_i, sp2_i, site_j, site_j] -= -0.25 * self.ce * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, sm2_i, site_j, site_j] -= -0.25 * self.ce * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, site_i, site_j, site_j] -= 0.5 * self.ce * coulomb_site[site_i, sm1_j]
                elhole_tensor[site_i, site_i, site_j, site_j] -= 0.5 * self.ce * coulomb_site[site_i, sp1_j]
                elhole_tensor[site_i, sp2_i, site_j, site_j] -= -0.5 * self.ce * coulomb_site[site_i, sm1_j]
                elhole_tensor[site_i, sm2_i, site_j, site_j] -= -0.5 * self.ce * coulomb_site[site_i, sp1_j]

                # Contribution from (III)
                elhole_tensor[site_i, site_i, site_j, site_j] -= 1. * self.ch * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, site_i, site_j, sp2_j] -= -0.25 * self.ch * coulomb_site[site_i, sp2_j]
                elhole_tensor[site_i, site_i, site_j, sm2_j] -= -0.25 * self.ch * coulomb_site[site_i, sm2_j]
                elhole_tensor[site_i, site_i, site_j, sp2_j] -= -0.25 * self.ch * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, site_i, site_j, sm2_j] -= -0.25 * self.ch * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, site_i, site_j, site_j] -= 0.5 * self.ch * coulomb_site[site_i, sm1_j]
                elhole_tensor[site_i, site_i, site_j, site_j] -= 0.5 * self.ch * coulomb_site[site_i, sp1_j]
                elhole_tensor[site_i, site_i, site_j, sp2_j] -= -0.5 * self.ch * coulomb_site[site_i, sp1_j]
                elhole_tensor[site_i, site_i, site_j, sm2_j] -= -0.5 * self.ch * coulomb_site[site_i, sm1_j] 

                # Contribution from (IV)
                elhole_tensor[site_i, sp1_i, site_j, sm1_j] += 0.25 * self.ceh * coulomb_site[site_i, sm2_j]
                elhole_tensor[site_i, sm1_i, site_j, sp1_j] += 0.25 * self.ceh * coulomb_site[site_i, sp2_j]
                elhole_tensor[site_i, sp1_i, site_j, sm1_j] += 0.25 * self.ceh * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, sm1_i, site_j, sp1_j] += 0.25 * self.ceh * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, sp1_i, site_j, sp1_j] += -0.25 * self.ceh * coulomb_site[site_i, sp1_j]
                elhole_tensor[site_i, sm1_i, site_j, sm2_j] += -0.25 * self.ceh * coulomb_site[site_i, sm1_j]
                elhole_tensor[site_i, sp1_i, site_j, sp1_j] += -0.25 * self.ceh * coulomb_site[site_i, sm1_j]
                elhole_tensor[site_i, sm1_i, site_j, sm1_j] += -0.25 * self.ceh * coulomb_site[site_i, sp1_j]
                elhole_tensor[site_i, sp1_i, site_j, sm1_j] += -0.5 * self.ceh * coulomb_site[site_i, site_j]
                elhole_tensor[site_i, sm1_i, site_j, sp1_j] += -0.5 * self.ceh * coulomb_site[site_i, site_j]

        elhole_tensor *=  - self.e**2 / (2 * self.L)

        return elhole_tensor

    def build(self):
        """"""
        self.Te, self.Th = self.build_T()
        self.quade, self.quadh = self.build_quadratic()
        self.ge_tensor, self.gh_tensor = self.build_g()
        self.elhole_tensor = self.build_elhole_interaction()


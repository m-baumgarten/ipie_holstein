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
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
from ipie.utils.backend import arraylib as xp

class BondSSHModel(HolsteinModel):
    """Class for SSH model carrying elph tensor and system parameters
    """
    def build_g(self) -> numpy.ndarray:
        """"""
        g_tensor = numpy.zeros((self.N, self.N, self.N), dtype=numpy.complex128)
        for site in range(self.N):
            g_tensor[(site+1) % self.N, site, site] = 1.
            g_tensor[site, (site+1) % self.N, site] = 1.
        g_tensor *= self.g
        return g_tensor

class OpticalSSHModel(HolsteinModel):
    def build_g(self) -> numpy.ndarray:
        """"""
        g_tensor = numpy.zeros((self.N, self.N, self.N), dtype=numpy.complex128)
        for site in range(self.N):
            i = site
            j = (site+1) % self.N
            g_tensor[j, i, i] = -1
            g_tensor[i, j, i] = -1
            g_tensor[j, i, j] = 1.
            g_tensor[i, j, j] = 1.
        g_tensor *= self.g
        return g_tensor

class DualCouplingModel(HolsteinModel):
    r"""Benchmarked with MA and DiagMC by Berciu in https://link.aps.org/doi/10.1103/PhysRevB.95.035117"""
    def __init__(self, g: float, g_ssh: float, t: float, w0: float, nsites: int, pbc: bool):
        super().__init__(g, t, w0, nsites, pbc)
        self.g_ssh = g_ssh

    def build_g(self) -> numpy.ndarray:
        """"""
        g_tensor = numpy.zeros((self.N, self.N, self.N), dtype=numpy.complex128)
        for site in range(self.N):
            g_tensor[site, site, site] = self.g
            g_tensor[(site+1) % self.N, site, site] = -1
            g_tensor[site, (site+1) % self.N, site] = -1
            g_tensor[(site+1) % self.N, site, (site+1) % self.N] = 1.
            g_tensor[site, (site+1) % self.N, (site+1) % self.N] = 1.
        g_tensor *= self.g_ssh
        return g_tensor

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


class BondSSHModel(GenericEPhModel):
    """Class for SSH model carrying elph tensor and system parameters
    """

    def __init__(self, g: float, v: float, w: float, 
                 w0: float, ncell: int, pbc: bool):
        """Each site corresponds to a unit cell of 2 atoms, 
        one on each sublattice A and B"""
        super().__init__(nsites=2*ncell, pbc=pbc)
        self.g = g
        self.v = v
        self.w = w
        self.w0 = w0
        self.m = 1/self.w0
        self.const = numpy.sqrt(2. * self.m * self.w0)

    def build(self):
        """"""
        self.T = self.build_T()
        self.g_tensor = self.build_g()
        self.X_connectivity = self.build_X_connectivity()

    def build_T(self) -> Sequence[numpy.ndarray]:
        """"""
        offdiags = numpy.ones(self.nsites-1)
        offdiags[::2] *= -self.v
        offdiags[1::2] *= -self.w

        T = numpy.diag(offdiags, 1)
        T += numpy.diag(offdiags, -1)
    
        if self.pbc:
            T[0,-1] = T[-1,0] = -self.w

        T = [T.copy(), T.copy()]
        return T

    def build_g(self) -> numpy.ndarray:
        """"""
        g_tensor = numpy.diag(numpy.ones(self.nsites-1), 1)
        g_tensor += numpy.diag(numpy.ones(self.nsites-1), -1)
        g_tensor *= -self.g
        if self.pbc:
            g_tensor[0,-1] = g_tensor[-1,0] = -self.g

        return g_tensor

    def build_X_connectivity(self) -> numpy.ndarray:
        """"""
        return numpy.eye(self.nsites)

class AcousticSSHModel(BondSSHModel):
    def __init__(self, g: float, v: float, w: float,
                 w0: float, ncell: int, pbc: bool):
        super().__init__(g, v, w, w0, ncell, pbc)

    def build_X_connectivity(self) -> numpy.ndarray:
        """"""
        X_connectivity = numpy.diag(-numpy.ones(self.nsites))
        X_connectivity += numpy.diag(numpy.ones(self.nsites-1), 1)
        if self.pbc:
            X_connectivity[-1, 0] = 1.0
#        X_connectivity = numpy.diag(numpy.ones(self.nsites-1), 1)
#        X_connectivity += numpy.diag(-numpy.ones(self.nsites-1), -1)
        return X_connectivity

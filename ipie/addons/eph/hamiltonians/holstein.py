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

class HolsteinModel(GenericEPhModel):
    r"""Class carrying parameters specifying a 1D Holstein chain.

    The Holstein model is described by the Hamiltonian

    .. math::
        \hat{H} = -t \sum_{\langle ij\rangle} \hat{a}_i^\dagger \hat{a}_j
        - g \sqrt{2 w_0 m} \sum_i \hat{a}_i^\dagger \hat{a}_i \hat{X}_i
        + \bigg(\sum_i \frac{m w_0^2}{2} \hat{X}_i^2 + \frac{1}{2m} \hat{P}_i^2
        - \frac{w_0}{2}\bigg),

    where :math:`t` is associated with the electronic hopping, :math:`g` with
    the electron-phonon coupling strength, and :math:``w_0` with the phonon
    frequency.

    Parameters
    ----------
    g : :class:`float`
        Electron-phonon coupling strength
    t : :class:`float`
        Electron hopping parameter
    w0 : :class:`float`
        Phonon frequency
    nsites : :class:`int`
        Length of the 1D Holstein chain
    pbc : :class:``bool`
        Boolean specifying whether periodic boundary conditions should be
        employed.
    """

    def __init__(self, g: float, t: float, w0: float, nsites: int, pbc: bool):
        super().__init__(nsites, pbc)
        self.g = g
        self.t = t
        self.w0 = w0
        self.m = 1 / self.w0
        self.const = numpy.sqrt(2.0 * self.m * self.w0)

    def build(self) -> None:
        """Sets electronic hopping, electron-phonon couling tensor and defines
        what lattice vibrations we are looking at."""
        self.T = self.build_T()
        self.g_tensor = self.build_g()

    def build_T_1D(self, nsites) -> Sequence[numpy.ndarray]:
        T = numpy.diag(numpy.ones(nsites - 1), 1)
        T += numpy.diag(numpy.ones(nsites - 1), -1)
        if self.pbc:
            T[0, -1] = T[-1, 0] = 1.0
        T *= -self.t
        return T

    def build_T(self) -> Sequence[numpy.ndarray]:
        """Constructs electronic hopping matrix."""
        if self.dim == 1:
            T = self.build_T_1D(self.nsites[0]) 

        if self.dim == 2:
            Tx = self.build_T_1D(self.nsites[0])
            Ty = self.build_T_1D(self.nsites[1])
            Ix = numpy.eye(self.nsites[0])
            Iy = numpy.eye(self.nsites[1])
            T = numpy.kron(Tx, Iy) + numpy.kron(Ix, Ty)

        if self.dim == 3:
            Tx = self.build_T_1D(self.nsites[0])
            Ty = self.build_T_1D(self.nsites[1])
            Tz = self.build_T_1D(self.nsites[2])
            Ix = numpy.eye(self.nsites[0])
            Iy = numpy.eye(self.nsites[1])
            Iz = numpy.eye(self.nsites[2])
            T = numpy.kron(Tx, Iy, Iz) + numpy.kron(Ix, Ty, Iz) + numpy.kron(Ix, Iy, Tz)

        T = [T.copy(), T.copy()]
        return T

    def build_g(self) -> numpy.ndarray:
        """Constructs the electron-phonon tensor. For the Holstein model"""
        g_tensor = numpy.zeros((self.N, self.N, self.N), dtype=numpy.complex128)
        for site in range(self.N):
            g_tensor[site, site, site] = 1.
        g_tensor *= self.g
        return g_tensor


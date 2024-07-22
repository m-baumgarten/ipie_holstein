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

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, qr, qr_mode, synchronize
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers


class EPhCSWalkers(EPhWalkers):
    """Class tailored to el-ph models where keeping track of phonon overlaps is
    required. Each walker carries along its Slater determinants a phonon
    displacement vector, self.phonon_disp.

    Parameters
    ----------
    initial_walker :
        Walker that we start the simulation from. Ideally chosen according to
        the trial.
    nup :
        Number of electrons in up-spin space.
    ndown :
        Number of electrons in down-spin space.
    nbasis :
        Number of sites in the 1D Holstein chain.
    nwalkers :
        Number of walkers in the simulation.
    verbose :
        Print level.
    """

    def __init__(
        self,
        initial_walker: numpy.ndarray,
        nup: int,
        ndown: int,
        nbasis: int,
        nwalkers: int,
        verbose: bool = False,
    ):
        super().__init__(initial_walker, nup, ndown, nbasis, nwalkers, verbose)
#        self.phonon_disp *= numpy.sqrt(2 / (trial.m * trial.w0)) # Makes this expected position instead of beta
        self.phonon_disp /= numpy.sqrt(2 / (trial.m * trial.w0)) #to counter build of EPhWalkers
        self.coherent_state_shift = self.phonon_disp #list aliasing, pop_controller uses phonon_disp



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
from ipie.walkers.base_walkers import BaseWalkers

from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

class EPhWalkersFP(EPhWalkers):
#    def __init__(self, initial_walker: numpy.ndarray, nup: int, ndown: int, nbasis: int, nwalkers: int, verbose: bool = False):
#        super().__init__(initial_walker, nup, ndown, nbasis, nwalkers, verbose)

    def orthogonalise(self, free_projection=True):
        """Orthogonalise all walkers.

        Parameters
        ----------
        free_projection : bool
            This flag is not used here.
        """
        detR = self.reortho()
        if free_projection:
            magn, dtheta = xp.abs(self.detR), xp.angle(self.detR)
            self.weight *= magn
            self.phase *= xp.exp(1j * dtheta)
        return detR
    
    def reortho(self):
        """reorthogonalise walkers for free projection, retaining normalization.

        parameters
        ----------
        """
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        else:
            ndown = self.ndown
            detR = []
            for iw in range(self.nwalkers):
                (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
                det_i = xp.prod(xp.diag(Rup))
                if det_i < 0:
                    det_i *= -1
                    self.phia[iw] *= -1

                if ndown > 0:
                    (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                    det_i *= xp.prod(xp.diag(Rdn))

                detR += [det_i]
                self.log_detR[iw] += xp.log(detR[iw])
                self.detR[iw] = detR[iw]
                
                self.el_ovlp[iw, ...] = self.el_ovlp[iw, ...] / detR[iw]
                self.ovlp_perm[iw, ...] = self.ovlp_perm[iw, ...] / detR[iw]
                self.ovlp[iw] = self.ovlp[iw] / detR[iw]

        synchronize()
        return self.detR

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
from typing import Tuple

from ipie.addons.eph.walkers.eph_walkers import EPhCoherentStateWalkers
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import circ_perm
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial

class ToyozawaTrialCoherentState(ToyozawaTrial):  
    def __init__(
        self,
        wavefunction: np.ndarray,
        w0: float,
        num_elec: Tuple[int, int],
        num_basis: int,
        K: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(wavefunction, w0, num_elec, num_basis, K, verbose=verbose)

    def calc_overlap_perms(self, walkers: EPhCoherentStateWalkers) -> np.ndarray:
        r""""""
        for ip, perm in enumerate(self.perms):
            ph_ov = np.exp(-0.5 * (np.abs(self.beta_shift[perm])**2 + np.abs(walkers.coherent_state_shift)**2) - 2*self.beta_shift[perm].conj() * walkers.coherent_state_shift)
            walkers.ph_ovlp[:, ip] = np.prod(ph_ov, axis=1)
        return walkers.ph_ovlp

    def calc_phonon_displacement(self, walkers: EPhCoherentStateWalkers) -> np.ndarray:
        r""""""
        displacement = np.zeros((walkers.nwalkers, self.nperms), dtype=xp.complex128)
        for ip, (Ga, Gb, perm) in enumerate(zip(walkers.Ga_perm.T, walkers.Gb_perm.T, self.perms)):
            displacement[:, ip] = np.einsum('iin,ni->n', Ga, self.beta_shift[perm].conj() + walkers.coherent_state_shift)
#            displacement[:, ip] = np.sum(Ga.diagonal() * np.sum(self.beta_shift[perm].conj() + walkers.coherent_state_shift, axis=1), axis=1)
            if self.ndown > 0:
#                displacement[:, ip] += Gb.diagonal() * np.sum(self.beta_shift[perm].conj() * walkers.coherent_state_shift, axis=1)
                displacement[:, ip] += np.einsum('iin,ni->n', Gb, self.beta_shift[perm].conj() + walkers.coherent_state_shift)
        
        displacement = np.einsum("np,n->n", displacement, 1 / np.sum(walkers.ovlp_perm, axis=1)) 

#        return np.sum(displacement, axis=1)
        return displacement 

    def calc_harm_osc(self, walkers: EPhCoherentStateWalkers) -> np.ndarray:
        r""""""
        harm_osc = np.zeros((walkers.nwalkers, self.nperms), dtype=xp.complex128)
        for ip, (ovlp, perm) in enumerate(zip(walkers.ovlp_perm.T, self.perms)):
            harm_osc[:, ip] = ovlp * np.sum(self.beta_shift[perm].conj() * walkers.coherent_state_shift, axis=1)
        
        harm_osc = np.einsum("np,n->n", harm_osc, 1 / np.sum(walkers.ovlp_perm, axis=1))         
        
        return harm_osc
 

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

from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp
from ipie.addons.eph.trial_wavefunction.toyozawa_coherent_state import ToyozawaTrialCoherentState
from ipie.addons.eph.trial_wavefunction.toyozawa import circ_perm

class CoherentStateTrialCoherentState(ToyozawaTrialCoherentState):
    def __init__(self, wavefunction, w0, num_elec, num_basis, verbose=False): 
        super().__init__(wavefunction, w0, num_elec, num_basis, K=0., verbose=verbose)
        self.perms = [circ_perm(np.arange(self.nbasis))[0]]
        self.nperms = 1
        self.kcoeffs = np.exp(1j * self.K * np.arange(self.nbasis))



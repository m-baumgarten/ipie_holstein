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
from typing import List, Union
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
import jax
import jax.numpy as npj
import plum
from ipie.addons.eph.trial_wavefunction.variational.jax.dd1 import dD1Variational

class D1Variational(dD1Variational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        K: float = 0.,
        cplx: bool = True,
    ):
        super().__init__(shift_init, electron_init, hamiltonian, system, K, cplx)
        assert K == 0.
        self.perms = np.array([self.perms[0]])
        self.nperms = 1
        self.Kcoeffs = np.array([self.Kcoeffs[0]])


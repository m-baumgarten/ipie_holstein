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
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational

class CoherentStateVariational(ToyozawaVariational):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        K: float = 0.0,
        cplx: bool = True,
    ):
        self.K = np.zeros(hamiltonian.dim)
        super().__init__(shift_init, electron_init, hamiltonian, system, K, cplx)
        assert np.allclose(K, np.zeros(hamiltonian.dim))
        
        self.perms = np.array([self.perms[0]])
        self.nperms = 1
        self.Kcoeffs = np.array([self.Kcoeffs[0]])

    def _gradient(self, x, *args) -> np.ndarray:
        """"""
        shift, c0a, c0b = self.unpack_x(x)
        shift = np.squeeze(shift)
        
        G = np.outer(c0a[:,0].conj(), c0a[:,0])
        ovlp = np.sum(G.diagonal())
        one_body = self.ham.T[0] + np.einsum('ijk,k->ij', self.ham.g_tensor, 2 * shift.real)

        shift_grad_real = 2 * shift.real + 2 * np.einsum('ijk,ij->k', self.ham.g_tensor, G) / ovlp
        shift_grad_imag = 2 * shift.imag
        
        psia_grad = 2 * np.einsum('ij,je->i', one_body, c0a) / ovlp
        psia_grad -= (np.einsum('ie,i->', c0a.conj(), psia_grad) / ovlp) * c0a[:,0]

        dx = np.hstack([shift_grad_real, shift_grad_imag, psia_grad.real, psia_grad.imag]).astype(np.float64)        
        return dx

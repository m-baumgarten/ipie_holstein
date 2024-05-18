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

np.random.seed(125)

from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.variational.coherent_state import CoherentStateVariational
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational

# System Parameters
nup = 1
ndown = 0
nelec = (nup, ndown)

# Hamiltonian Parameters
g = 1.0
t = 1.0
w0 = 1.0
nsites = 32
pbc = True

# System and Hamiltonian setup
system = Generic(nelec)
ham = HolsteinModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
ham.build()

nk = nsites//2 + 1
K = np.linspace(0, np.pi, nk, endpoint=True)

# Variational procedure
initial_electron = np.random.random((nsites, nup + ndown))
initial_phonons = np.random.normal(size=(nsites))

var = CoherentStateVariational(initial_phonons, initial_electron, ham, system, cplx=True)
#var.initial_guess(ham)
_, initial_phonons, initial_electron = var.run()

etrial = np.zeros(nk)
for i,k in enumerate(K):

    var = ToyozawaVariational(initial_phonons, initial_electron, ham, system, k, cplx=True)
    etrial[i], initial_phonons, initial_electron = var.run()

    np.save('trial_ph_{:02d}.npy'.format(i), initial_phonons.copy())
    np.save('trial_el_{:02d}.npy'.format(i), initial_electron.copy())
    exit()

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
from ipie.addons.eph.trial_wavefunction.variational.dd1 import dD1Variational
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational
from ipie.addons.eph.trial_wavefunction.variational.globallocal import GlobalLocalVariational
from ipie.addons.eph.trial_wavefunction.variational.utils import D2_to_DTilde, DTilde_to_D1

from ipie.addons.eph.hamiltonians.ssh import OpticalSSHModel

# System Parameters
nup = 1
ndown = 0
nelec = (nup, ndown)

# Hamiltonian Parameters
g = 0.7
t = 1.0
w0 = 0.5
nsites = 16
pbc = True

# Setup initial guess for variational optimization
initial_electron = np.random.random((nsites, nup + ndown))
initial_phonons = np.ones(nsites) * 0.1
#initial_phonons = np.random.normal(size=(nsites))

# System and Hamiltonian setup
system = Generic(nelec)
ham = OpticalSSHModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
ham.build()

nk = nsites//2 + 1
K = np.linspace(0, np.pi, nk, endpoint=True)


var = ToyozawaVariational(initial_phonons, initial_electron, ham, system, 0., cplx=True)
_, initial_phonons, initial_electron = var.run()
initial_phonons = D2_to_DTilde(np.squeeze(initial_phonons))

#var = GlobalLocalVariational(initial_phonons, initial_electron, ham, system, 0., cplx=True)
#_, initial_phonons, initial_electron = var.run()
initial_phonons = DTilde_to_D1(np.squeeze(initial_phonons))

etrial = np.zeros(nk)

#initial_electron = np.load('el_dd1_4_new.npy')
#initial_phonons = np.load('ph_dd1_4_new.npy')
#etrial = np.load('etrial_dd1_new.npy')

for i,k in enumerate(K):
#for i, k in enumerate(K[:5][::-1]):    
#    i = 4 - i
#    i = i+1
    print(f'step {i}', k)

    var = dD1Variational(initial_phonons, initial_electron, ham, system, k, cplx=True)
    etrial[i], initial_phonons, initial_electron = var.run()
    np.save('etrial_dd1.npy', etrial)
    np.save(f'ph_dd1_{i}.npy', initial_phonons)
    np.save(f'el_dd1_{i}.npy', initial_electron)


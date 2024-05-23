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
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectron
from ipie.addons.eph.trial_wavefunction.variational.coherent_state import CoherentStateVariational
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational

# System Parameters
nup = 1
ndown = 0
nelec = (nup, ndown)

# Hamiltonian Parameters
g = 0.2 / np.sqrt(25)
ge = 1.
te = 1.
w0 = te
wtilde = te / 25
je = 1.
nsites = 50
pbc = True

# Setup initial guess for variational optimization
initial_electron = np.random.random((nsites, nup + ndown))
initial_phonons = np.random.normal(size=(nsites))

# System and Hamiltonian setup
system = Generic(nelec)
ham = ExcitonPhononCavityElectron(g=g, ge=ge, te=te, w0=w0, wtilde=wtilde, je=je, nsites=nsites, pbc=pbc)
ham.build()

nk = nsites // 2 + 1
K = np.linspace(0, np.pi, nk, endpoint=True)

# Variational procedure
etrial = np.zeros(nk)
#var = CoherentStateVariational(initial_phonons, initial_electron, ham, system, cplx=True)
#var.initial_guess(ham)
#_, initial_phonons, initial_electron = var.run()

initial_electron = np.load('trials_el/trial_el_25.npy')
initial_phonons = np.load('trials_el/trial_ph_25.npy')


for i, k in enumerate(K[::-1]):
    i = len(K) - 1 - i
    print(k / np.pi)
    var = ToyozawaVariational(initial_phonons, initial_electron, ham, system, k, cplx=True)
    etrial[i], initial_phonons, initial_electron = var.run()
    print(initial_phonons, initial_electron)
    print(etrial[i])

    np.save("trials_el/trial_ph_{:02d}.npy".format(i), initial_phonons.copy())
    np.save("trials_el/trial_el_{:02d}.npy".format(i), initial_electron.copy())
    np.save("trials_el/band.npy", etrial)
    print(k)

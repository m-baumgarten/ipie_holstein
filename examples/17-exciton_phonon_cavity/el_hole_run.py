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
from ipie.addons.eph.hamiltonians.exciton_phonon_cavity import ExcitonPhononCavityElectronHole
from ipie.addons.eph.trial_wavefunction.variational.el_hole.coherent_state_el_hole import CoherentStateVariationalElectronHole
from ipie.addons.eph.trial_wavefunction.variational.el_hole.toyozawa_el_hole import ToyozawaVariationalElectronHole

# System Parameters
nup = 1
ndown = 0
nelec = (nup, ndown)

# Hamiltonian Parameters
#g = 0.001
g = 0.2 / np.sqrt(25)
ge = 1.
te = 1.
w0 = te
wtilde = te / 25
je = 1.
nsites = 50
pbc = True

gh = 1.2
th = 0.9
jh = 1.

u0 = 1.
u1 = 0.25
L = 1.
e = 0.5

# Setup initial guess for variational optimization
initial_phonons = np.random.normal(size=(nsites))
initial_electron = np.random.random((nsites, nup + ndown))
initial_hole = np.random.random((nsites, nup + ndown))

# System and Hamiltonian setup
system = Generic(nelec)
ham = ExcitonPhononCavityElectronHole(g=g, ge=ge, gh=gh, te=te, th=th, w0=w0, wtilde=wtilde, je=je, jh=jh, u0=u0, u1=u1, L=L, e=e, nsites=nsites, pbc=pbc)
ham.build()

nk = nsites // 2 + 1
K = np.linspace(0, np.pi, nk, endpoint=True)

# Variational procedure
etrial = np.zeros(nk)
#initial_phonons = np.load('shift.npy')
#initial_electron = np.load('elec.npy')
#initial_hole = np.load('hole.npy')
var = CoherentStateVariationalElectronHole(initial_phonons, initial_electron, initial_hole, ham, system, cplx=True)
_, initial_phonons, initial_electron, init_hole = var.run()
#np.save('shift.npy', initial_phonons)
#np.save('elec.npy', initial_electron)
#np.save('hole.npy', initial_hole)
#exit()

for i, k in enumerate(K):
    print(k / np.pi)
    var = ToyozawaVariationalElectronHole(initial_phonons, initial_electron, initial_hole, ham, system, k, cplx=True)
    etrial[i], initial_phonons, initial_electron, initial_hole = var.run()
    print(initial_phonons, initial_electron)
    print(etrial[i])

    np.save("trials/trial_ph_{:02d}.npy".format(i), initial_phonons.copy())
    np.save("trials/trial_el_{:02d}.npy".format(i), initial_electron.copy())
    np.save("trials/band.npy", etrial)
    print(k)

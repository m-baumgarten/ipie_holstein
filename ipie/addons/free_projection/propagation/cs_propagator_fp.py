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
import time
import scipy.linalg
from typing import Sequence

from ipie.addons.eph.propagation.eph_propagator import construct_one_body_propagator, EPhPropagatorFree, EPhPropagator 
from ipie.addons.eph.propagation.cs_propagator import CoherentStatePropagator
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel

from ipie.utils.backend import synchronize
from ipie.propagation.operations import propagate_one_body
from ipie.propagation.continuous_base import PropagatorTimer
import matplotlib.pyplot as plt


class CoherentStatePropagatorFP(CoherentStatePropagator):
    def __init__(self, timestep: float, verbose: bool = False, exp_nmax: int = 10, ene_0: float = 0.) -> None:
        super().__init__(timestep, verbose)
        self.exp_nmax = exp_nmax
        self.eshift = ene_0

    def _update_weight(self, walkers, ovlp, ovlp_new) -> None:
        walkers.weight *= numpy.exp(self.dt_ph * self.eshift)

    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        ratio = ovlp_new / ovlp
        phase = numpy.angle(ratio)

        walkers.weight *= numpy.abs(ratio)
        walkers.phase *= numpy.exp(1j * numpy.angle(phase))

    def propagate_electron(
        self, 
        walkers: EPhCSWalkers, 
        hamiltonian: HolsteinModel, 
        trial: EPhTrialWavefunctionBase
    ) -> None:
        
        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time

        N = numpy.random.normal(loc=0.0, scale=1/numpy.sqrt(6), size=(2, walkers.nwalkers, self.nsites))
        new_coherent_shift = N[0] + 1j * N[1]

        EPh = self.construct_EPh(walkers, hamiltonian, new_coherent_shift, trial)
        expEph = scipy.linalg.expm(-self.dt * EPh)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum("nij,nje->nie", expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])

        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum("nij,nje->nie", expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])

        weight_fac = numpy.exp(1j * numpy.sum(new_coherent_shift.real * walkers.coherent_state_shift.imag - new_coherent_shift.imag * walkers.coherent_state_shift.real, axis=1))
        walkers.weight *= np.abs(weight_fac)
        phase = numpy.angle(weight_fac)
        walkers.phase *= np.exp(1j * phase)
        mf = numpy.exp(hamiltonian.g * self.dt * numpy.sum(trial.mf_eph))
        walkers.weight *= numpy.abs(mf)
        walkers.phase *= numpy.exp(1j * np.phase(mf))
        
        walkers.coherent_state_shift += new_coherent_shift
        
        walkers.weight *= numpy.exp(2.5 * numpy.sum(numpy.abs(new_coherent_shift)**2, axis=1)) / numpy.sqrt(6)




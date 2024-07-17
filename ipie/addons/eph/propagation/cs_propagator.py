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

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

from ipie.utils.backend import synchronize
from ipie.propagation.operations import propagate_one_body
from ipie.propagation.continuous_base import PropagatorTimer

from ipie.addons.eph.propagation.eph_propagator import EPhPropagatorFree

class CoherentStatePropagator(EPhPropagatorFree):
    """"""
    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def propagate_phonons(
        self,
        walkers: EPhCoherentStateWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
    ) -> None:
        walkers.coherent_state_shift *= numpy.exp(-self.dt_ph * hamiltonian.w0)
        weight_exponent = -0.5 * numpy.sum(numpy.abs(walkers.coherent_state_shift) ** 2, axis=1)
        weight_exponent *= numpy.exp(self.dt * hamiltonian.w0) - 1
        walkers.weight *= numpy.exp(weight_exponent)

def propagate_electron(
        self,
        walkers: EPhCoherentStateWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
    ) -> None:

        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time

        N = numpy.random.normal(loc=0.0, scale=1.0, size=(2, walkers.nwalkers, self.nsites))
        new_coherent_shift = N[0] + 1j * N[1]

        EPh = self.construct_EPh(walkers, hamiltonian, new_coherent_shift, trial)
        expEph = scipy.linalg.expm(-self.dt * EPh)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum("ni,nie->nie", expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])

        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum("ni,nie->nie", expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])


        gamma = new_coherent_shift + walkers.coherent_state_shift
        weight_fac = numpy.exp(1j * numpy.sum(gamma.real * walkers.coherent_state_shift.imag - gamma.imag * walkers.coherent_state_shift.real, axis=1))
        phase = numpy.angle(weight_fac)
        abs_phase = numpy.abs(phase)
        cos_phase = numpy.cos(phase)
        walkers.weight *= numpy.where(
            abs_phase < 0.5 * numpy.pi,  # <
            numpy.abs(weight_fac) * numpy.where(cos_phase > 0.0, cos_phase, 0.0),  # >
            0.0,
        ).astype(numpy.complex128)

        walkers.weight *= 2 * numpy.exp(hamiltonian.g * self.dt * numpy.sum(trial.mf_eph)) # * np.abs(phase)

        walkers.coherent_state_shift += new_coherent_shift

    def construct_EPh(
        self, walkers: EPhCoherentStateWalkers, hamiltonian: HolsteinModel, new_shift: numpy.ndarray, trial
    ) -> numpy.ndarray:
        cs_displ = new_shift.conj() + 2 * walkers.coherent_state_shift.real - trial.mf_eph
        return numpy.einsum('ijk,k->ij', hamiltonian.g_tensor, cs_displ)

    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        ratio = ovlp_new / ovlp #* walkers.weight
        phase = numpy.angle(ratio)
        abs_phase = numpy.abs(phase)
        cos_phase = numpy.cos(phase)
        walkers.weight *= numpy.where(
            abs_phase < 0.5 * numpy.pi,  # <
            numpy.abs(ratio) * numpy.where(cos_phase > 0.0, cos_phase, 0.0),  # >
            0.0,
        ).astype(numpy.complex128)


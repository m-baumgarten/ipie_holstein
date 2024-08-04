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
from ipie.addons.free_projection.walkers.cs_walkers import EPhCSWalkersFP
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

    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        walkers.weight_log += self.dt_ph * self.eshift
        walkers.weight *= numpy.exp(self.dt_ph * self.eshift)
        walkers.ovlp = ovlp_new
        #print('olo')

    def _update_weight(self, walkers, ovlp, ovlp_new) -> None:
        ratio = ovlp_new / ovlp
        phase = numpy.angle(ratio)

        walkers.weight *= numpy.abs(ratio)
        walkers.phase *= numpy.exp(1j * numpy.angle(phase))
    
    def propagate_phonons(
        self,
        walkers: EPhCSWalkersFP,
        hamiltonian: GenericEPhModel,
        trial: EPhTrialWavefunctionBase,
    ) -> None:
        
        # Normalized CS
#        walkers.coherent_state_shift *= numpy.exp(-self.dt_ph * hamiltonian.w0)
#        weight_exponent = -0.5 * numpy.sum(numpy.abs(walkers.coherent_state_shift) ** 2, axis=1)
#        weight_exponent *= numpy.exp(self.dt * hamiltonian.w0) - 1
#        walkers.weight *= numpy.exp(weight_exponent)
#        walkers.weight_log += weight_exponent

        # Unnormalized CS
        walkers.coherent_state_shift *= numpy.exp(-self.dt_ph * hamiltonian.w0)

    def propagate_electron(
        self, 
        walkers: EPhCSWalkersFP, 
        hamiltonian: GenericEPhModel, 
        trial: EPhTrialWavefunctionBase
    ) -> None:
        new_scale = 0.1

        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time
        
        # Normalized CS
#        N = numpy.random.normal(loc=0.0, scale=new_scale, size=(2, walkers.nwalkers, self.nsites))
        
        # Unnormalized CS
        N = numpy.random.normal(loc=0.0, scale=1/numpy.sqrt(2), size=(2, walkers.nwalkers, self.nsites))
        new_coherent_shift = N[0] #+ 1j * N[1]

        EPh = self.construct_EPh(walkers, hamiltonian, new_coherent_shift, trial)
        expEph = scipy.linalg.expm(-self.dt * EPh)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum("nij,nje->nie", expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])

        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum("nij,nje->nie", expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])

        
        # Unnormalized CS
        walkers.weight *= numpy.exp(numpy.sum(new_coherent_shift.real * walkers.coherent_state_shift.real + new_coherent_shift.imag * walkers.coherent_state_shift.imag, axis=1))
        walkers.weight_log += numpy.sum(new_coherent_shift.real * walkers.coherent_state_shift.real + new_coherent_shift.imag * walkers.coherent_state_shift.imag, axis=1)
#        walkers.weight *= numpy.exp(hamiltonian.g * self.dt * numpy.sum(trial.mf_eph))
#        walkers.weight_log += hamiltonian.g * self.dt * numpy.sum(trial.mf_eph)
        walkers.coherent_state_shift = new_coherent_shift

        # Normalized CS
#        weight_fac = numpy.exp(1j * numpy.sum(new_coherent_shift.real * walkers.coherent_state_shift.imag - new_coherent_shift.imag * walkers.coherent_state_shift.real, axis=1))
#        walkers.weight_log += numpy.log(numpy.abs(weight_fac))
#        walkers.weight *= numpy.abs(weight_fac)

#        phase = numpy.angle(weight_fac)
#        walkers.phase *= numpy.exp(1j * phase)
        
#        mf = numpy.exp(hamiltonian.g * self.dt * numpy.sum(trial.mf_eph))
#        walkers.weight_log += numpy.log(numpy.abs(mf))
#        walkers.weight *= numpy.abs(mf)
#        walkers.phase *= numpy.exp(1j * numpy.angle(mf))
        
#        walkers.coherent_state_shift += new_coherent_shift
        
#        walkers.weight_log += numpy.log(new_scale) -0.5 * (1 - 1/new_scale**2) * numpy.sum(numpy.abs(new_coherent_shift)**2, axis=1)
#        walkers.weight *= new_scale * numpy.exp(-0.5 * (1 - 1/new_scale**2) * numpy.sum(numpy.abs(new_coherent_shift)**2, axis=1))

    def construct_EPh(
        self, walkers: EPhCSWalkersFP, hamiltonian: GenericEPhModel, new_shift: numpy.ndarray, trial
    ) -> numpy.ndarray:
        # Normalized CS
#        cs_displ = new_shift.conj() + 2 * walkers.coherent_state_shift.real - trial.mf_eph
        
        # Unnormalized CS
        cs_displ = new_shift.conj() + walkers.coherent_state_shift #- trial.mf_eph
        return numpy.einsum('ijk,nk->nij', hamiltonian.g_tensor, cs_displ)


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
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel

from ipie.utils.backend import synchronize
from ipie.propagation.operations import propagate_one_body
from ipie.propagation.continuous_base import PropagatorTimer
import matplotlib.pyplot as plt


class EPhPropagatorFP(EPhPropagatorFree):
    def __init__(self, timestep: float, verbose: bool = False, exp_nmax: int = 10, ene_0: float = 0.) -> None:
        super().__init__(timestep, verbose)
        self.exp_nmax = exp_nmax
        self.eshift = ene_0

    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        walkers.weight *= numpy.exp(self.dt_ph * self.eshift)

    def propagate_phonons(
        self, walkers: EPhWalkers, hamiltonian: GenericEPhModel, trial: EPhTrialWavefunctionBase
    ) -> None:
        r"""Propagates phonon displacements by adjusting weigths according to
        bosonic on-site energies and sampling the momentum contribution, again
        by trotterizing the phonon propagator.
        
        .. math::
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2} \approx
            \mathrm{e}^{\Delta \tau N \omega / 4}
            \mathrm{e}^{-\Delta \tau \sum_i m \omega \hat{X}_i^2 / 8}
            \mathrm{e}^{-\Delta \tau \sum_i \hat{P}_i^2 / (4 \omega)}
            \mathrm{e}^{-\Delta \tau \sum_i m \omega \hat{X}_i^2 / 8}
        
        One can obtain the sampling prescription by insertion of resolutions of
        identity, :math:`\int dX |X\rangle \langleX|, and performin the resulting
        Fourier transformation.

        Parameters
        ----------
        walkers :
            Walkers class
        """
        start_time = time.time()
        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot_real, pot_imag = numpy.real(pot), numpy.imag(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot_real)
        walkers.phase *= numpy.exp(-1j * self.dt_ph * pot_imag)

        N = numpy.random.normal(loc=0.0, scale=self.scale, size=(walkers.nwalkers, self.nsites))
        walkers.phonon_disp = walkers.phonon_disp + N

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot_real, pot_imag = numpy.real(pot), numpy.imag(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot_real)
        walkers.phase *= numpy.exp(-1j * self.dt_ph * pot_imag)

        # Does not matter for estimators but helps with population control
        walkers.weight *= numpy.exp(self.dt_ph * self.nsites * self.w0 / 2)
        synchronize()
        self.timer.tgemm += time.time() - start_time

class EPhPropagatorFPImportance(EPhPropagatorFP):
    def __init__(self, timestep: float, verbose: bool = False, exp_nmax: int = 10, ene_0: float = 0.) -> None:
        super().__init__(timestep, verbose, exp_nmax, ene_0)
   
    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        ratio = ovlp_new / ovlp
        phase = numpy.angle(ratio)

        walkers.weight *= numpy.abs(ratio)
        walkers.phase *= numpy.exp(1j * numpy.angle(phase))

    def propagate_phonons(
        self, walkers: EPhWalkers, hamiltonian: GenericEPhModel, trial: EPhTrialWavefunctionBase
    ) -> None:
        start_time = time.time()

        ovlp_old = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp_old

        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian(walkers) / hamiltonian.m
        pot -= 0.5 * hamiltonian.nsites * hamiltonian.w0
        pot_real, pot_imag = numpy.real(pot), numpy.imag(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot_real / 2)
        walkers.phase *= numpy.exp(-1j * self.dt_ph * pot_imag / 2)

        N = numpy.random.normal(loc=0.0, scale=self.scale, size=(walkers.nwalkers, self.nsites))
        drift = numpy.real(trial.calc_phonon_gradient(walkers)).astype(numpy.complex128)
        walkers.phonon_disp = walkers.phonon_disp + N + self.dt_ph * drift / hamiltonian.m

        ovlp_new = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp_new

        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian(walkers) / hamiltonian.m
        pot -= 0.5 * hamiltonian.nsites * hamiltonian.w0
        pot_real, pot_imag = numpy.real(pot), numpy.imag(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot_real / 2)
        walkers.phase *= numpy.exp(-1j * self.dt_ph * pot_imag / 2)

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_walkers(
        self,
        walkers: EPhWalkers,
        hamiltonian: GenericEPhModel,
        trial: EPhTrialWavefunctionBase,
        eshift: float = None,
    ) -> None:
        r"""Propagates walkers by trotterized propagator.

        Parameters
        ----------
        walkers :
            EPhWalkers object
        hamiltonian :
            HolsteinModel object
        trial :
            EPhTrialWavefunctionBase object
        eshift :
            Only purpose is compatibility with AFQMC object, irrelevant for
            propagation
        """
        synchronize()
        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time

        # Update Walkers
        # a) DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        # b) One-body propagation for electrons
        ovlp = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp
        self.propagate_electron(walkers, hamiltonian, trial)
        ovlp_new = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp_new
        self.update_weight(walkers, ovlp, ovlp_new)
        
        # c) DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        # Update weights (and later do phaseless for multi-electron)
        start_time = time.time()
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        synchronize()
        self.timer.tupdate += time.time() - start_time


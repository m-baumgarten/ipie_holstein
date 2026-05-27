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
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers

from ipie.utils.backend import synchronize
from ipie.propagation.operations import propagate_one_body
from ipie.propagation.continuous_base import PropagatorTimer

from ipie.addons.eph.propagation.eph_propagator import EPhPropagatorFree

class CoherentStatePropagator(EPhPropagatorFree):
    """"""
    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def build(
        self,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase = None,
        walkers: EPhWalkers = None,
        mpi_handler=None,
    ) -> None:
        super().build(hamiltonian, trial=trial, walkers=walkers, mpi_handler=mpi_handler)
        if trial is not None and getattr(trial, "coherent_state_convention", None) != "normalized":
            raise TypeError("CoherentStatePropagator requires a normalized coherent-state trial.")
        self.nsites = hamiltonian.N
        self._local_holstein_coupling = self._is_local_holstein_coupling(hamiltonian)

    def propagate_phonons(
        self,
        walkers: EPhCSWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
    ) -> None:
        
        # Normalized CS
        walkers.coherent_state_shift *= numpy.exp(-self.dt_ph * hamiltonian.w0)
        weight_exponent = -0.5 * numpy.sum(numpy.abs(walkers.coherent_state_shift) ** 2, axis=1)
        weight_exponent *= numpy.exp(self.dt * hamiltonian.w0) - 1
        walkers.weight *= numpy.exp(weight_exponent)

    def propagate_electron(
        self,
        walkers: EPhCSWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
    ) -> None:

        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time

        gaussian = numpy.random.normal(
            loc=0.0, scale=1.0, size=(2, walkers.nwalkers, hamiltonian.N)
        )
        new_coherent_shift = gaussian[0] + 1j * gaussian[1]

        EPh = self.construct_EPh(walkers, hamiltonian, new_coherent_shift, trial)
        expEph = scipy.linalg.expm(-self.dt * EPh)
        exp_bch = self.construct_bch_propagator(walkers, hamiltonian)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum("nij,nje->nie", expEph, walkers.phia)
        walkers.phia = self.apply_bch_propagator(walkers.phia, exp_bch)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])

        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum("nij,nje->nie", expEph, walkers.phib)
            walkers.phib = self.apply_bch_propagator(walkers.phib, exp_bch)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])

        weight_fac = numpy.exp(1j * numpy.sum(new_coherent_shift.real * walkers.coherent_state_shift.imag - new_coherent_shift.imag * walkers.coherent_state_shift.real, axis=1))
        walkers._cs_weight_fac = (2 ** hamiltonian.N) * weight_fac
        walkers.coherent_state_shift += new_coherent_shift

    def construct_EPh(
        self, walkers: EPhCSWalkers, hamiltonian: HolsteinModel, new_shift: numpy.ndarray, trial
    ) -> numpy.ndarray:
        cs_displ = new_shift.conj() + 2 * walkers.coherent_state_shift.real
        return numpy.einsum('ijk,nk->nij', hamiltonian.g_tensor, cs_displ)

    def construct_bch_propagator(
        self, walkers: EPhCSWalkers, hamiltonian: HolsteinModel
    ) -> numpy.ndarray:
        r"""Build the BCH factor from normal-ordering the el-ph exponential.

        For local Holstein coupling the operators :math:`g n_i` commute. For a
        one-spin walker, :math:`n_i^2 = n_i`, so this is a deterministic
        one-body propagator. With both spin sectors present, the
        :math:`n_{i\uparrow} n_{i\downarrow}` term is represented by the
        equivalent continuous HS average.
        """
        if not self._local_holstein_coupling:
            raise NotImplementedError(
                "The coherent-state BCH correction is implemented for local "
                "Holstein density coupling only."
            )

        if walkers.nup > 0 and walkers.ndown > 0:
            fields = numpy.random.normal(loc=0.0, scale=1.0, size=(walkers.nwalkers, hamiltonian.N))
            bch = self.dt * numpy.einsum("ijk,nk->nij", hamiltonian.g_tensor, fields)
            return scipy.linalg.expm(bch)

        bch = numpy.zeros((hamiltonian.N, hamiltonian.N), dtype=numpy.complex128)
        for imode in range(hamiltonian.N):
            coupling = hamiltonian.g_tensor[:, :, imode]
            bch += coupling.dot(coupling)
        return scipy.linalg.expm(0.5 * self.dt * self.dt * bch)

    @staticmethod
    def apply_bch_propagator(phi: numpy.ndarray, exp_bch: numpy.ndarray) -> numpy.ndarray:
        if exp_bch.ndim == 2:
            return propagate_one_body(phi, exp_bch)
        return numpy.einsum("nij,nje->nie", exp_bch, phi)

    @staticmethod
    def _is_local_holstein_coupling(hamiltonian: HolsteinModel) -> bool:
        if not hasattr(hamiltonian, "g") or not hasattr(hamiltonian, "g_tensor"):
            return False
        expected = numpy.zeros_like(hamiltonian.g_tensor)
        for site in range(hamiltonian.N):
            expected[site, site, site] = hamiltonian.g
        return numpy.allclose(hamiltonian.g_tensor, expected)

    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        cs_weight_fac = numpy.asarray(getattr(walkers, "_cs_weight_fac", 1.0), dtype=numpy.complex128)
        if cs_weight_fac.ndim == 0:
            cs_weight_fac = numpy.full_like(ovlp_new, cs_weight_fac)
        ratio = numpy.zeros_like(ovlp_new, dtype=numpy.complex128)
        finite = numpy.isfinite(ovlp) & numpy.isfinite(ovlp_new) & numpy.isfinite(cs_weight_fac)
        nonzero = numpy.abs(ovlp) > 0.0
        valid = finite & nonzero
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio[valid] = (ovlp_new[valid] / ovlp[valid]) * cs_weight_fac[valid]
        ratio = numpy.where(numpy.isfinite(ratio), ratio, 0.0)
        phase = numpy.angle(ratio)
        abs_phase = numpy.abs(phase)
        cos_phase = numpy.cos(phase)
        walkers.weight *= numpy.where(
            abs_phase < 0.5 * numpy.pi,  # <
            numpy.abs(ratio) * numpy.where(cos_phase > 0.0, cos_phase, 0.0),  # >
            0.0,
        ).astype(numpy.complex128)
        walkers._cs_weight_fac = numpy.ones(walkers.nwalkers, dtype=numpy.complex128)

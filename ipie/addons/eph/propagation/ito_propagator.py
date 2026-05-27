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

import time
import numpy

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers

from ipie.utils.backend import synchronize

from ipie.addons.eph.propagation.eph_propagator import EPhPropagatorFree


class EulerItoPropagator(EPhPropagatorFree):
    r"""Bare Euler--Maruyama coherent-state propagator.

    This implements the proper-complex-noise version of the D2 Ito update

    .. math::
        df_\mu = -\omega_\mu f_\mu d\tau + dZ_\mu,

    .. math::
        d\phi = -\left(h+\sum_\mu f_\mu G_\mu\right)\phi d\tau
                -\sum_\mu G_\mu^\dagger\phi\,dZ_\mu^*,

    with :math:`E[dZ_\mu dZ_\nu]=0` and
    :math:`E[dZ_\mu^* dZ_\nu]=\delta_{\mu\nu}d\tau`.
    """

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

        if getattr(trial, "coherent_state_convention", None) != "unnormalized":
            raise TypeError("EulerItoPropagator requires an unnormalized coherent-state trial.")

        self.nmodes = hamiltonian.N
        self.nbasis = hamiltonian.N
        self.g_tensor_dagger = numpy.swapaxes(hamiltonian.g_tensor.conj(), 0, 1)
        self._local_holstein_coupling = self._is_local_holstein_coupling(hamiltonian)

    def sample_complex_noise(self, walkers: EPhCSWalkers, hamiltonian: HolsteinModel) -> numpy.ndarray:
        r"""Draw proper complex Gaussian increments with variance ``dt``."""
        gaussian = numpy.random.normal(
            loc=0.0, scale=1.0, size=(2, walkers.nwalkers, hamiltonian.N)
        )
        return numpy.sqrt(0.5 * self.dt) * (gaussian[0] + 1j * gaussian[1])

    def propagate(
        self,
        walkers: EPhCSWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
    ) -> None:
        r"""Apply one bare Euler--Maruyama step to all walkers."""
        start_time = time.time()

        shift_old = walkers.coherent_state_shift.copy()
        dZ = self.sample_complex_noise(walkers, hamiltonian)

        h_eff = self.construct_annihilation_matrix(shift_old, hamiltonian)
        creation = self.construct_creation_matrix(dZ, hamiltonian)

        walkers.phia = self.apply_electronic_euler_step(walkers.phia, h_eff[0], creation)
        if walkers.ndown > 0:
            walkers.phib = self.apply_electronic_euler_step(walkers.phib, h_eff[1], creation)

        walkers.coherent_state_shift += -self.dt * hamiltonian.w0 * shift_old + dZ

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def construct_annihilation_matrix(
        self, shifts: numpy.ndarray, hamiltonian: HolsteinModel
    ) -> numpy.ndarray:
        r"""Build ``h + sum_mu f_mu G_mu`` for each walker and spin sector."""
        eph = numpy.einsum("ijk,nk->nij", hamiltonian.g_tensor, shifts)
        return numpy.array(
            [
                hamiltonian.T[0][None, :, :] + eph,
                hamiltonian.T[1][None, :, :] + eph,
            ]
        )

    def construct_creation_matrix(
        self, dZ: numpy.ndarray, hamiltonian: HolsteinModel
    ) -> numpy.ndarray:
        r"""Build ``sum_mu dZ_mu^* G_mu^\dagger`` for each walker."""
        return numpy.einsum("ijk,nk->nij", self.g_tensor_dagger, dZ.conj())

    def apply_electronic_euler_step(
        self, phi: numpy.ndarray, h_eff: numpy.ndarray, creation: numpy.ndarray
    ) -> numpy.ndarray:
        drift = numpy.einsum("nij,nje->nie", h_eff, phi)
        tangent = numpy.einsum("nij,nje->nie", creation, phi)
        return phi - self.dt * drift - tangent

    def propagate_walkers(
        self,
        walkers: EPhCSWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
        eshift: float = None,
    ) -> None:
        self.propagate(walkers, hamiltonian, trial)

        start_time = time.time()
        walkers.ovlp = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

    def update_weight(self, walkers, ovlp=None, ovlp_new=None) -> None:
        """Bare projection: trial-overlap ratios do not update walker weights."""
        return None

    @staticmethod
    def _is_local_holstein_coupling(hamiltonian: HolsteinModel) -> bool:
        if not hasattr(hamiltonian, "g") or not hasattr(hamiltonian, "g_tensor"):
            return False
        expected = numpy.zeros_like(hamiltonian.g_tensor)
        for site in range(hamiltonian.N):
            expected[site, site, site] = hamiltonian.g
        return numpy.allclose(hamiltonian.g_tensor, expected)

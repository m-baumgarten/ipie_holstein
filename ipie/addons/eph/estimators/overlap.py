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

from ipie.estimators.estimator_base import EstimatorBase
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.systems.generic import Generic
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers


class OverlapEstimator(EstimatorBase):
    def __init__(
        self,
        system=None,
        ham=None,
        trial=None,
        filename=None,
    ):
        assert system is not None
        assert ham is not None
        assert trial is not None
        super().__init__()
        self._eshift = 0.0
        self.scalar_estimator = True
        self._data = {"OvTotal": 0.0j, "OvPh": 0.0j, "OvEl": 0.0j, "Sign": 0.0j}
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(
        self,
        system: Generic,
        walkers: EPhWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
        istep=1,
    ):
        # TODO I dont think its necessary to recompute overlap here,
        # but just making sure for now!
        # NOTE OvPh * OvEl != OvTotal
        trial.calc_overlap(walkers)
        # Need to be able to dispatch here

        if hasattr(trial, "nperms"):
            ph_ovlp = xp.sum(walkers.ph_ovlp, axis=1)
            el_ovlp = xp.sum(walkers.el_ovlp, axis=1)
        else:
            ph_ovlp = walkers.ph_ovlp.copy()
            el_ovlp = walkers.el_ovlp.copy()

        self._data["OvTotal"] = xp.sum(walkers.weight * walkers.ovlp.real) / walkers.nwalkers
        self._data["OvPh"] = xp.sum(walkers.weight * ph_ovlp.real) / walkers.nwalkers
        self._data["OvEl"] = xp.sum(walkers.weight * el_ovlp.real) / walkers.nwalkers
        self._data["Sign"] = xp.mean(xp.sign(walkers.ovlp.real))
        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

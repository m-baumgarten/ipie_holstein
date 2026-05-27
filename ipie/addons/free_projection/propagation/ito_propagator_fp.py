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

from ipie.addons.eph.propagation.ito_propagator import EulerItoPropagator


class EulerItoPropagatorFP(EulerItoPropagator):
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

    def update_weight(self, walkers, ovlp=None, ovlp_new=None) -> None:
        """Bare free projection: leave weight, phase, and weight_log untouched."""
        return None

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
"""Smoke tests for the four-site Holstein model through the ab-initio path."""

import numpy as np

try:
    import pytest
except ImportError:
    class _UnitMark:
        @staticmethod
        def unit(func):
            return func

    class _PytestShim:
        mark = _UnitMark()

    pytest = _PytestShim()

from fixture import build_holstein_inputs, build_workflow
from ipie.addons.eph.estimators.local_energy_abinitio import local_energy_abinitio


@pytest.mark.unit
def test_holstein_abinitio_tensors():
    inputs = build_holstein_inputs(nsites=4, t=1.0, g=1.0, w0=1.0, nwalkers=3)

    expected_eps = np.array([-2.0, 0.0, 2.0, 0.0])[:, None]
    np.testing.assert_allclose(inputs.eps_kj, expected_eps, atol=1.0e-14)
    np.testing.assert_allclose(inputs.omega_qnu, np.ones((4, 1)))
    np.testing.assert_allclose(inputs.g_qnu_kmn, 0.5 * np.ones((4, 1, 4, 1, 1)))

    np.testing.assert_allclose(inputs.psi_kj, 0.5 * np.ones((4, 1)))
    np.testing.assert_allclose(inputs.beta_qnu, -0.5 * np.ones((4, 1)))
    np.testing.assert_allclose(inputs.initial_X_qnu[:, 3, :], inputs.initial_X_qnu[:, 1, :].conj())
    np.testing.assert_allclose(inputs.initial_X_qnu[:, 0, :].imag, 0.0)
    np.testing.assert_allclose(inputs.initial_X_qnu[:, 2, :].imag, 0.0)


@pytest.mark.unit
def test_holstein_abinitio_workflow_smoke():
    inputs = build_holstein_inputs(nwalkers=8, seed=12)
    system, ham, trial, walkers, propagator = build_workflow(inputs, timestep=0.005)

    initial_energy = local_energy_abinitio(system, ham, walkers, trial)
    assert initial_energy.shape == (walkers.nwalkers, 4)
    assert np.all(np.isfinite(initial_energy))

    rng_state = np.random.get_state()
    np.random.seed(17)
    try:
        for _ in range(3):
            propagator.propagate_walkers(walkers, ham, trial, eshift=0.0)
    finally:
        np.random.set_state(rng_state)

    final_energy = local_energy_abinitio(system, ham, walkers, trial)
    assert np.all(np.isfinite(final_energy))
    assert np.all(np.isfinite(walkers.weight))
    assert np.all(np.isfinite(walkers.phi_kj))
    assert np.all(np.isfinite(walkers.X_qnu))

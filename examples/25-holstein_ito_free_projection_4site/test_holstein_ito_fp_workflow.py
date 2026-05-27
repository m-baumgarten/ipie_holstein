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
"""Smoke tests for the Holstein FP Ito coherent-state walker path."""

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

from fixture import build_holstein_ito_fp_inputs, build_workflow, load_trial_wavefunction
from ipie.addons.free_projection.estimators.energy_eph import EnergyEstimatorFP
from ipie.addons.free_projection.propagation.ito_propagator_fp import EulerItoPropagatorFP
from ipie.addons.free_projection.walkers.cs_walkers import EPhCSWalkersFP


@pytest.mark.unit
def test_holstein_ito_fp_inputs():
    inputs = build_holstein_ito_fp_inputs(nwalkers=5)

    assert inputs.wavefunction.shape == (4, 2)
    assert inputs.nelec == (1, 0)
    assert inputs.nwalkers == 5
    np.testing.assert_allclose(inputs.electron_orbital.conj().T @ inputs.electron_orbital, 1.0)
    assert np.all(inputs.beta_shift.real <= 0.0)


@pytest.mark.unit
def test_holstein_ito_fp_workflow_smoke():
    inputs = build_holstein_ito_fp_inputs(nwalkers=8)
    system, ham, trial, walkers, propagator = build_workflow(inputs, timestep=1.0e-4)

    assert isinstance(walkers, EPhCSWalkersFP)
    assert isinstance(propagator, EulerItoPropagatorFP)
    assert trial.coherent_state_convention == "unnormalized"
    assert walkers.coherent_state_shift.shape == (inputs.nwalkers, inputs.nsites)

    weight_before = walkers.weight.copy()
    phase_before = walkers.phase.copy()
    weight_log_before = walkers.weight_log.copy()

    rng_state = np.random.get_state()
    np.random.seed(17)
    try:
        for _ in range(3):
            propagator.propagate_walkers(walkers, ham, trial, eshift=0.0)
    finally:
        np.random.set_state(rng_state)

    assert np.allclose(walkers.weight, weight_before)
    assert np.allclose(walkers.phase, phase_before)
    assert np.allclose(walkers.weight_log, weight_log_before)
    assert np.all(np.isfinite(walkers.ovlp))
    assert np.all(np.isfinite(walkers.phia))
    assert np.all(np.isfinite(walkers.coherent_state_shift))

    estimator = EnergyEstimatorFP(system=system, ham=ham, trial=trial)
    estimator.compute_estimator(system, walkers, ham, trial)
    assert np.isfinite(estimator["EDenom"])


@pytest.mark.unit
def test_load_trial_wavefunction_roundtrip(tmp_path):
    inputs = build_holstein_ito_fp_inputs(nwalkers=4)
    trial_file = tmp_path / "trial.npz"
    np.savez(trial_file, wavefunction=inputs.wavefunction)

    loaded = load_trial_wavefunction(trial_file, inputs.nsites)
    np.testing.assert_allclose(loaded, inputs.wavefunction)

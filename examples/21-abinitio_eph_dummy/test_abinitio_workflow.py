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
"""Ab-initio electron-phonon dD2 workflow using fixture data.

The fixture stores bands, phonon frequencies, a low-rank representation of the
short-range e-ph vertex, the long-range e-ph contribution, and converged dD2
parameters.  The production code should eventually use the low-rank factors
directly; this example reconstructs the full ``g_qnu_kmn`` tensor explicitly so
the current dense Hamiltonian, estimator, and propagator can be exercised.
"""

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

from fixture import build_fixture_inputs, build_workflow
from ipie.addons.eph.estimators.local_energy_abinitio import local_energy_abinitio


def run_dummy_workflow(num_steps: int = 3, seed: int = 11):
    """Run a few propagation steps and return the final objects and energies."""
    inputs = build_fixture_inputs()
    system, ham, trial, walkers, propagator = build_workflow(inputs)

    initial_energy = local_energy_abinitio(system, ham, walkers, trial)
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for _ in range(num_steps):
            propagator.propagate_walkers(walkers, ham, trial, eshift=0.0)
    finally:
        np.random.set_state(rng_state)

    final_energy = local_energy_abinitio(system, ham, walkers, trial)
    return {
        "system": system,
        "hamiltonian": ham,
        "trial": trial,
        "walkers": walkers,
        "propagator": propagator,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
    }


@pytest.mark.unit
def test_abinitio_workflow_smoke():
    result = run_dummy_workflow(num_steps=2)
    walkers = result["walkers"]
    ham = result["hamiltonian"]

    assert result["initial_energy"].shape == (walkers.nwalkers, 4)
    assert result["final_energy"].shape == (walkers.nwalkers, 4)
    assert walkers.S.shape == (walkers.nwalkers, ham.nk)
    assert walkers.M.shape == (walkers.nwalkers, ham.nq)
    assert walkers.A_el.shape == (walkers.nwalkers, result["trial"].nR)
    assert walkers.A_ph.shape == walkers.A_el.shape

    assert np.all(np.isfinite(result["initial_energy"]))
    assert np.all(np.isfinite(result["final_energy"]))
    assert np.all(np.isfinite(walkers.phi_kj))
    assert np.all(np.isfinite(walkers.X_qnu))
    assert np.all(np.isfinite(walkers.weight))
    assert np.all(walkers.weight.real >= 0.0)


if __name__ == "__main__":
    output = run_dummy_workflow(num_steps=3)
    print("Initial mean local energy:", np.mean(output["initial_energy"][:, 0]))
    print("Final mean local energy:  ", np.mean(output["final_energy"][:, 0]))
    print("Final mean walker weight:", np.mean(output["walkers"].weight))

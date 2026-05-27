import numpy as np
import pytest

import ipie.addons.free_projection.estimators.energy_eph as energy_eph_module
from ipie.addons.free_projection.estimators.energy_eph import (
    EnergyEstimatorFP,
    EnergyEstimatorFPImportance,
)
from ipie.addons.free_projection.estimators.handler import EstimatorHandlerFP


@pytest.mark.unit
def test_energy_eph_fp_estimator_uses_direct_mixed_overlap(monkeypatch):
    class Walkers:
        pass

    class Trial:
        def __init__(self, ovlp):
            self.ovlp = ovlp
            self.greens_calls = 0

        def calc_overlap(self, walkers):
            return self.ovlp.copy()

        def calc_greens_function(self, walkers):
            self.greens_calls += 1
            return None

    walkers = Walkers()
    walkers.weight = np.array([1.0, 2.0, -0.5j], dtype=np.complex128)
    walkers.phase = np.array([1.0, -1.0j, 0.5 + 0.25j], dtype=np.complex128)
    walkers.weight_log = np.array([100.0, -200.0, 50.0], dtype=np.complex128)
    ovlp = np.array([1.0 + 0.5j, -0.25j, 2.0 - 1.0j], dtype=np.complex128)
    energy = np.array(
        [
            [1.0 + 1.0j, 0.5 - 0.25j, -2.0j],
            [-1.0 + 0.25j, 2.0 + 0.5j, 3.0 - 1.0j],
            [0.75 - 0.5j, -1.5 + 0.2j, 0.25 + 0.75j],
        ],
        dtype=np.complex128,
    )

    def fake_local_energy(system, hamiltonian, walkers_arg, trial_arg):
        assert walkers_arg is walkers
        return energy.copy()

    monkeypatch.setattr(energy_eph_module, "local_energy", fake_local_energy)

    trial = Trial(ovlp)
    estimator = EnergyEstimatorFP(system=object(), ham=object(), trial=trial)
    estimator.compute_estimator(system=object(), walkers=walkers, hamiltonian=object(), trial=trial)

    mixed_weight = walkers.weight * walkers.phase * ovlp
    assert trial.greens_calls == 1
    assert np.allclose(walkers.ovlp, ovlp)
    assert np.allclose(estimator["ENumer"], np.sum(mixed_weight * energy[:, 0]))
    assert np.allclose(estimator["EDenom"], np.sum(mixed_weight))
    assert np.allclose(estimator["E1Body"], np.sum(mixed_weight * energy[:, 1]))
    assert np.allclose(estimator["E2Body"], np.sum(mixed_weight * energy[:, 2]))
    assert not np.allclose(estimator["EDenom"], np.sum(np.exp(walkers.weight_log)))


@pytest.mark.unit
def test_energy_eph_fp_importance_estimator_remains_separate():
    class Comm:
        rank = 0

    handler = EstimatorHandlerFP(
        Comm(),
        system=object(),
        hamiltonian=object(),
        trial=object(),
        importance_sampling=False,
    )

    assert isinstance(handler["energy"], EnergyEstimatorFP)
    assert not isinstance(handler["energy"], EnergyEstimatorFPImportance)

import numpy as np
import pytest

from ipie.addons.free_projection.propagation.ito_propagator_fp import EulerItoPropagatorFP


@pytest.mark.unit
def test_ito_fp_update_weight_is_noop_without_importance_sampling():
    class Walkers:
        pass

    walkers = Walkers()
    walkers.weight = np.array([1.0, 2.0j, -3.0], dtype=np.complex128)
    walkers.phase = np.array([1.0, -1.0j, 0.5 + 0.5j], dtype=np.complex128)
    walkers.weight_log = np.array([0.0, 0.5, -1.0], dtype=np.complex128)
    weight_before = walkers.weight.copy()
    phase_before = walkers.phase.copy()
    weight_log_before = walkers.weight_log.copy()

    prop = EulerItoPropagatorFP(0.01)
    ovlp = np.array([1.0, 0.0, 1.0], dtype=np.complex128)
    ovlp_new = np.array([2.0j, 1.0, -1.0], dtype=np.complex128)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        prop.update_weight(walkers, ovlp, ovlp_new)

    assert np.allclose(walkers.weight, weight_before)
    assert np.allclose(walkers.phase, phase_before)
    assert np.allclose(walkers.weight_log, weight_log_before)

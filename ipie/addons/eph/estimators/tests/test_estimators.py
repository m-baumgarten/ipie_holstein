import tempfile

import numpy as np
import pytest

import ipie.addons.eph.estimators.energy as energy_module
from ipie.addons.eph.estimators.energy import EnergyEstimator, EnergyEstimatorNoImportance
from ipie.addons.eph.estimators.local_energy_generic import local_energy_generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.toyozawa_cs_unnormalized import (
    ToyozawaTrialUnnormalizedCoherentState,
)
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers
from ipie.systems import Generic
from ipie.addons.eph.utils.testing import gen_random_test_instances


@pytest.mark.unit
def test_energy_estimator():
    pbc = True
    nbasis = 4
    nelec = (2, 2)
    trial_type = "toyozawa"
    nwalkers = 500
    sys, ham, walkers, trial = gen_random_test_instances(nelec, nbasis, nwalkers, trial_type)
    estim = EnergyEstimator(sys, ham, trial)
    estim.compute_estimator(sys, walkers, ham, trial)
    assert len(estim.names) == 6
    assert estim["ENumer"].real == pytest.approx(-3136.7469620055163)
    assert estim["ETotal"] == pytest.approx(0.0)
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    assert tmp[estim.get_index("ETotal")].real == pytest.approx(-6.273493924011032)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (6,)


@pytest.mark.unit
def test_energy_estimator_no_importance_uses_complex_weighted_overlap(monkeypatch):
    class Walkers:
        pass

    class Trial:
        def __init__(self, ovlp):
            self.ovlp = ovlp

        def calc_overlap(self, walkers):
            return self.ovlp.copy()

    walkers = Walkers()
    walkers.weight = np.array([1.0, 2.0, -0.5j], dtype=np.complex128)
    ovlp = np.array([1.0 + 1.0j, -0.25 + 0.5j, 2.0 - 0.75j], dtype=np.complex128)
    energy = np.array(
        [
            [1.0 + 2.0j, 0.5 + 0.25j, 3.0 - 1.0j, -0.5j],
            [-2.0 + 0.5j, 1.5 - 0.25j, -1.0 + 2.0j, 0.75 + 0.5j],
            [0.25 - 1.0j, -0.75j, 2.5 + 0.5j, -3.0 + 1.0j],
        ],
        dtype=np.complex128,
    )

    def fake_local_energy(system, hamiltonian, walkers_arg, trial_arg):
        assert walkers_arg is walkers
        return energy.copy()

    monkeypatch.setattr(energy_module, "local_energy", fake_local_energy)

    trial = Trial(ovlp)
    estimator = EnergyEstimatorNoImportance(system=object(), ham=object(), trial=trial)
    estimator.compute_estimator(system=object(), walkers=walkers, hamiltonian=object(), trial=trial)

    mixed_weight = walkers.weight * ovlp
    assert np.allclose(walkers.ovlp, ovlp)
    assert np.allclose(estimator["ENumer"], np.sum(mixed_weight * energy[:, 0]))
    assert np.allclose(estimator["EDenom"], np.sum(mixed_weight))
    assert np.allclose(estimator["EEl"], np.sum(mixed_weight * energy[:, 1]))
    assert np.allclose(estimator["EElPh"], np.sum(mixed_weight * energy[:, 2]))
    assert np.allclose(estimator["EPh"], np.sum(mixed_weight * energy[:, 3]))
    assert not np.allclose(estimator["EDenom"], np.sum(walkers.weight * np.abs(ovlp)))


@pytest.mark.unit
def test_coherent_state_local_energy_preserves_complex_total():
    nsites = 4
    system = Generic((1, 0))
    ham = HolsteinModel(g=0.8, t=0.7, w0=1.1, nsites=nsites, pbc=True)
    ham.build()

    beta = np.array([-0.4 + 0.1j, 0.2 - 0.05j, -0.1 + 0.2j, 0.15 + 0.03j])
    orbital = np.array([0.8 + 0.1j, -0.2 + 0.4j, 0.3 - 0.15j, 0.1 + 0.2j])
    orbital = orbital / np.linalg.norm(orbital)
    wavefunction = np.column_stack([beta, orbital])

    trial = ToyozawaTrialUnnormalizedCoherentState(
        wavefunction=wavefunction,
        w0=ham.w0,
        num_elec=(1, 0),
        num_basis=nsites,
        K=0.0,
    )
    walkers = EPhCSWalkers(
        initial_walker=wavefunction,
        nup=1,
        ndown=0,
        nbasis=nsites,
        nwalkers=2,
    )
    walkers.build(trial)
    walkers.coherent_state_shift[0] += np.array([0.03j, -0.02j, 0.04j, -0.01j])
    walkers.phia[0, :, 0] += np.array([0.02j, -0.03j, 0.01j, 0.04j])

    energy = local_energy_generic(system, ham, walkers, trial)
    component_sum = np.sum(energy[:, 1:], axis=1)

    np.testing.assert_allclose(energy[:, 0], component_sum)
    assert np.max(np.abs(component_sum.imag)) > 1e-8


if __name__ == "__main__":
    test_energy_estimator()

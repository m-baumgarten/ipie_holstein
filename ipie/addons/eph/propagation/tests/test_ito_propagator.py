import numpy as np
import pytest

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.propagation.ito_propagator import EulerItoPropagator
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers


class _UnnormalizedTrial:
    coherent_state_convention = "unnormalized"


class _NormalizedTrial:
    coherent_state_convention = "normalized"


class _OverlapTrial(_UnnormalizedTrial):
    def __init__(self, ovlp):
        self.ovlp = np.asarray(ovlp, dtype=np.complex128)
        self.calls = 0

    def calc_overlap(self, walkers):
        self.calls += 1
        return self.ovlp.copy()


@pytest.mark.unit
def test_ito_propagate_fixed_complex_noise(monkeypatch):
    nsites = 3
    nwalkers = 2
    dt = 0.04
    ham = HolsteinModel(g=0.7, t=0.2, w0=1.3, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5j, 0.25], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia])
    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers)

    prop = EulerItoPropagator(dt)
    prop.build(ham, trial=_UnnormalizedTrial(), walkers=walkers)

    gaussian = np.array(
        [
            [[0.1, -0.3, 0.2], [-0.4, 0.5, -0.6]],
            [[0.7, -0.2, 0.3], [-0.1, -0.8, 0.4]],
        ]
    )

    def fake_normal(loc=0.0, scale=1.0, size=None):
        assert loc == 0.0
        assert scale == 1.0
        assert size == gaussian.shape
        return gaussian.copy()

    monkeypatch.setattr(np.random, "normal", fake_normal)

    alpha_before = walkers.coherent_state_shift.copy()
    phia_before = walkers.phia.copy()
    prop.propagate(walkers, ham, trial=None)

    dZ = np.sqrt(0.5 * dt) * (gaussian[0] + 1j * gaussian[1])
    g_dagger = np.swapaxes(ham.g_tensor.conj(), 0, 1)
    h_eff = ham.T[0][None, :, :] + np.einsum("ijk,nk->nij", ham.g_tensor, alpha_before)
    creation = np.einsum("ijk,nk->nij", g_dagger, dZ.conj())

    expected_phia = phia_before
    expected_phia = expected_phia - dt * np.einsum("nij,nje->nie", h_eff, phia_before)
    expected_phia = expected_phia - np.einsum("nij,nje->nie", creation, phia_before)
    expected_alpha = alpha_before - dt * ham.w0 * alpha_before + dZ
    assert np.allclose(walkers.phia, expected_phia)
    assert np.allclose(walkers.coherent_state_shift, expected_alpha)
    assert not hasattr(walkers, "_ito_weight_fac")


@pytest.mark.unit
def test_ito_update_weight_is_noop_without_importance_sampling():
    nwalkers = 4
    alpha = np.zeros(2, dtype=np.complex128)
    phia = np.array([1.0, 0.0], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia])
    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=0, nbasis=2, nwalkers=nwalkers)
    walkers.weight = np.array([1.0, 2.0j, -3.0, 4.0 - 0.5j], dtype=np.complex128)
    walkers._ito_weight_fac = np.array([3.0, 1.0, np.inf, 7.0], dtype=np.complex128)
    weight_before = walkers.weight.copy()

    prop = EulerItoPropagator(0.01)
    ovlp = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.complex128)
    ovlp_new = np.array([2.0, 1.0, 2.0j, -1.0], dtype=np.complex128)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        prop.update_weight(walkers, ovlp, ovlp_new)

    assert np.allclose(walkers.weight, weight_before)


@pytest.mark.unit
def test_ito_propagate_walkers_refreshes_overlap_without_weight_ratio(monkeypatch):
    nwalkers = 2
    alpha = np.zeros(2, dtype=np.complex128)
    phia = np.array([1.0, 0.0], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia])
    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=0, nbasis=2, nwalkers=nwalkers)
    walkers.weight = np.array([1.5 + 0.2j, -0.5j], dtype=np.complex128)
    weight_before = walkers.weight.copy()
    trial = _OverlapTrial([2.0j, -1.0 + 0.5j])

    prop = EulerItoPropagator(0.01)

    def fake_propagate(walkers_arg, hamiltonian_arg, trial_arg):
        walkers_arg.phia *= 2.0

    monkeypatch.setattr(prop, "propagate", fake_propagate)
    prop.propagate_walkers(walkers, hamiltonian=None, trial=trial)

    assert trial.calls == 1
    assert np.allclose(walkers.weight, weight_before)
    assert np.allclose(walkers.ovlp, trial.ovlp)
    assert np.allclose(walkers.phia, 2.0 * np.array([[[1.0], [0.0]]] * nwalkers))


@pytest.mark.unit
def test_ito_build_requires_unnormalized_trial():
    ham = HolsteinModel(g=0.7, t=0.2, w0=1.3, nsites=2, pbc=False)
    ham.build()
    alpha = np.zeros(2, dtype=np.complex128)
    phia = np.array([1.0, 0.0], dtype=np.complex128)[:, None]
    walkers = EPhCSWalkers(
        np.column_stack([alpha, phia]), nup=1, ndown=0, nbasis=2, nwalkers=1
    )

    prop = EulerItoPropagator(0.01)
    with pytest.raises(TypeError, match="unnormalized coherent-state trial"):
        prop.build(ham, trial=_NormalizedTrial(), walkers=walkers)

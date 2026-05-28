import numpy as np
import pytest

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers
from ipie.addons.free_projection.propagation.ito_propagator_fp import (
    EulerItoPropagatorFP,
    construct_trial_eph_mean_field,
)


class _UnnormalizedTrial:
    coherent_state_convention = "unnormalized"


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


@pytest.mark.unit
def test_ito_fp_default_propagation_matches_unsubtracted_update(monkeypatch):
    nsites = 3
    nwalkers = 2
    dt = 0.04
    ham = HolsteinModel(g=0.7, t=0.2, w0=1.3, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5j, 0.25], dtype=np.complex128)[:, None]
    walkers = EPhCSWalkers(
        np.column_stack([alpha, phia]), nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers
    )

    prop = EulerItoPropagatorFP(dt)
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

    assert np.allclose(prop.eph_mean_field, np.zeros(nsites))
    assert np.allclose(walkers.phia, expected_phia)
    assert np.allclose(walkers.coherent_state_shift, expected_alpha)


@pytest.mark.unit
def test_ito_fp_explicit_mean_field_subtraction_fixed_noise(monkeypatch):
    nsites = 3
    nwalkers = 2
    dt = 0.04
    mean_field = np.array([0.2 + 0.1j, -0.3 + 0.05j, 0.15 - 0.2j])
    ham = HolsteinModel(g=0.7, t=0.2, w0=1.3, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5j, 0.25], dtype=np.complex128)[:, None]
    walkers = EPhCSWalkers(
        np.column_stack([alpha, phia]), nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers
    )
    walkers.weight = np.array([1.0 + 0.5j, -0.25j], dtype=np.complex128)
    walkers.phase = np.array([1.0, -1.0j], dtype=np.complex128)
    weight_before = walkers.weight.copy()
    phase_before = walkers.phase.copy()

    prop = EulerItoPropagatorFP(dt, mean_field_shift=mean_field)
    prop.build(ham, trial=_UnnormalizedTrial(), walkers=walkers)

    identity = np.eye(nsites, dtype=np.complex128)
    expected_residual = ham.g_tensor.copy()
    expected_residual -= identity[:, :, None] * mean_field[None, None, :]
    assert np.allclose(prop.eph_mean_field, mean_field)
    assert np.allclose(prop.g_tensor_residual, expected_residual)
    assert np.allclose(prop.g_tensor_residual_dagger, np.swapaxes(expected_residual.conj(), 0, 1))

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
    h_eff = ham.T[0][None, :, :] + np.einsum("ijk,nk->nij", ham.g_tensor, alpha_before)
    creation = np.einsum("ijk,nk->nij", prop.g_tensor_residual_dagger, dZ.conj())
    expected_phia = phia_before
    expected_phia = expected_phia - dt * np.einsum("nij,nje->nie", h_eff, phia_before)
    expected_phia = expected_phia - np.einsum("nij,nje->nie", creation, phia_before)
    expected_alpha = alpha_before - dt * ham.w0 * alpha_before - dt * mean_field.conj() + dZ

    assert np.allclose(walkers.phia, expected_phia)
    assert np.allclose(walkers.coherent_state_shift, expected_alpha)
    assert np.allclose(walkers.weight, weight_before)
    assert np.allclose(walkers.phase, phase_before)


@pytest.mark.unit
def test_ito_fp_mean_field_shift_shape_is_validated():
    ham = HolsteinModel(g=0.7, t=0.2, w0=1.3, nsites=3, pbc=False)
    ham.build()
    prop = EulerItoPropagatorFP(0.01, mean_field_shift=np.ones(2))
    with pytest.raises(ValueError, match="mean_field_shift must have shape"):
        prop.build(ham, trial=_UnnormalizedTrial())


class _SimpleCoherentTrial:
    coherent_state_convention = "unnormalized"

    def __init__(self, psia, beta_shift=None):
        self.psia = np.asarray(psia, dtype=np.complex128)
        self.psib = np.zeros((self.psia.shape[0], 0), dtype=np.complex128)
        self.beta_shift = (
            np.zeros(self.psia.shape[0], dtype=np.complex128)
            if beta_shift is None
            else np.asarray(beta_shift, dtype=np.complex128)
        )
        self.ndown = 0


class _ToyozawaLikeTrial(_SimpleCoherentTrial):
    def __init__(self, psia, beta_shift=None):
        super().__init__(psia, beta_shift=beta_shift)
        sites = np.arange(self.psia.shape[0])
        self.perms = np.array([np.roll(sites, shift) for shift in range(sites.size)])
        self.kcoeffs = np.ones(self.psia.shape[0], dtype=np.complex128)


@pytest.mark.unit
def test_construct_trial_eph_mean_field_localized_holstein_density():
    ham = HolsteinModel(g=0.8, t=0.2, w0=1.3, nsites=4, pbc=False)
    ham.build()
    psia = np.zeros((ham.N, 1), dtype=np.complex128)
    psia[2, 0] = 1.0
    trial = _SimpleCoherentTrial(psia)

    mean_field = construct_trial_eph_mean_field(ham, trial)
    expected = np.zeros(ham.N, dtype=np.complex128)
    expected[2] = ham.g
    assert np.allclose(mean_field, expected)


@pytest.mark.unit
def test_ito_fp_build_computes_trial_mean_field_when_enabled():
    ham = HolsteinModel(g=0.8, t=0.2, w0=1.3, nsites=4, pbc=False)
    ham.build()
    psia = np.zeros((ham.N, 1), dtype=np.complex128)
    psia[2, 0] = 1.0
    trial = _SimpleCoherentTrial(psia)

    prop = EulerItoPropagatorFP(0.01, mean_field_subtraction=True)
    prop.build(ham, trial=trial)

    expected = np.zeros(ham.N, dtype=np.complex128)
    expected[2] = ham.g
    assert np.allclose(prop.eph_mean_field, expected)


@pytest.mark.unit
def test_construct_trial_eph_mean_field_toyozawa_uniform_holstein_density():
    ham = HolsteinModel(g=0.8, t=0.2, w0=1.3, nsites=4, pbc=False)
    ham.build()
    psia = np.zeros((ham.N, 1), dtype=np.complex128)
    psia[0, 0] = 1.0
    trial = _ToyozawaLikeTrial(psia)

    mean_field = construct_trial_eph_mean_field(ham, trial)
    expected = np.full(ham.N, ham.g / ham.N, dtype=np.complex128)
    assert np.allclose(mean_field, expected)

import numpy as np
import pytest
from scipy.linalg import expm

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers
from ipie.addons.free_projection.propagation.ito_propagator_fp import (
    EulerItoPropagatorFP,
    construct_trial_eph_mean_field,
)
from ipie.addons.free_projection.propagation.ito_second_order_fp import (
    ItoSymmSplitImportancePropagatorFP,
    ItoSymmSplitPropagatorFP,
)


class _UnnormalizedTrial:
    coherent_state_convention = "unnormalized"


def _batch_expm_apply(generators, phi):
    propagators = np.stack([expm(generator) for generator in generators])
    return np.einsum("nij,nje->nie", propagators, phi)


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
def test_ito_fp_reference_energy_scales_weight_and_log():
    class Walkers:
        pass

    walkers = Walkers()
    walkers.weight = np.array([1.0, 2.0j, -3.0], dtype=np.complex128)
    walkers.phase = np.array([1.0, -1.0j, 0.5 + 0.5j], dtype=np.complex128)
    walkers.weight_log = np.array([0.0, 0.5, -1.0], dtype=np.complex128)
    weight_before = walkers.weight.copy()
    phase_before = walkers.phase.copy()
    weight_log_before = walkers.weight_log.copy()

    dt = 0.01
    reference_energy = -1.75
    prop = EulerItoPropagatorFP(dt, reference_energy=reference_energy)
    prop.update_weight(walkers, eshift=0.0)

    log_factor = dt * reference_energy
    assert np.allclose(walkers.weight, weight_before * np.exp(log_factor))
    assert np.allclose(walkers.phase, phase_before)
    assert np.allclose(walkers.weight_log, weight_log_before + log_factor)


@pytest.mark.unit
def test_ito_symm_importance_gaussian_log_likelihood_ratio():
    drift = np.array(
        [[0.2 + 0.1j, -0.3 + 0.4j], [0.5 - 0.2j, -0.1 - 0.6j]],
        dtype=np.complex128,
    )
    dW = np.array(
        [[0.01 - 0.02j, -0.03 + 0.04j], [0.05 + 0.02j, -0.01 - 0.03j]],
        dtype=np.complex128,
    )
    step_size = 0.125

    log_ratio = ItoSymmSplitImportancePropagatorFP.gaussian_log_likelihood_ratio(
        drift, dW, step_size
    )
    expected = (
        -2.0 * np.real(np.sum(drift.conj() * dW, axis=1))
        - step_size * np.sum(np.abs(drift) ** 2, axis=1)
    )
    assert np.allclose(log_ratio, expected)


@pytest.mark.unit
def test_ito_symm_importance_update_weight_splits_overlap_ratio_and_phase():
    class Walkers:
        pass

    walkers = Walkers()
    walkers.nwalkers = 3
    walkers.weight = np.array([2.0, 3.0, 4.0], dtype=np.complex128)
    walkers.phase = np.array([1.0, -1.0j, 1.0j], dtype=np.complex128)
    walkers.weight_log = np.array([0.0, 0.5, -0.25], dtype=np.complex128)

    dt = 0.1
    reference_energy = -1.2
    prop = ItoSymmSplitImportancePropagatorFP(
        dt,
        reference_energy=reference_energy,
        zero_overlap_threshold=1.0e-12,
    )

    ovlp = np.array([1.0 + 0.0j, 1.0j, 0.0j], dtype=np.complex128)
    ovlp_new = np.array([2.0j, -3.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    log_likelihood = np.array([0.2, -0.1, 0.5])

    weight_before = walkers.weight.copy()
    phase_before = walkers.phase.copy()
    weight_log_before = walkers.weight_log.copy()
    prop.update_weight(walkers, ovlp, ovlp_new, log_likelihood=log_likelihood)

    log_scalar = log_likelihood[:2] + dt * reference_energy
    ratio = ovlp_new[:2] / ovlp[:2]
    expected_weight = weight_before[:2] * np.exp(log_scalar) * np.abs(ratio)
    expected_phase = phase_before[:2] * ratio / np.abs(ratio)
    expected_log = weight_log_before[:2] + log_scalar + np.log(np.abs(ratio))

    assert np.allclose(walkers.weight[:2], expected_weight)
    assert np.allclose(walkers.phase[:2], expected_phase)
    assert np.allclose(walkers.weight_log[:2], expected_log)
    assert walkers.weight[2] == 0.0
    assert np.isneginf(walkers.weight_log[2].real)


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
def test_ito_symm_split_middle_creation_fixed_noise(monkeypatch):
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

    prop = ItoSymmSplitPropagatorFP(dt, mean_field_shift=mean_field)
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

    half_damp = np.exp(-0.5 * dt * ham.w0)
    alpha_1 = half_damp * alpha_before
    h_eff_1 = ham.T[0][None, :, :] + np.einsum("ijk,nk->nij", ham.g_tensor, alpha_1)
    phia_1 = _batch_expm_apply(-0.5 * dt * h_eff_1, phia_before)

    dZ = np.sqrt(0.5 * dt) * (gaussian[0] + 1j * gaussian[1])
    creation = np.einsum("ijk,nk->nij", prop.g_tensor_residual_dagger, dZ.conj())
    alpha_2 = alpha_1 - dt * mean_field.conj() + dZ
    phia_2 = _batch_expm_apply(-creation, phia_1)

    h_eff_2 = ham.T[0][None, :, :] + np.einsum("ijk,nk->nij", ham.g_tensor, alpha_2)
    expected_phia = _batch_expm_apply(-0.5 * dt * h_eff_2, phia_2)
    expected_alpha = half_damp * alpha_2

    assert np.allclose(walkers.phia, expected_phia)
    assert np.allclose(walkers.coherent_state_shift, expected_alpha)


@pytest.mark.unit
def test_ito_symm_split_zero_coupling_skips_creation_noise(monkeypatch):
    nsites = 3
    nwalkers = 2
    dt = 0.04
    ham = HolsteinModel(g=0.0, t=0.2, w0=1.3, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5j, 0.25], dtype=np.complex128)[:, None]
    walkers = EPhCSWalkers(
        np.column_stack([alpha, phia]), nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers
    )

    prop = ItoSymmSplitPropagatorFP(dt)
    prop.build(ham, trial=_UnnormalizedTrial(), walkers=walkers)

    def fail_normal(*args, **kwargs):
        raise AssertionError("zero residual creation should not sample noise")

    monkeypatch.setattr(np.random, "normal", fail_normal)

    alpha_before = walkers.coherent_state_shift.copy()
    phia_before = walkers.phia.copy()
    prop.propagate(walkers, ham, trial=None)

    expected_alpha = np.exp(-dt * ham.w0) * alpha_before
    expected_phia = _batch_expm_apply(
        np.broadcast_to(-dt * ham.T[0], (nwalkers, nsites, nsites)).copy(), phia_before
    )

    assert np.allclose(walkers.coherent_state_shift, expected_alpha)
    assert np.allclose(walkers.phia, expected_phia)


@pytest.mark.unit
def test_ito_symm_importance_middle_creation_uses_one_full_noise(monkeypatch):
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

    prop = ItoSymmSplitImportancePropagatorFP(dt, force_bias="zero")
    prop.build(ham, trial=_UnnormalizedTrial(), walkers=walkers)

    dW = np.array(
        [[0.01 - 0.02j, -0.03 + 0.04j, 0.02 + 0.01j],
         [0.05 + 0.02j, -0.01 - 0.03j, 0.04 - 0.02j]],
        dtype=np.complex128,
    )
    calls = []
    likelihood_calls = []
    expected_log_likelihood = np.array([0.3, -0.4])

    def fake_noise(walkers_arg, hamiltonian_arg, step_size):
        assert walkers_arg is walkers
        assert hamiltonian_arg is ham
        calls.append(step_size)
        return dW.copy()

    monkeypatch.setattr(prop, "sample_complex_noise_with_step", fake_noise)

    def fake_log_likelihood(drift, dW_arg, step_size):
        likelihood_calls.append((drift.copy(), dW_arg.copy(), step_size))
        return expected_log_likelihood.copy()

    monkeypatch.setattr(prop, "gaussian_log_likelihood_ratio", fake_log_likelihood)

    log_likelihood = prop.propagate(walkers, ham, trial=_UnnormalizedTrial())

    assert calls == [dt]
    assert len(likelihood_calls) == 1
    drift, dW_seen, step_size = likelihood_calls[0]
    assert np.allclose(drift, np.zeros((nwalkers, nsites)))
    assert np.allclose(dW_seen, dW)
    assert step_size == dt
    assert np.allclose(log_likelihood, expected_log_likelihood)


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


class _GaugeDerivativeTrial(_UnnormalizedTrial):
    def __init__(self, A, B):
        self.A = np.asarray(A, dtype=np.complex128)
        self.B = np.asarray(B, dtype=np.complex128)

    def calc_ito_log_derivatives(
        self,
        walkers,
        g_tensor_residual_dagger,
        zero_overlap_threshold=1.0e-14,
    ):
        return self.A.copy(), self.B.copy()


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


@pytest.mark.unit
def test_ito_symm_phase_cancel_split_gauge_sets_adjusted_B_and_drift():
    nsites = 3
    nwalkers = 2
    ham = HolsteinModel(g=0.7, t=0.2, w0=1.3, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5j, 0.25], dtype=np.complex128)[:, None]
    walkers = EPhCSWalkers(
        np.column_stack([alpha, phia]), nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers
    )

    A = np.array(
        [[0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j],
         [0.1 - 0.2j, 0.4 + 0.3j, -0.2 + 0.6j]],
        dtype=np.complex128,
    )
    B = np.array(
        [[-0.1 + 0.3j, 0.2 - 0.5j, 0.7 + 0.1j],
         [0.3 + 0.2j, -0.6 + 0.1j, 0.2 - 0.4j]],
        dtype=np.complex128,
    )
    trial = _GaugeDerivativeTrial(A, B)

    prop = ItoSymmSplitImportancePropagatorFP(
        0.01,
        mean_field_shift=np.zeros(nsites),
        split_gauge="phase_cancel",
    )
    prop.build(ham, trial=trial, walkers=walkers)

    delta_lambda, A_out, B_gauged = prop.construct_split_gauge(walkers, ham, trial)
    drift = prop.construct_force_bias(walkers, ham, trial, A=A_out, B=B_gauged)

    np.testing.assert_allclose(delta_lambda, B + A.conj())
    np.testing.assert_allclose(B_gauged, -A.conj())
    np.testing.assert_allclose(drift, A.conj())

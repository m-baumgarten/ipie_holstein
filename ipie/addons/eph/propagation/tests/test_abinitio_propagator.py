import numpy as np
import pytest
import scipy.linalg

from ipie.addons.eph.estimators.local_energy_abinitio import phonon_local_energy
from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.propagation.abinitio import AbInitioEPhPropagator
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial
from ipie.addons.eph.walkers.abinitio import AbInitioEPhWalkers
from ipie.propagation.propagator import Propagator


def build_hamiltonian():
    eps_kj = np.array([[0.1, 0.3], [0.4, 0.7], [1.0, 1.2]])
    omega_qnu = np.array([[0.5, 0.8], [1.1, 1.4], [1.7, 2.0]])
    g_qnu_kmn = (
        np.arange(3 * 2 * 3 * 2 * 2).reshape(3, 2, 3, 2, 2) / 50.0
        + 0.1j * np.arange(3 * 2 * 3 * 2 * 2).reshape(3, 2, 3, 2, 2) / 70.0
    )
    return AbInitioEPhHamiltonian(eps_kj, g_qnu_kmn, omega_qnu)


def build_trial(ham):
    psi_kj = np.array(
        [[1.0 + 0.2j, 0.4 - 0.1j], [0.3 + 0.5j, 0.2], [-0.6j, 0.8 + 0.3j]]
    )
    beta_qnu = np.array(
        [[0.2 + 0.1j, -0.3 + 0.4j], [0.5 - 0.2j, 0.7j], [-0.1 + 0.6j, 0.9]]
    )
    return AbInitioDD2Trial(psi_kj, beta_qnu).build(ham)


def build_walkers():
    phi = np.array(
        [
            [[0.7 + 0.3j, -0.2 + 0.5j], [0.1 - 0.4j, 0.9], [0.6j, -0.1 + 0.2j]],
            [[-0.3 + 0.1j, 0.8 - 0.2j], [0.5 + 0.4j, -0.7j], [0.2, 0.3 - 0.6j]],
        ]
    )
    X = np.array(
        [
            [[0.2 - 0.1j, -0.4 + 0.2j], [0.7 + 0.3j, 0.1], [-0.2j, 0.6 - 0.5j]],
            [[-0.3 + 0.4j, 0.2j], [0.5 - 0.1j, -0.2 + 0.8j], [0.4, -0.7j]],
        ]
    )
    return AbInitioEPhWalkers(phi, X, nwalkers=2)


@pytest.mark.unit
def test_abinitio_hamiltonian_dispatches_to_abinitio_propagator():
    assert Propagator[AbInitioEPhHamiltonian] is AbInitioEPhPropagator


@pytest.mark.unit
def test_abinitio_propagate_electron_split_matches_reference():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    walkers = build_walkers()
    walkers.build(trial)
    phi0 = walkers.phi_kj.copy()

    dt = 0.03
    prop = AbInitioEPhPropagator(dt, electron_step="split", energy_offset=0.0)
    prop.build(ham, trial, walkers)
    prop.propagate_electron(walkers, ham, trial)

    exp_half = np.exp(-0.5 * dt * ham.eps_kj)
    expected = phi0 * exp_half[None, :, :]
    expected_flat = expected.reshape(walkers.nwalkers, ham.nbasis)
    for iw in range(walkers.nwalkers):
        exp_eph = scipy.linalg.expm(-dt * ham.construct_eph_matrix(walkers.X_qnu[iw]))
        expected_flat[iw] = exp_eph @ expected_flat[iw]
    expected *= exp_half[None, :, :]

    np.testing.assert_allclose(walkers.phi_kj, expected)
    np.testing.assert_allclose(walkers.phia[:, :, 0], expected.reshape(2, ham.nbasis))


@pytest.mark.unit
def test_abinitio_default_energy_offset_centers_electronic_propagator():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    walkers = build_walkers()
    walkers.build(trial)
    phi0 = walkers.phi_kj.copy()

    dt = 0.03
    prop = AbInitioEPhPropagator(dt, electron_step="split")
    prop.build(ham, trial, walkers)
    prop.propagate_electron(walkers, ham, trial)

    assert prop.energy_offset == pytest.approx(np.min(ham.eps_kj))
    exp_half = np.exp(-0.5 * dt * (ham.eps_kj - prop.energy_offset))
    expected = phi0 * exp_half[None, :, :]
    expected_flat = expected.reshape(walkers.nwalkers, ham.nbasis)
    for iw in range(walkers.nwalkers):
        exp_eph = scipy.linalg.expm(-dt * ham.construct_eph_matrix(walkers.X_qnu[iw]))
        expected_flat[iw] = exp_eph @ expected_flat[iw]
    expected *= exp_half[None, :, :]

    np.testing.assert_allclose(walkers.phi_kj, expected)


@pytest.mark.unit
def test_abinitio_propagate_electron_combined_matches_reference():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    walkers = build_walkers()
    walkers.build(trial)
    phi0 = walkers.phi_kj.copy()

    dt = 0.03
    prop = AbInitioEPhPropagator(dt, electron_step="combined", energy_offset=0.0)
    prop.build(ham, trial, walkers)
    prop.propagate_electron(walkers, ham, trial)

    h1 = np.diag(ham.eps_kj.reshape(ham.nbasis)).astype(np.complex128)
    expected = phi0.reshape(walkers.nwalkers, ham.nbasis)
    for iw in range(walkers.nwalkers):
        h_eff = h1 + ham.construct_eph_matrix(walkers.X_qnu[iw])
        expected[iw] = scipy.linalg.expm(-dt * h_eff) @ expected[iw]

    np.testing.assert_allclose(walkers.phi_kj, expected.reshape(walkers.phi_kj.shape))


@pytest.mark.unit
def test_abinitio_propagate_phonons_deterministic_with_seed():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    walkers = build_walkers()
    walkers.build(trial)
    X0 = walkers.X_qnu.copy()
    weight0 = walkers.weight.copy()

    dt_step = 0.01
    eshift = 0.4
    e_old = phonon_local_energy(ham, walkers, trial).real
    drift = trial.calc_phonon_gradient(walkers).real.astype(np.complex128)

    seed = 17
    np.random.seed(seed)
    noise = np.random.normal(scale=np.sqrt(dt_step), size=walkers.X_qnu.shape)
    X_expected = X0 + dt_step * drift + noise

    ref_walkers = build_walkers()
    ref_walkers.build(trial)
    ref_walkers.X_qnu[:] = X_expected
    ref_walkers.phonon_disp = ref_walkers.X_qnu
    trial.calc_overlap(ref_walkers)
    e_new = phonon_local_energy(ham, ref_walkers, trial).real
    weight_expected = weight0 * np.exp(-0.5 * dt_step * (e_old + e_new) + dt_step * eshift)

    prop = AbInitioEPhPropagator(0.02)
    prop.build(ham, trial, walkers)
    np.random.seed(seed)
    prop.propagate_phonons(walkers, ham, trial, dt_step=dt_step, eshift=eshift)

    np.testing.assert_allclose(walkers.X_qnu, X_expected)
    np.testing.assert_allclose(walkers.weight, weight_expected)


@pytest.mark.unit
def test_abinitio_hybrid_energy_ignores_zero_weight_killed_walkers():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    walkers = build_walkers()
    walkers.build(trial)

    prop = AbInitioEPhPropagator(0.02)
    prop.build(ham, trial, walkers)

    old_weight = np.array([0.0, 2.0], dtype=np.complex128)
    walkers.weight = np.array([0.0, 1.0], dtype=np.complex128)
    prop.update_hybrid_energy(walkers, old_weight, eshift=0.3)

    assert np.all(np.isfinite(walkers.hybrid_energy))
    assert walkers.hybrid_energy[0] == pytest.approx(0.3)
    assert walkers.hybrid_energy[1] == pytest.approx(0.3 - np.log(0.5) / 0.02)


@pytest.mark.unit
def test_abinitio_propagate_walkers_smoke():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    walkers = build_walkers()
    walkers.build(trial)

    prop = AbInitioEPhPropagator(0.005)
    prop.build(ham, trial, walkers)
    np.random.seed(123)
    prop.propagate_walkers(walkers, ham, trial, eshift=0.1)

    assert np.all(np.isfinite(walkers.phi_kj))
    assert np.all(np.isfinite(walkers.X_qnu))
    assert np.all(np.isfinite(walkers.weight))
    assert np.all(np.isfinite(walkers.ovlp))

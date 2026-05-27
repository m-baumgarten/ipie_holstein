import numpy as np
import pytest

from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial


def build_test_hamiltonian():
    eps_kj = np.array([[0.1, 0.3], [0.4, 0.7], [1.0, 1.2]])
    omega_qnu = np.array([[0.5, 0.8], [1.1, 1.4], [1.7, 2.0]])
    g_qnu_kmn = np.zeros((3, 2, 3, 2, 2), dtype=np.complex128)
    return AbInitioEPhHamiltonian(eps_kj, g_qnu_kmn, omega_qnu)


def build_test_trial():
    psi_kj = np.array(
        [[1.0 + 0.2j, 0.4 - 0.1j], [0.3 + 0.5j, 0.2], [-0.6j, 0.8 + 0.3j]]
    )
    beta_qnu = np.array(
        [[0.2 + 0.1j, -0.3 + 0.4j], [0.5 - 0.2j, 0.7j], [-0.1 + 0.6j, 0.9]]
    )
    return AbInitioDD2Trial(psi_kj, beta_qnu)


@pytest.mark.unit
def test_abinitio_dd2_trial_defaults_and_centers():
    ham = build_test_hamiltonian()
    trial = build_test_trial().build(ham)

    assert trial.nk == 3
    assert trial.nq == 3
    assert trial.nband == 2
    assert trial.nmode == 2
    assert trial.nbasis == 6
    assert trial.nphonon_modes == 6
    np.testing.assert_array_equal(trial.minus_q, np.array([0, 2, 1]))

    beta_minus_conj = np.conj(trial.beta_qnu[trial.minus_q])
    expected_x0 = (trial.beta_qnu + beta_minus_conj) / np.sqrt(2.0 * ham.omega_qnu)
    expected_p0 = -1.0j * np.sqrt(0.5 * ham.omega_qnu) * (
        trial.beta_qnu - beta_minus_conj
    )

    np.testing.assert_allclose(trial.x0_qnu, expected_x0)
    np.testing.assert_allclose(trial.p0_qnu, expected_p0)
    np.testing.assert_allclose(trial.electronic_amplitudes_flat(), trial.psi_kj.reshape(6))


@pytest.mark.unit
def test_abinitio_dd2_trial_translated_centers():
    ham = build_test_hamiltonian()
    trial = build_test_trial().build(ham)

    x_qnu_R, p_qnu_R = trial.translated_centers()
    expected_x = trial.x0_qnu[:, :, None] * np.conj(trial.phase_qR)[:, None, :]
    expected_p = trial.p0_qnu[:, :, None] * np.conj(trial.phase_qR)[:, None, :]

    assert x_qnu_R.shape == (3, 2, 3)
    assert p_qnu_R.shape == (3, 2, 3)
    np.testing.assert_allclose(x_qnu_R, expected_x)
    np.testing.assert_allclose(p_qnu_R, expected_p)


@pytest.mark.unit
def test_abinitio_dd2_trial_accepts_custom_phases():
    psi_kj = np.ones((2, 2), dtype=np.complex128)
    beta_qnu = np.ones((2, 1), dtype=np.complex128)
    translations = np.array([0, 2, 5])
    phase_kR = np.exp(0.2j * np.arange(6)).reshape(2, 3)
    phase_qR = np.exp(-0.3j * np.arange(6)).reshape(2, 3)
    phase_KR = np.exp(0.4j * np.arange(3))

    trial = AbInitioDD2Trial(
        psi_kj,
        beta_qnu,
        translations=translations,
        phase_kR=phase_kR,
        phase_qR=phase_qR,
        phase_KR=phase_KR,
    )

    np.testing.assert_array_equal(trial.translations, translations)
    np.testing.assert_allclose(trial.phase_kR, phase_kR)
    np.testing.assert_allclose(trial.phase_qR, phase_qR)
    np.testing.assert_allclose(trial.phase_KR, phase_KR)


@pytest.mark.unit
def test_abinitio_dd2_trial_validation():
    ham = build_test_hamiltonian()

    with pytest.raises(ValueError, match="psi_kj"):
        AbInitioDD2Trial(np.ones(3), np.ones((3, 2)))

    with pytest.raises(ValueError, match="beta_qnu"):
        AbInitioDD2Trial(np.ones((3, 2)), np.ones(3))

    with pytest.raises(ValueError, match="omega_qnu"):
        AbInitioDD2Trial(np.ones((3, 2)), np.ones((2, 2))).build(ham)

    with pytest.raises(ValueError, match="phase_qR"):
        AbInitioDD2Trial(
            np.ones((3, 2)),
            np.ones((3, 2)),
            phase_kR=np.ones((3, 2)),
            phase_qR=np.ones((3, 3)),
        )

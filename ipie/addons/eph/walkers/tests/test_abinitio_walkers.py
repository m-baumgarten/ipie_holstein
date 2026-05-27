import numpy as np
import pytest

from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial
from ipie.addons.eph.walkers.abinitio import AbInitioEPhWalkers


def build_hamiltonian():
    eps_kj = np.array([[0.1, 0.3], [0.4, 0.7], [1.0, 1.2]])
    omega_qnu = np.array([[0.5, 0.8], [1.1, 1.4], [1.7, 2.0]])
    g_qnu_kmn = np.zeros((3, 2, 3, 2, 2), dtype=np.complex128)
    return AbInitioEPhHamiltonian(eps_kj, g_qnu_kmn, omega_qnu)


def build_trial():
    psi_kj = np.array(
        [[1.0 + 0.2j, 0.4 - 0.1j], [0.3 + 0.5j, 0.2], [-0.6j, 0.8 + 0.3j]]
    )
    beta_qnu = np.array(
        [[0.2 + 0.1j, -0.3 + 0.4j], [0.5 - 0.2j, 0.7j], [-0.1 + 0.6j, 0.9]]
    )
    return AbInitioDD2Trial(psi_kj, beta_qnu).build(build_hamiltonian())


def build_walker_state():
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
    return phi, X


def direct_contractions(trial, phi_kj, X_qnu):
    nR = trial.nR
    A_el = np.zeros(nR, dtype=np.complex128)
    A_ph = np.zeros(nR, dtype=np.complex128)
    log_norm = 0.25 * np.sum(np.log(trial.omega_qnu / np.pi))

    for iR in range(nR):
        A_el[iR] = np.sum(trial.psi_kj.conj() * phi_kj * trial.phase_kR[:, iR, None])

        x_R = trial.x0_qnu * trial.phase_qR[:, iR, None].conj()
        p_R = trial.p0_qnu * trial.phase_qR[:, iR, None].conj()
        exponent = np.sum(
            -0.5 * trial.omega_qnu * np.abs(X_qnu - x_R) ** 2
            - 1.0j * p_R.conj() * X_qnu
            + 0.5j * x_R.conj() * p_R
        )
        A_ph[iR] = np.exp(log_norm + exponent)

    T = trial.phase_KR * A_el * A_ph
    O = np.sum(T)
    S = np.einsum("R,R,kR->k", A_ph, trial.phase_KR, trial.phase_kR, optimize=True)
    M = np.einsum("R,qR->q", T, trial.phase_qR.conj(), optimize=True)
    return A_el, A_ph, T, O, S, M


@pytest.mark.unit
def test_abinitio_walkers_cache_optimized_contractions():
    trial = build_trial()
    phi, X = build_walker_state()
    walkers = AbInitioEPhWalkers(phi, X, nwalkers=2)

    walkers.build(trial)

    assert walkers.A_el.shape == (2, trial.nR)
    assert walkers.A_ph.shape == (2, trial.nR)
    assert walkers.T.shape == (2, trial.nR)
    assert walkers.O.shape == (2,)
    assert walkers.S.shape == (2, trial.nk)
    assert walkers.M.shape == (2, trial.nq)

    for iw in range(2):
        A_el, A_ph, T, O, S, M = direct_contractions(trial, phi[iw], X[iw])
        np.testing.assert_allclose(walkers.A_el[iw], A_el)
        np.testing.assert_allclose(walkers.A_ph[iw], A_ph)
        np.testing.assert_allclose(walkers.T[iw], T)
        np.testing.assert_allclose(walkers.O[iw], O)
        np.testing.assert_allclose(walkers.S[iw], S)
        np.testing.assert_allclose(walkers.M[iw], M)


@pytest.mark.unit
def test_abinitio_walkers_trial_overlap_delegates_to_cache():
    trial = build_trial()
    phi, X = build_walker_state()
    walkers = AbInitioEPhWalkers(phi, X, nwalkers=2)

    ovlp = trial.calc_overlap(walkers)

    np.testing.assert_allclose(ovlp, walkers.O)
    np.testing.assert_allclose(walkers.ovlp, walkers.O)


@pytest.mark.unit
def test_abinitio_walkers_phonon_log_gradient_uses_M_over_O():
    trial = build_trial()
    phi, X = build_walker_state()
    walkers = AbInitioEPhWalkers(phi, X, nwalkers=2)
    walkers.build(trial)

    grad = walkers.phonon_log_gradient(trial)
    prefactor = trial.omega_qnu * trial.x0_qnu - 1.0j * trial.p0_qnu
    expected = -trial.omega_qnu[None, :, :] * X
    expected += (walkers.M[:, :, None] / walkers.O[:, None, None]) * prefactor[None, :, :]

    np.testing.assert_allclose(grad, expected)


@pytest.mark.unit
def test_abinitio_walkers_validation_and_reortho():
    trial = build_trial()
    phi, X = build_walker_state()
    walkers = AbInitioEPhWalkers(phi[0], X[0], nwalkers=3)

    assert walkers.phi_kj.shape == (3, trial.nk, trial.nband)
    assert walkers.X_qnu.shape == (3, trial.nq, trial.nmode)

    walkers.build(trial)
    old_O = walkers.O.copy()
    norms = np.linalg.norm(walkers.phi_kj.reshape(3, trial.nbasis), axis=1)
    detR = walkers.reortho()

    np.testing.assert_allclose(detR, norms)
    np.testing.assert_allclose(
        np.linalg.norm(walkers.phi_kj.reshape(3, trial.nbasis), axis=1), 1.0
    )
    np.testing.assert_allclose(walkers.O, old_O / norms)

    with pytest.raises(ValueError, match="initial_electron"):
        AbInitioEPhWalkers(np.ones((2, 3, 2)), X[0], nwalkers=3)

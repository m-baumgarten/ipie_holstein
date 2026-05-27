import numpy as np
import pytest

from ipie.addons.eph.estimators.local_energy_abinitio import local_energy_abinitio
from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian
from ipie.addons.eph.trial_wavefunction.abinitio import AbInitioDD2Trial
from ipie.addons.eph.walkers.abinitio import AbInitioEPhWalkers


def build_hamiltonian():
    eps_kj = np.array([[0.1, 0.3], [0.4, 0.7], [1.0, 1.2]])
    omega_qnu = np.array([[0.5, 0.8], [1.1, 1.4], [1.7, 2.0]])
    g_qnu_kmn = (
        np.arange(3 * 2 * 3 * 2 * 2).reshape(3, 2, 3, 2, 2) / 50.0
        + 0.1j * np.arange(3 * 2 * 3 * 2 * 2).reshape(3, 2, 3, 2, 2) / 70.0
    )
    return AbInitioEPhHamiltonian(eps_kj, g_qnu_kmn, omega_qnu, ecore=0.25)


def build_trial(ham):
    psi_kj = np.array(
        [[1.0 + 0.2j, 0.4 - 0.1j], [0.3 + 0.5j, 0.2], [-0.6j, 0.8 + 0.3j]]
    )
    beta_qnu = np.array(
        [[0.2 + 0.1j, -0.3 + 0.4j], [0.5 - 0.2j, 0.7j], [-0.1 + 0.6j, 0.9]]
    )
    return AbInitioDD2Trial(psi_kj, beta_qnu).build(ham)


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


def direct_overlap_factors(trial, phi_kj, X_qnu):
    A_el = np.zeros(trial.nR, dtype=np.complex128)
    A_ph = np.zeros(trial.nR, dtype=np.complex128)
    log_norm = 0.25 * np.sum(np.log(trial.omega_qnu / np.pi))

    for iR in range(trial.nR):
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
    return A_el, A_ph, T, np.sum(T)


def direct_local_energy(ham, trial, phi_kj, X_qnu):
    A_el, A_ph, T, O = direct_overlap_factors(trial, phi_kj, X_qnu)
    el_num = 0.0j
    eph_num = 0.0j
    ph_num = 0.0j

    for iR in range(trial.nR):
        phase_weight = trial.phase_KR[iR] * A_ph[iR]

        el_num += phase_weight * np.sum(
            ham.eps_kj * trial.psi_kj.conj() * phi_kj * trial.phase_kR[:, iR, None]
        )

        for ik in range(ham.nk):
            for iq in range(ham.nq):
                kout = ham.k_plus_q[ik, iq]
                phase_out = trial.phase_kR[kout, iR]
                for inu in range(ham.nmode):
                    factor = np.sqrt(2.0 * ham.omega_qnu[iq, inu]) * X_qnu[iq, inu]
                    for m in range(ham.nband):
                        for n in range(ham.nband):
                            eph_num += (
                                phase_weight
                                * factor
                                * ham.g_qnu_kmn[iq, inu, ik, m, n]
                                * trial.psi_kj[kout, m].conj()
                                * phi_kj[ik, n]
                                * phase_out
                            )

        h_ph_R = 0.0j
        for iq in range(ham.nq):
            for inu in range(ham.nmode):
                omega = ham.omega_qnu[iq, inu]
                x_R = trial.x0_qnu[iq, inu] * trial.phase_qR[iq, iR].conj()
                p_R = trial.p0_qnu[iq, inu] * trial.phase_qR[iq, iR].conj()
                X = X_qnu[iq, inu]
                h_ph_R += (
                    omega**2 * X.conj() * x_R
                    - 0.5 * omega**2 * abs(x_R) ** 2
                    + 0.5 * abs(p_R) ** 2
                    - 1.0j * omega * (X - x_R).conj() * p_R
                )
        ph_num += T[iR] * h_ph_R

    energy = np.zeros(4, dtype=np.complex128)
    energy[1] = el_num / O + ham.ecore
    energy[2] = eph_num / O
    energy[3] = ph_num / O
    energy[0] = np.sum(energy[1:])
    return energy


@pytest.mark.unit
def test_abinitio_local_energy_matches_direct_translation_sum():
    ham = build_hamiltonian()
    trial = build_trial(ham)
    phi, X = build_walker_state()
    walkers = AbInitioEPhWalkers(phi, X, nwalkers=2)

    energy = local_energy_abinitio(None, ham, walkers, trial)

    for iw in range(2):
        expected = direct_local_energy(ham, trial, phi[iw], X[iw])
        np.testing.assert_allclose(energy[iw], expected)

    np.testing.assert_allclose(energy[:, 0], np.sum(energy[:, 1:], axis=1))

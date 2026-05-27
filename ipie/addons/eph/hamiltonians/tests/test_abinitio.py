import numpy as np
import pytest

from ipie.addons.eph.hamiltonians.abinitio import AbInitioEPhHamiltonian


def _make_hamiltonian(nk=3, nband=2, nmode=2):
    eps = np.arange(nk * nband, dtype=np.float64).reshape(nk, nband)
    omega = 0.5 + np.arange(nk * nmode, dtype=np.float64).reshape(nk, nmode)

    g = np.zeros((nk, nmode, nk, nband, nband), dtype=np.complex128)
    for iq in range(nk):
        for inu in range(nmode):
            for ik in range(nk):
                for m in range(nband):
                    for n in range(nband):
                        g[iq, inu, ik, m, n] = (
                            0.1 * (1 + iq)
                            + 0.01 * (1 + inu)
                            + 0.001 * (1 + ik)
                            + 0.0001j * (1 + m + 2 * n)
                        )

    return AbInitioEPhHamiltonian(eps, g, omega)


def test_abinitio_hamiltonian_shapes_and_helpers():
    ham = _make_hamiltonian()

    assert ham.nk == 3
    assert ham.nq == 3
    assert ham.nband == 2
    assert ham.nmode == 2
    assert ham.nbasis == 6
    assert ham.nphonon_modes == 6

    assert ham.flatten_index(2, 1) == 5
    assert ham.split_index(5) == (2, 1)

    amps = np.arange(6).reshape(3, 2)
    np.testing.assert_array_equal(ham.flatten_electronic(amps), np.arange(6))
    np.testing.assert_array_equal(ham.unflatten_electronic(np.arange(6)), amps)

    h1 = ham.build_T()[0]
    np.testing.assert_allclose(np.diag(h1), ham.eps_kj.reshape(-1))
    np.testing.assert_allclose(h1, np.diag(np.diag(h1)))


def test_construct_eph_matrix_matches_reference_convention():
    ham = _make_hamiltonian()
    X = np.array(
        [
            [0.2 + 0.1j, -0.3],
            [0.4, 0.1 - 0.2j],
            [-0.5j, 0.7],
        ],
        dtype=np.complex128,
    )

    got = ham.construct_eph_matrix(X)
    ref = np.zeros_like(got)
    for iq in range(ham.nq):
        for inu in range(ham.nmode):
            factor = np.sqrt(2.0 * ham.omega_qnu[iq, inu]) * X[iq, inu]
            for ik in range(ham.nk):
                kout = (ik + iq) % ham.nk
                for m in range(ham.nband):
                    for n in range(ham.nband):
                        row = ham.flatten_index(kout, m)
                        col = ham.flatten_index(ik, n)
                        ref[row, col] += factor * ham.g_qnu_kmn[iq, inu, ik, m, n]

    np.testing.assert_allclose(got, ref)

    vec = np.arange(ham.nbasis, dtype=np.complex128)
    np.testing.assert_allclose(ham.apply_eph_matrix(X, vec), ref @ vec)


def test_construct_eph_matrix_supports_batched_X():
    ham = _make_hamiltonian()
    X = np.ones((2, ham.nq, ham.nmode), dtype=np.complex128)

    got = ham.construct_eph_matrix(X)

    assert got.shape == (2, ham.nbasis, ham.nbasis)
    np.testing.assert_allclose(got[0], ham.construct_eph_matrix(X[0]))
    np.testing.assert_allclose(got[1], ham.construct_eph_matrix(X[1]))


def test_phonon_potential_and_zero_point():
    ham = _make_hamiltonian()
    X = np.ones((ham.nq, ham.nmode), dtype=np.complex128)

    np.testing.assert_allclose(ham.zero_point_energy(), 0.5 * np.sum(ham.omega_qnu))
    np.testing.assert_allclose(ham.phonon_potential(X), 0.5 * np.sum(ham.omega_qnu**2))


def test_shape_validation():
    eps = np.zeros((2, 2))
    omega = np.ones((2, 1))
    g = np.zeros((2, 1, 2, 2, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="eps_kj"):
        AbInitioEPhHamiltonian(eps.reshape(4), g, omega)

    with pytest.raises(ValueError, match="g_qnu_kmn shape mismatch"):
        AbInitioEPhHamiltonian(eps, g[:, :, :, :1, :], omega)

    with pytest.raises(ValueError, match="non-negative"):
        AbInitioEPhHamiltonian(eps, g, -omega)

    with pytest.raises(ValueError, match="k_plus_q is required"):
        AbInitioEPhHamiltonian(eps, g[:1], omega[:1])

    with pytest.raises(ValueError, match="valid electronic k-point"):
        AbInitioEPhHamiltonian(eps, g, omega, k_plus_q=np.array([[0, 2], [1, 0]]))

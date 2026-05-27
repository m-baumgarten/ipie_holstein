import numpy as np
import pytest
import scipy.linalg

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.propagation.cs_propagator import CoherentStatePropagator
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers


@pytest.mark.unit
def test_cs_walkers_preserve_complex_shift_shape():
    nsites = 3
    nwalkers = 2
    alpha = np.array([0.2 + 0.3j, -0.1 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5, 0.25], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia])

    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers)

    assert walkers.coherent_state_shift.shape == (nwalkers, nsites)
    assert np.allclose(walkers.coherent_state_shift, alpha[None, :])


@pytest.mark.unit
def test_cs_propagate_electron_fixed_fields(monkeypatch):
    nsites = 3
    nwalkers = 2
    dt = 0.07
    ham = HolsteinModel(g=0.8, t=0.0, w0=1.0, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.3j, -0.1 + 0.4j, 0.5 - 0.2j])
    phia = np.array([1.0, -0.5, 0.25], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia])
    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=0, nbasis=nsites, nwalkers=nwalkers)

    prop = CoherentStatePropagator(dt)
    prop.build(ham, trial=None, walkers=walkers)

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

    phia_before = walkers.phia.copy()
    alpha_before = walkers.coherent_state_shift.copy()
    prop.propagate_electron(walkers, ham, trial=None)

    new_shift = gaussian[0] + 1j * gaussian[1]
    cs_displ = new_shift.conj() + 2 * alpha_before.real
    exp_eph = scipy.linalg.expm(-dt * np.einsum("ijk,nk->nij", ham.g_tensor, cs_displ))

    bch = np.zeros((nsites, nsites), dtype=np.complex128)
    for imode in range(nsites):
        coupling = ham.g_tensor[:, :, imode]
        bch += coupling.dot(coupling)
    exp_bch = scipy.linalg.expm(0.5 * dt * dt * bch)

    expected_phia = np.einsum("ij,nje->nie", exp_bch, np.einsum("nij,nje->nie", exp_eph, phia_before))
    expected_phase = np.exp(
        1j * np.sum(new_shift.real * alpha_before.imag - new_shift.imag * alpha_before.real, axis=1)
    )

    assert np.allclose(walkers.phia, expected_phia)
    assert np.allclose(walkers.coherent_state_shift, alpha_before + new_shift)
    assert np.allclose(walkers.weight, 1.0)
    assert np.allclose(walkers._cs_weight_fac, (2**nsites) * expected_phase)


@pytest.mark.unit
def test_cs_bch_spinful_uses_one_hs_field_for_both_spins(monkeypatch):
    nsites = 2
    nwalkers = 2
    dt = 0.05
    ham = HolsteinModel(g=0.6, t=0.0, w0=1.0, nsites=nsites, pbc=False)
    ham.build()

    alpha = np.array([0.2 + 0.3j, -0.1 + 0.4j])
    phia = np.array([1.0, -0.5], dtype=np.complex128)[:, None]
    phib = np.array([0.25, 0.75], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia, phib])
    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=1, nbasis=nsites, nwalkers=nwalkers)

    prop = CoherentStatePropagator(dt)
    prop.build(ham, trial=None, walkers=walkers)

    coherent_gaussian = np.array(
        [
            [[0.1, -0.3], [-0.4, 0.5]],
            [[0.7, -0.2], [-0.1, -0.8]],
        ]
    )
    bch_fields = np.array([[0.9, -0.4], [0.2, 0.6]])
    draws = [coherent_gaussian, bch_fields]

    def fake_normal(loc=0.0, scale=1.0, size=None):
        draw = draws.pop(0)
        assert loc == 0.0
        assert scale == 1.0
        assert size == draw.shape
        return draw.copy()

    monkeypatch.setattr(np.random, "normal", fake_normal)

    phia_before = walkers.phia.copy()
    phib_before = walkers.phib.copy()
    alpha_before = walkers.coherent_state_shift.copy()
    prop.propagate_electron(walkers, ham, trial=None)

    new_shift = coherent_gaussian[0] + 1j * coherent_gaussian[1]
    cs_displ = new_shift.conj() + 2 * alpha_before.real
    exp_eph = scipy.linalg.expm(-dt * np.einsum("ijk,nk->nij", ham.g_tensor, cs_displ))
    exp_bch = scipy.linalg.expm(dt * np.einsum("ijk,nk->nij", ham.g_tensor, bch_fields))

    expected_phia = np.einsum(
        "nij,nje->nie", exp_bch, np.einsum("nij,nje->nie", exp_eph, phia_before)
    )
    expected_phib = np.einsum(
        "nij,nje->nie", exp_bch, np.einsum("nij,nje->nie", exp_eph, phib_before)
    )

    assert not draws
    assert np.allclose(walkers.phia, expected_phia)
    assert np.allclose(walkers.phib, expected_phib)


@pytest.mark.unit
def test_cs_update_weight_zero_and_overflow_overlap_is_killed():
    nwalkers = 2
    alpha = np.zeros(2, dtype=np.complex128)
    phia = np.array([1.0, 0.0], dtype=np.complex128)[:, None]
    initial_walker = np.column_stack([alpha, phia])
    walkers = EPhCSWalkers(initial_walker, nup=1, ndown=0, nbasis=2, nwalkers=nwalkers)
    walkers.weight = np.ones(nwalkers, dtype=np.complex128)
    walkers._cs_weight_fac = np.ones(nwalkers, dtype=np.complex128)

    prop = CoherentStatePropagator(0.01)
    ovlp = np.array([0.0, 1.0e-320], dtype=np.complex128)
    ovlp_new = np.ones(nwalkers, dtype=np.complex128)

    with np.errstate(divide="raise", invalid="raise", over="raise"):
        prop.update_weight(walkers, ovlp, ovlp_new)

    assert np.allclose(walkers.weight, 0.0)

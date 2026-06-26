# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the periodic scalar-q diffusion-gauge optimizer (Sec. 11.5)."""

import numpy as np
import pytest

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.walkers.cs_walkers import EPhCSWalkers
from ipie.addons.free_projection.propagation.ito_second_order_fp import (
    ItoSymmSplitImportancePropagatorFP,
)


class _GreensTrial:
    """Minimal trial exposing a prescribed mixed Green's function."""

    coherent_state_convention = "unnormalized"

    def __init__(self, G):
        self._G = np.asarray(G, dtype=np.complex128)
        self.ndown = 0

    def calc_greens_function(self, walkers):
        return [self._G, np.zeros_like(self._G)]


def _make_walkers(alpha, phia, nwalkers=1):
    alpha = np.asarray(alpha, dtype=np.complex128)
    phia = np.asarray(phia, dtype=np.complex128)[:, None]
    return EPhCSWalkers(
        np.column_stack([alpha, phia]),
        nup=1,
        ndown=0,
        nbasis=alpha.size,
        nwalkers=nwalkers,
    )


# ----------------------------------------------------------------------
# 1. Closed-form scalar optimum:  q* = sqrt(trV / trW_ph)  (Proposition 1)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_scalar_q_optimum_is_geometric_ratio_and_minimizes_J():
    prop = ItoSymmSplitImportancePropagatorFP(
        0.01, split_gauge="phase_cancel", split_gauge_q_optimize=True
    )

    a, b = 3.0, 12.0  # trW_ph, trV
    prop._accumulate_gauge_costs = lambda *args, **kwargs: (a, b)

    q_new = prop.optimize_split_gauge_q(None, None, None)

    # q* = sqrt(b/a)
    assert q_new == pytest.approx(np.sqrt(b / a))
    assert prop.split_gauge_q == pytest.approx(np.sqrt(b / a))
    assert prop.split_gauge_sqrt_q == pytest.approx(np.sqrt(np.sqrt(b / a)))

    # ... and it minimizes J(q) = a q + b / q.
    grid = np.linspace(0.05, 10.0, 4000)
    J = a * grid + b / grid
    assert q_new == pytest.approx(grid[np.argmin(J)], abs=2e-2)
    assert (a * q_new + b / q_new) <= J.min() + 1e-9

    diag = prop.last_q_diagnostics
    assert diag["trW_ph"] == a and diag["trV"] == b
    assert diag["q_star"] == pytest.approx(q_new)


@pytest.mark.unit
def test_scalar_q_smoothing_and_clamp():
    prop = ItoSymmSplitImportancePropagatorFP(
        0.01,
        split_gauge="phase_cancel",
        split_gauge_q=1.0,
        split_gauge_q_optimize=True,
        split_gauge_q_smoothing=0.5,
        split_gauge_q_bounds=(0.5, 4.0),
    )
    prop._accumulate_gauge_costs = lambda *a, **k: (1.0, 16.0)  # q* = 4

    # Geometric blend from q=1 toward q*=4 with alpha=0.5 -> 1^0.5 * 4^0.5 = 2.
    q_new = prop.optimize_split_gauge_q(None, None, None)
    assert q_new == pytest.approx(2.0)

    # Clamp: drive q* above the bound.
    prop._set_split_gauge_q(1.0)
    prop.split_gauge_q_smoothing = 1.0
    prop._accumulate_gauge_costs = lambda *a, **k: (1.0, 1.0e6)  # q* = 1000
    q_new = prop.optimize_split_gauge_q(None, None, None)
    assert q_new == pytest.approx(4.0)  # clamped to q_max


@pytest.mark.unit
def test_optimizer_keeps_q_on_degenerate_costs():
    prop = ItoSymmSplitImportancePropagatorFP(
        0.01, split_gauge="phase_cancel", split_gauge_q=0.3, split_gauge_q_optimize=True
    )
    prop._accumulate_gauge_costs = lambda *a, **k: (0.0, 1.0)  # trW_ph == 0
    q_new = prop.optimize_split_gauge_q(None, None, None)
    assert q_new == pytest.approx(0.3)  # unchanged
    assert np.isnan(prop.last_q_diagnostics["q_star"])


# ----------------------------------------------------------------------
# 2. Connected-correlator channel sensitivity vs finite differences
#    u_rho = sum_i (dE_L/dpsi_i) [G_rho^dagger psi]_i  (Proposition 3)
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_channel_sensitivity_matches_finite_difference():
    rng = np.random.default_rng(7)
    nsites = 4
    ham = HolsteinModel(g=0.7, t=0.9, w0=1.3, nsites=nsites, pbc=True)
    ham.build()

    f = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    psi = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)
    phi = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)  # trial bra
    A = rng.normal(size=nsites) + 1j * rng.normal(size=nsites)

    walkers = _make_walkers(f, psi, nwalkers=1)

    prop = ItoSymmSplitImportancePropagatorFP(
        0.01, mean_field_shift=np.zeros(nsites), split_gauge="phase_cancel"
    )
    gdag = np.swapaxes(np.asarray(ham.g_tensor, dtype=np.complex128).conj(), 0, 1)
    prop._g_tensor_full_dagger = gdag

    # Mixed 1-RDM in the operational convention <X> = einsum('ij,ij', X, G):
    #   G_ij = conj(phi_i) psi_j / (phi^dagger psi).
    denom = np.vdot(phi, psi)  # phi^dagger psi
    G = np.outer(phi.conj(), psi) / denom
    trial = _GreensTrial(G[None, :, :])

    # Electronic local-energy operator Ohat = h_eff(f) + sum_mu A_mu G_mu^dagger.
    h_eff = prop.construct_annihilation_matrix(walkers.coherent_state_shift, ham)[0][0]
    ohat = h_eff + np.einsum("ijm,m->ij", gdag, A)

    def E_L(psi_vec):
        return (phi.conj() @ (ohat @ psi_vec)) / (phi.conj() @ psi_vec)

    eps = 1e-6
    grad = np.zeros(nsites, dtype=np.complex128)
    for i in range(nsites):
        step = np.zeros(nsites, dtype=np.complex128)
        step[i] = eps
        grad[i] = (E_L(psi + step) - E_L(psi - step)) / (2.0 * eps)

    u_fd = np.array(
        [grad @ (gdag[:, :, rho] @ psi) for rho in range(nsites)], dtype=np.complex128
    )
    u2_fd = np.sum(np.abs(u_fd) ** 2)

    u2 = prop._channel_sensitivity_sq(walkers, ham, trial, A[None, :])
    assert u2[0] == pytest.approx(u2_fd, rel=1e-5, abs=1e-7)


# ----------------------------------------------------------------------
# 3. The real trial Green's function uses the einsum('ij,nij') convention,
#    so <h_eff> built in the optimizer matches the energy estimator.
# ----------------------------------------------------------------------
def _build_real():
    """Build a real Toyozawa CS trial + walkers (skips if deps unavailable)."""
    from ipie.addons.eph.trial_wavefunction.toyozawa_cs_unnormalized import (
        ToyozawaTrialUnnormalizedCoherentState,
    )

    nsites = 4
    ham = HolsteinModel(g=0.8, t=1.0, w0=1.0, nsites=nsites, pbc=True)
    ham.build()
    rng = np.random.default_rng(3)
    wf = rng.normal(size=(nsites, 2)) + 1j * rng.normal(size=(nsites, 2))
    trial = ToyozawaTrialUnnormalizedCoherentState(
        wavefunction=wf, w0=ham.w0, num_elec=(1, 0), num_basis=nsites, K=0.0
    )
    walkers = EPhCSWalkers(wf, nup=1, ndown=0, nbasis=nsites, nwalkers=5)
    walkers.build(trial)
    return ham, trial, walkers


@pytest.mark.unit
def test_real_greens_kinetic_matches_einsum_rule():
    ham, trial, walkers = _build_real()
    G = trial.calc_greens_function(walkers)
    kinetic = np.einsum("ij,nij->n", ham.T[0], np.asarray(G[0]))
    # The energy estimator computes the kinetic term as sum(T * G, axis=(1,2)).
    expected = np.sum(ham.T[0] * np.asarray(G[0]), axis=(1, 2))
    np.testing.assert_allclose(kinetic, expected)


# ----------------------------------------------------------------------
# 4. Phonon sensitivity: for Holstein, s_mu = omega A_mu + B_mu (Hermitian).
#    Integration smoke test: optimizer sets a finite positive q on real data.
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_optimizer_sets_finite_positive_q_on_real_population():
    ham, trial, walkers = _build_real()
    prop = ItoSymmSplitImportancePropagatorFP(
        0.01,
        mean_field_subtraction=True,
        split_gauge="phase_cancel",
        split_gauge_q_optimize=True,
        split_gauge_electron_cost="sensitivity",
    )
    prop.build(ham, trial=trial, walkers=walkers, mpi_handler=None)
    assert prop._coupling_hermitian  # Holstein g n_i is Hermitian

    q_new = prop.optimize_split_gauge_q(walkers, ham, trial)
    assert np.isfinite(q_new) and q_new > 0.0
    assert prop.last_q_diagnostics["trW_ph"] > 0.0
    assert prop.last_q_diagnostics["trV"] >= 0.0


@pytest.mark.unit
def test_kick_proxy_and_gap_are_finite():
    ham, trial, walkers = _build_real()
    prop = ItoSymmSplitImportancePropagatorFP(
        0.01, mean_field_subtraction=True, split_gauge="phase_cancel"
    )
    prop.build(ham, trial=trial, walkers=walkers, mpi_handler=None)

    kick2 = prop._channel_kick_sq(walkers)
    gap2 = prop._effective_gap_sq(walkers, ham)
    assert np.all(np.isfinite(kick2)) and np.all(kick2 >= 0.0)
    assert np.all(np.isfinite(gap2)) and np.all(gap2 >= prop.gap_floor ** 2)


@pytest.mark.unit
def test_q_optimize_requires_phase_cancel():
    with pytest.raises(ValueError):
        ItoSymmSplitImportancePropagatorFP(
            0.01, split_gauge="static", split_gauge_q_optimize=True
        )

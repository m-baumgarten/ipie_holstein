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
"""
Validate `EPhPropagatorFree` (no importance sampling) by piecewise
comparison to the analytic action of the matching part of the Trotter
factorization.

Pieces tested
-------------
1. `propagate_electron` — fully deterministic given the walker. Should
   apply the symmetric Trotter
       e^{-Δτ T/2} · e^{-Δτ H_eph(X)} · e^{-Δτ T/2}
   to walker.phia, with H_eph(X) diagonal in the lattice basis.

2. `propagate_phonons` (free) — Diffusion MC on phonon coordinates.
   With a fixed RNG seed we can verify:
     - walker.phonon_disp = X_old + Gaussian noise with the right scale,
     - walker.weight is multiplied by the expected potential factors
       and the zero-point correction.

Sign convention: matches the QMC code, i.e. H_eph = +g Σ_i n_i (b_i +
b_i^†), so e^{-Δτ H_eph(X)} on a Slater determinant multiplies row i
by exp(-Δτ √(2mω) g X_i). This is opposite in sign to the docstring
convention of HolsteinModel; the two are unitary equivalents (X → -X).
"""

import numpy as np
import pytest
import scipy.linalg


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _make_walker_and_propagator(seed, N, dt, g=0.5, t=1.0, w0=1.0,
                                K=0.0, nwalkers=2):
    from ipie.systems import Generic                                   # noqa: F401
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
    from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
    from ipie.addons.eph.propagation.eph_propagator import EPhPropagatorFree

    rng = np.random.RandomState(seed)
    alpha = (rng.randn(N) + 1j * rng.randn(N)).astype(np.complex128)
    beta = (0.4 * rng.randn(N)).astype(np.complex128)

    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=N, pbc=True)
    ham.build()

    wfn_T = np.column_stack([beta[:, None], alpha[:, None]])
    trial = ToyozawaTrial(wavefunction=wfn_T, w0=w0, num_elec=(1, 0),
                          num_basis=N, K=K)

    # Walker init with a simple non-trivial phia and zero phonon shift.
    phia0 = (rng.randn(N, 1) + 1j * rng.randn(N, 1)).astype(np.complex128)
    init = np.column_stack([np.zeros(N)[:, None], phia0])
    walkers = EPhWalkers(init, nup=1, ndown=0, nbasis=N,
                         nwalkers=nwalkers, verbose=False)
    walkers.build(trial)
    # Override walker 0 with a chosen X and phia (tests use walker 0 only).
    X0 = (0.6 * rng.randn(N)).astype(np.float64)
    walkers.phonon_disp[0] = X0.astype(np.complex128)
    walkers.phia[0] = phia0
    walkers.weight[0] = 1.0  # start from a known reference weight

    prop = EPhPropagatorFree(time_step=dt)
    prop.build(ham, trial=trial, walkers=walkers)
    return ham, trial, walkers, prop, phia0, X0


# ============================================================================
# Test A: propagate_electron is exact second-order Trotter on phia
# ============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("dt", [0.005, 0.05])
def test_propagate_electron_deterministic(seed, N, dt):
    ham, trial, walkers, prop, phia0, X0 = _make_walker_and_propagator(
        seed, N, dt
    )

    phia_before = walkers.phia[0].copy()
    X_before = walkers.phonon_disp[0].copy()
    weight_before = complex(walkers.weight[0])

    prop.propagate_electron(walkers, ham, trial)

    # Reference: e^{-dt T/2} diag(exp(c X)) e^{-dt T/2} phia
    # where c = -sqrt(2 m w0) * dt * g (because g_tensor is diagonal +g).
    T = ham.T[0]
    expT_half = scipy.linalg.expm(-0.5 * dt * T)
    c = -np.sqrt(2.0 * ham.m * ham.w0) * dt * ham.g
    expEph_diag = np.exp(c * X_before.real)  # diagonal of exp(c X)
    expEph = np.diag(expEph_diag).astype(np.complex128)
    phia_ref = expT_half @ expEph @ expT_half @ phia_before

    err = np.max(np.abs(walkers.phia[0] - phia_ref))
    assert err < 1e-12, (
        f"propagate_electron mismatch: max|err| = {err:.3e}\n"
        f"got:\n{walkers.phia[0]}\nexpected:\n{phia_ref}"
    )

    # X and weight should be unchanged by propagate_electron
    assert np.max(np.abs(walkers.phonon_disp[0] - X_before)) == 0, \
        "propagate_electron should not modify phonon_disp"
    assert walkers.weight[0] == weight_before, \
        "propagate_electron should not modify weight"


# ============================================================================
# Test B: propagate_phonons (free) deterministic given fixed RNG
# ============================================================================

@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("dt", [0.005, 0.05])
def test_propagate_phonons_free_deterministic(seed, N, dt):
    ham, trial, walkers, prop, phia0, X0 = _make_walker_and_propagator(
        seed, N, dt
    )

    # Snapshot
    X_before_all = walkers.phonon_disp.copy()
    w_before_all = walkers.weight.copy()

    # First weight kick: weight *= exp(-dt_ph * 0.25 * m w^2 sum X^2)
    dt_ph = 0.5 * dt
    mw2 = ham.m * ham.w0 ** 2
    pot1 = 0.25 * mw2 * np.sum(X_before_all.real ** 2, axis=1)
    expected_w_after_first = w_before_all * np.exp(-dt_ph * pot1)

    # Generate the SAME random Gaussian noise the propagator will use.
    rng_seed = 12345
    np.random.seed(rng_seed)
    noise = np.random.normal(loc=0.0, scale=np.sqrt(dt_ph / ham.m),
                             size=(walkers.nwalkers, N))
    X_after_step = X_before_all + noise

    # Second weight kick (same form, evaluated at new X)
    pot2 = 0.25 * mw2 * np.sum(X_after_step.real ** 2, axis=1)
    expected_w_after_second = expected_w_after_first * np.exp(-dt_ph * pot2)

    # Zero-point correction: weight *= exp(dt_ph * N * w0 / 2)
    expected_w_final = expected_w_after_second * np.exp(dt_ph * N * ham.w0 / 2)

    # Now run the propagator with the same RNG seed
    np.random.seed(rng_seed)
    prop.propagate_phonons(walkers, ham, trial)

    # Compare X
    err_X = np.max(np.abs(walkers.phonon_disp - X_after_step))
    assert err_X < 1e-14, f"phonon_disp mismatch: max err = {err_X:.3e}"

    # Compare weight
    err_w = np.max(np.abs(walkers.weight - expected_w_final))
    assert err_w < 1e-12, (
        f"weight mismatch: max err = {err_w:.3e}\n"
        f"got      = {walkers.weight}\n"
        f"expected = {expected_w_final}"
    )

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
Validate `ToyozawaTrial.calc_phonon_gradient` and
`ToyozawaTrial.calc_phonon_laplacian` against jax autodiff of the
*full* (electronic × phononic) overlap function.

Importance-sampling DMC for el-ph systems uses the drift

    D_l = d_{X_l} <Psi_T | psi, X> / <Psi_T | psi, X>

and the kinetic-energy contribution to the local energy involves the
laplacian

    L = sum_l d^2_{X_l} <Psi_T | psi, X> / <Psi_T | psi, X>.

Both are derivatives of the FULL trial overlap, not the bosonic part
alone -- the electronic Slater determinant doesn't depend on X, but the
sum over translations is weighted by phonon overlaps that DO depend on
X, so the electronic structure does enter the gradient/laplacian
through the per-permutation weighting.

We compare the code's analytic formulas to numerical derivatives
obtained from `jax.grad` of the same analytic Gaussian * Slater overlap
expression (which we've already validated against Fock-basis ED in
test_toyozawa_trial_fock_ed.py).

Real beta only (the QMC code's calc_phonon_overlap_perms phase term is
only canonical for real beta -- see test_toyozawa_trial_fock_ed.py
docstring).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------
# Analytic <Psi_T | phi, X> as a JAX-differentiable function of X (real).
#
# Same formula the QMC code uses internally:
#   <Psi_T | phi, X>
#     = sum_ip kcoeffs[ip].conj() * det(T_ip(alpha)^* . phia)
#               * prod_k exp(-(mw/2) (X_k - T_ip(beta)_k)^2)
# Note: the (mw/pi)^{N/4} prefactor is dropped (it's an overall constant
# in X, doesn't affect drift or laplacian).
# ----------------------------------------------------------------------------


def _circ_perm_1D(N):
    """Match ipie's fixed convention: perms[ip][k] = (k - ip) mod N
    (forward translation by ip)."""
    sites = np.arange(N)
    out = np.stack([np.roll(sites, ip) for ip in range(N)], axis=0)
    return out


def _toyo_overlap(X, alpha, beta_real, K, perms, mw, nup):
    """Returns <Psi_T | phi, X> as a complex jax scalar.

    Inputs
    ------
    X           : shape (N,) real walker positions
    alpha       : shape (N, nup) complex (electronic SD of trial)
    beta_real   : shape (N,) real (phonon shift of trial; canonical
                  beta = sqrt(mw/2) * <X>)
    K           : scalar momentum (BZ value)
    perms       : (N, N) int permutation table (perms[ip][k] = (k-ip)%N)
    mw          : product m*w0
    nup         : 1 (used as a constant)
    walkers_phia: shape (N, nup) complex
    """
    pass  # see _toyo_overlap_factory


def _toyo_overlap_factory(alpha, beta_real, K, perms, mw, walkers_phia):
    """Returns a JAX function f(X) -> complex scalar.

    Important: the QMC code's `calc_phonon_overlap_perms` uses
    `beta_shift = beta * sqrt(2/(mw))` and centers the Gaussian at
    `beta_shift.real`, i.e. at <X>_beta. To match that convention here
    we pass in the *canonical* beta and rescale internally.
    """
    Kn = jnp.exp(1j * K * jnp.arange(perms.shape[0]))
    a_np = np.atleast_2d(np.asarray(alpha))
    if a_np.shape[0] == 1 and a_np.shape[1] != 1:
        a_np = a_np.T  # was (1, N) -> make (N, 1)
    a = jnp.asarray(a_np, dtype=jnp.complex128)
    # beta_shift.real = sqrt(2/(mw)) * canonical beta (real part). The
    # phonon Gaussian is exp(-(mw/2) (X - beta_shift.real)^2).
    b_shift_real = jnp.asarray(beta_real, dtype=jnp.float64) * np.sqrt(2.0 / mw)
    perms_j = jnp.asarray(perms, dtype=jnp.int32)
    phi = jnp.asarray(walkers_phia, dtype=jnp.complex128)

    def f(X):
        out = 0.0 + 0.0j
        for ip in range(perms_j.shape[0]):
            perm = perms_j[ip]
            a_tr = a[perm, :]
            b_tr = b_shift_real[perm]
            M = jnp.conj(a_tr).T @ phi
            el_ov = jnp.linalg.det(M)
            ph_ov = jnp.exp(-0.5 * mw * jnp.sum((X - b_tr) ** 2))
            kc = jnp.conj(Kn[ip])
            out = out + kc * el_ov * ph_ov
        return out

    return f


def _grad_complex(f, X):
    """Returns d f/d X_l as a complex array, using two real-output grads."""
    g_re = jax.grad(lambda x: f(x).real)(X)
    g_im = jax.grad(lambda x: f(x).imag)(X)
    return g_re + 1j * g_im


def _hessian_complex(f, X):
    """Returns d^2 f / d X_l d X_m as a complex matrix."""
    H_re = jax.hessian(lambda x: f(x).real)(X)
    H_im = jax.hessian(lambda x: f(x).imag)(X)
    return H_re + 1j * H_im


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

NSITES = [3, 4]
SEEDS = [0, 1]


def _bz(N):
    return [2 * np.pi * n / N for n in range(N)]


def _params():
    return [(s, n, K) for s in SEEDS for n in NSITES for K in _bz(n)]


def _make(seed, N, K, g=0.5, t=1.0, w0=1.0):
    from ipie.systems import Generic
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
    from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

    rng = np.random.RandomState(seed)
    alpha = (rng.randn(N) + 1j * rng.randn(N)).astype(np.complex128)
    beta = (0.4 * rng.randn(N)).astype(np.complex128)  # real (zero imag)

    sys_ = Generic((1, 0))
    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=N, pbc=True)
    ham.build()

    phia_w = (rng.randn(N, 1) + 1j * rng.randn(N, 1)).astype(np.complex128)
    X_w = (0.5 * rng.randn(N)).astype(np.float64)

    wfn_T = np.column_stack([beta[:, None], alpha[:, None]])
    trial = ToyozawaTrial(wavefunction=wfn_T, w0=w0, num_elec=(1, 0),
                          num_basis=N, K=K)

    init_walker = np.column_stack([np.zeros(N)[:, None], phia_w])
    walkers = EPhWalkers(init_walker, nup=1, ndown=0, nbasis=N, nwalkers=2,
                         verbose=False)
    walkers.build(trial)
    walkers.phonon_disp[0] = X_w.astype(np.complex128)
    walkers.phia[0] = phia_w.copy()

    return ham, trial, walkers, alpha, beta.real, X_w, phia_w


@pytest.mark.unit
@pytest.mark.parametrize("seed,N,K", _params())
def test_phonon_gradient_matches_jax(seed, N, K):
    ham, trial, walkers, alpha, beta_real, X_w, phia_w = _make(seed, N, K)

    # `calc_phonon_gradient` defensively refreshes walkers.ovlp_perm
    # internally; no manual calc_overlap needed.
    drift_code = np.asarray(trial.calc_phonon_gradient(walkers))[0]

    # JAX reference: differentiate the analytic Gaussian * Slater overlap
    # of the trial w.r.t. X_w.
    perms = _circ_perm_1D(N)
    f = _toyo_overlap_factory(alpha, beta_real, K, perms,
                              ham.m * ham.w0, phia_w)
    O = f(jnp.asarray(X_w))
    grad_O = _grad_complex(f, jnp.asarray(X_w))
    drift_jax = np.array(grad_O / O)

    rel = (np.linalg.norm(drift_code - drift_jax)
           / max(np.linalg.norm(drift_jax), 1e-12))
    abs_max = np.max(np.abs(drift_code - drift_jax))
    assert rel < 1e-9 and abs_max < 1e-8, (
        f"calc_phonon_gradient mismatch at (seed={seed}, N={N}, K={K:.4f}):\n"
        f"  code = {drift_code}\n"
        f"  jax  = {drift_jax}\n"
        f"  rel  = {rel:.3e}, amax = {abs_max:.3e}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("seed,N,K", _params())
def test_phonon_laplacian_matches_jax(seed, N, K):
    ham, trial, walkers, alpha, beta_real, X_w, phia_w = _make(seed, N, K)

    # `calc_phonon_laplacian` defensively refreshes walkers.ovlp_perm.
    lap_code = complex(np.asarray(trial.calc_phonon_laplacian(walkers))[0])

    # JAX reference
    perms = _circ_perm_1D(N)
    f = _toyo_overlap_factory(alpha, beta_real, K, perms,
                              ham.m * ham.w0, phia_w)
    O = f(jnp.asarray(X_w))
    H = _hessian_complex(f, jnp.asarray(X_w))
    lap_jax = complex(jnp.trace(H) / O)

    diff = abs(lap_code - lap_jax)
    rel = diff / max(abs(lap_jax), 1e-12)
    assert diff < 1e-8 and rel < 1e-9, (
        f"calc_phonon_laplacian mismatch at (seed={seed}, N={N}, K={K:.4f}):\n"
        f"  code = {lap_code}\n"
        f"  jax  = {lap_jax}\n"
        f"  diff = {diff:.3e}, rel = {rel:.3e}"
    )

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
Regression test for the K-projected Toyozawa variational energy and its
analytic gradient.

We compare the analytic implementations in
    ipie.addons.eph.trial_wavefunction.variational.toyozawa.ToyozawaVariational
against a slow-but-explicit JAX reference that evaluates

    E_K = sum_{j,i} K_j^* K_i <psi_j|H|psi_i>
        / sum_{j,i} K_j^* K_i <psi_j|psi_i>

via a direct double sum (no translation-symmetry trick), and uses
jax.grad for the gradient. The two should agree to numerical precision.

Covers 1D Holstein, single up-electron (the polaron use case).

Run with:  pytest -m unit ipie/addons/eph/trial_wavefunction/tests/test_variational_gradient.py
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Self-contained JAX reference (kept inside the test file so it stays in sync)
# -----------------------------------------------------------------------------


def _cs_overlap(beta_a, beta_b):
    log_ov = jnp.sum(
        -0.5 * (jnp.abs(beta_a) ** 2 + jnp.abs(beta_b) ** 2)
        + jnp.conj(beta_a) * beta_b
    )
    return jnp.exp(log_ov)


def _slater_overlap(psi_a, psi_b):
    return jnp.linalg.det(jnp.conj(psi_a).T @ psi_b)


def _gab(psi_a, psi_b):
    inv_O = jnp.linalg.inv(jnp.conj(psi_a).T @ psi_b)
    return (psi_b @ inv_O @ jnp.conj(psi_a).T).T


def _projected_energy(T, g_tensor, w0, G, beta_a, beta_b):
    kinetic = jnp.sum(T * G)
    el_ph = jnp.einsum(
        "ijk,ij,k->", g_tensor, G, jnp.conj(beta_a) + beta_b
    )
    phonon = w0 * jnp.sum(jnp.conj(beta_a) * beta_b)
    return kinetic + el_ph + phonon


def _cyclic_shift(arr, n):
    # T_{+n}: amplitude carried forward, (T_n psi)[k] = psi[(k-n) mod N].
    return jnp.roll(arr, n, axis=0)


def _unpack(x, nsites, nup):
    N = nsites
    s_re = x[0:N]
    s_im = x[N:2 * N]
    p_re = x[2 * N:2 * N + N * nup].reshape(nup, N).T
    p_im = x[2 * N + N * nup:2 * N + 2 * N * nup].reshape(nup, N).T
    return s_re + 1j * s_im, p_re + 1j * p_im


def _energy_jax(x, K, T, g_tensor, w0, nsites, nup=1):
    shift, psia = _unpack(x, nsites, nup)
    N = nsites
    Kn = jnp.exp(1j * K * jnp.arange(N))
    betas = jnp.stack([_cyclic_shift(shift, n) for n in range(N)], axis=1)
    psias = jnp.stack([_cyclic_shift(psia, n) for n in range(N)], axis=2)
    num = 0.0 + 0.0j
    den = 0.0 + 0.0j
    for j in range(N):
        for i in range(N):
            beta_a, beta_b = betas[:, j], betas[:, i]
            psi_a, psi_b = psias[:, :, j], psias[:, :, i]
            ov = _cs_overlap(beta_a, beta_b) * _slater_overlap(psi_a, psi_b)
            G = _gab(psi_a, psi_b)
            E_proj = _projected_energy(T, g_tensor, w0, G, beta_a, beta_b)
            phase = jnp.conj(Kn[j]) * Kn[i]
            num = num + phase * ov * E_proj
            den = den + phase * ov
    return (num / den).real


def _build_holstein(g, t, w0, nsites, pbc=True):
    T = np.zeros((nsites, nsites))
    for i in range(nsites - 1):
        T[i, i + 1] = -t
        T[i + 1, i] = -t
    if pbc and nsites > 1:
        T[0, -1] = -t
        T[-1, 0] = -t
    g_tensor = np.zeros((nsites, nsites, nsites), dtype=np.complex128)
    for i in range(nsites):
        g_tensor[i, i, i] = g
    return T, g_tensor


def _make_jax_fn(g, t, w0, nsites, K, nup=1):
    T_np, gT_np = _build_holstein(g, t, w0, nsites, pbc=True)
    Tj = jnp.asarray(T_np, dtype=jnp.complex128)
    gj = jnp.asarray(gT_np, dtype=jnp.complex128)

    def f(x):
        return _energy_jax(x, K, Tj, gj, w0, nsites, nup)

    return f, jax.grad(f)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

SEEDS = [0, 1, 2]
NSITES = [4, 6]


def _bz_k_values(nsites):
    """Allowed momenta of the N-site PBC chain: K = 2 pi n / N.

    The Toyozawa K-projector only produces a translation eigenstate at
    these K; for arbitrary real K the analytic single-loop formula and
    the JAX double-sum differ by finite-N edge terms (because
    sum_j e^{i K (j+d) mod N} only collapses to N e^{iKd} when e^{iKN}=1).
    """
    return [2 * np.pi * n / nsites for n in range(nsites)]


def _idparams(seed, nsites):
    return [(seed, nsites, K) for K in _bz_k_values(nsites)]


PARAMS = [p for s in SEEDS for n in NSITES for p in _idparams(s, n)]


@pytest.mark.unit
@pytest.mark.parametrize("seed,nsites,K", PARAMS)
def test_toyozawa_objective_matches_jax(seed, nsites, K):
    """Energy from objective_function == JAX-direct double-sum reference."""
    from ipie.systems import Generic
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.variational.toyozawa import (
        ToyozawaVariational,
    )

    np.random.seed(seed)
    elec = np.random.randn(nsites, 1) + 1j * np.random.randn(nsites, 1)
    shift = np.random.randn(nsites) + 1j * np.random.randn(nsites)

    sys_ = Generic((1, 0))
    ham = HolsteinModel(g=0.5, t=1.0, w0=1.0, nsites=nsites, pbc=True)
    ham.build()
    var = ToyozawaVariational(shift, elec, ham, sys_, K=K, cplx=True)
    x = var.pack_x()

    e_anal = var.objective_function(x)

    e_fn, _ = _make_jax_fn(0.5, 1.0, 1.0, nsites, K, nup=1)
    e_jax = float(e_fn(jnp.asarray(x)))

    assert abs(e_anal - e_jax) < 1e-10, (
        f"Energy mismatch: analytic={e_anal:.12g}, jax={e_jax:.12g}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("seed,nsites,K", PARAMS)
def test_toyozawa_gradient_matches_jax(seed, nsites, K):
    """Analytic gradient(x) == jax.grad(energy_ref)(x)."""
    from ipie.systems import Generic
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.variational.toyozawa import (
        ToyozawaVariational,
    )

    np.random.seed(seed)
    elec = np.random.randn(nsites, 1) + 1j * np.random.randn(nsites, 1)
    shift = np.random.randn(nsites) + 1j * np.random.randn(nsites)

    sys_ = Generic((1, 0))
    ham = HolsteinModel(g=0.5, t=1.0, w0=1.0, nsites=nsites, pbc=True)
    ham.build()
    var = ToyozawaVariational(shift, elec, ham, sys_, K=K, cplx=True)
    x = var.pack_x()

    g_anal = var.gradient(x)

    _, g_fn = _make_jax_fn(0.5, 1.0, 1.0, nsites, K, nup=1)
    g_jax = np.array(g_fn(jnp.asarray(x)))

    # Use both absolute and relative tolerance on the L2 norm of the residual
    rel = np.linalg.norm(g_anal - g_jax) / max(np.linalg.norm(g_jax), 1e-12)
    abs_max = np.max(np.abs(g_anal - g_jax))
    assert rel < 1e-9 and abs_max < 1e-8, (
        f"Gradient mismatch: max|dg|={abs_max:.3e}, "
        f"rel||dg||={rel:.3e}"
    )

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
Validate ToyozawaTrial.calc_overlap and ToyozawaTrial.calc_greens_function
for the QMC walker pipeline against an explicit construction in the
truncated Fock basis.

Setup
-----
Pick a 1D Holstein system (N sites, single up-electron, M phonons/site
truncation). Generate a random walker (phia, phonon_disp = real X) and a
random Toyozawa trial (alpha, beta REAL, K on the BZ).

Then:
  * `<Psi_T | walker>` from the QMC code is compared to
    `<Psi_T | phi> ⊗ |X>` constructed exactly in the Fock basis.
  * `G[w, p, q] = <Psi_T | c_q^dag c_p | walker> / <Psi_T | walker>`
    from the QMC code is compared to the same ratio computed from the
    explicit construction.

Why real beta
-------------
The QMC trial's `calc_phonon_overlap_perms` formula treats the
(beta_shift.real, beta_shift.imag) pair with a non-canonical phase that
ONLY reduces to <coh(beta)|X> when beta_shift is real. This convention
is something to revisit, but the typical Holstein polaron variational
optimum has real beta (by inversion symmetry at K=0; for K != 0 there
is in principle a momentum component, but the existing trials we tested
optimize to real beta at all K within numerical tolerance).

The (mω/π)^{N/4} prefactor missing from the code's overlap is a
multiplicative constant per walker. It cancels in any ratio (energy,
GF, drift), so it doesn't matter for QMC. We strip it before comparing
to make the absolute overlap match too.
"""

import math

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Fock-basis builders (re-used from the variational test)
# -----------------------------------------------------------------------------

def _boson_ops(M):
    import scipy.sparse as sp
    diag = np.sqrt(np.arange(1, M))
    b = sp.diags(diag, offsets=1, format="csr", shape=(M, M))
    bd = sp.diags(diag, offsets=-1, format="csr", shape=(M, M))
    n = sp.diags(np.arange(M), offsets=0, format="csr", shape=(M, M))
    return b, bd, n


def _site_op(op, k, N, M):
    import scipy.sparse as sp
    out = None
    for s in range(N):
        factor = op if s == k else sp.identity(M, format="csr")
        out = factor if out is None else sp.kron(out, factor, format="csr")
    return out


def _coh(beta, M):
    """Coherent state |coh(beta)> in number basis on one site (canonical
    convention beta = sqrt(mw/2) X_beta + i p_beta / sqrt(2 mw))."""
    v = np.zeros(M, dtype=np.complex128)
    log_norm = -0.5 * np.abs(beta) ** 2
    for n in range(M):
        if beta == 0:
            v[n] = 1.0 if n == 0 else 0.0
        else:
            log_amp = n * np.log(beta) - 0.5 * math.lgamma(n + 1)
            v[n] = np.exp(log_norm + log_amp)
    return v


def _phonon_product(beta_canonical, M):
    out = np.array([1.0 + 0j])
    for b in beta_canonical:
        out = np.kron(out, _coh(b, M))
    return out


def _T_m(state, m, N, M):
    """Apply T_{+m}: simultaneous translation of electron AND phonon-site
    labels by +m sites (matches `circ_perm_1D` after the convention fix
    and the corresponding T_m in the variational fock-ED test)."""
    arr = state.reshape((N,) + (M,) * N)
    arr = np.roll(arr, shift=m, axis=0)
    ph_axes = list(range(1, N + 1))
    new_ph_order = [ph_axes[(k - m) % N] for k in range(N)]
    arr = arr.transpose([0] + new_ph_order)
    return arr.reshape(state.shape)


def _build_K_state(alpha, beta_canonical, K, N, M):
    """|Psi_K> = sum_n e^{i K n} T_n(|alpha> ⊗ |coh(beta_canonical)>)
    in the (electron lattice) ⊗ (phonon Fock chain) basis."""
    psi0 = np.kron(alpha.astype(np.complex128),
                   _phonon_product(beta_canonical, M))
    out = np.zeros_like(psi0)
    for m in range(N):
        out = out + np.exp(1j * K * m) * _T_m(psi0, m, N, M)
    return out


def _ho_position_state(X, N, M, m_mass=None, w0=None, mw=None):
    """Position eigenstate |X_0, X_1, ..., X_{N-1}> in the Fock basis,
    expanded as |X> = (mw/pi)^{1/4} sum_n H_n(sqrt(mw) X) / sqrt(2^n n!)
                       * exp(-mw X^2 / 2) |n>
    on each site, then tensor product.

    We DO include the (mw/pi)^{N/4} prefactor so that <coh(beta)|X> here
    matches the analytic <coh(beta)|X>; the QMC code drops it (it
    cancels in ratios), so the test divides it out before comparing.
    """
    if mw is None:
        mw = m_mass * w0
    out = np.array([1.0 + 0j])
    factor = (mw / np.pi) ** 0.25
    sqrt_mw = np.sqrt(mw)
    for k in range(N):
        # Hermite polynomial values H_n(sqrt(mw) X_k), n = 0..M-1, via
        # the recursion H_{n+1}(y) = 2 y H_n(y) - 2 n H_{n-1}(y).
        y = sqrt_mw * X[k]
        H = np.zeros(M, dtype=np.float64)
        H[0] = 1.0
        if M > 1:
            H[1] = 2 * y
            for n in range(1, M - 1):
                H[n + 1] = 2 * y * H[n] - 2 * n * H[n - 1]
        # Wavefunction values <n|X_k>
        psi_n = np.zeros(M, dtype=np.complex128)
        norm_log = np.log(factor) - 0.5 * mw * X[k] ** 2
        for n in range(M):
            log_amp = -0.5 * (n * np.log(2.0) + math.lgamma(n + 1))
            psi_n[n] = H[n] * np.exp(norm_log + log_amp)
        out = np.kron(out, psi_n)
    return out


def _walker_state(phia, X, N, M, mw):
    """|walker> = |phia> ⊗ |X>, where |phia> is an N-site SD with one
    column (single up electron) in the lattice basis, and |X> is the
    truncated position eigenstate on the phonon chain."""
    # Single-electron Slater determinant: |phia> = sum_i phia[i] c_i^dag |0>
    # In the electron-site basis this is just the column phia[:].
    el = phia[:, 0].astype(np.complex128)
    ph = _ho_position_state(X.real, N, M, mw=mw)
    return np.kron(el, ph)


# -----------------------------------------------------------------------------
# Holstein H builder (electron site basis)
# -----------------------------------------------------------------------------

def _build_holstein_H(N, M, t, g, w0, pbc=True):
    """Same H as build_H_holstein_1D in the variational test, replicated
    here to keep the test self-contained."""
    import scipy.sparse as sp
    dim_ph = M ** N
    I_ph = sp.identity(dim_ph, format="csr")

    rows, cols, data = [], [], []
    for i in range(N - 1):
        rows += [i, i + 1]; cols += [i + 1, i]; data += [-t, -t]
    if pbc and N > 1:
        rows += [0, N - 1]; cols += [N - 1, 0]; data += [-t, -t]
    T_el = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    H_kin = sp.kron(T_el, I_ph, format="csr")

    b1, bd1, n1 = _boson_ops(M)
    H_phon_only = sp.csr_matrix((dim_ph, dim_ph))
    for k in range(N):
        H_phon_only = H_phon_only + _site_op(n1, k, N, M)
    H_phon_only *= w0
    H_phon = sp.kron(sp.identity(N, format="csr"), H_phon_only, format="csr")

    H_eph = sp.csr_matrix((N * dim_ph, N * dim_ph))
    for k in range(N):
        proj = sp.csr_matrix(([1.0], ([k], [k])), shape=(N, N))
        x_k = _site_op(b1 + bd1, k, N, M)
        H_eph = H_eph + g * sp.kron(proj, x_k, format="csr")

    return (H_kin + H_phon + H_eph).tocsr()


# -----------------------------------------------------------------------------
# c_q^dag c_p in the lattice ⊗ Fock basis
# -----------------------------------------------------------------------------

def _build_cdag_c(p, q, N, M):
    """One-electron operator c_q^dag c_p ⊗ I_phonon."""
    import scipy.sparse as sp
    rows = [q]; cols = [p]; data = [1.0]
    op_el = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return sp.kron(op_el, sp.identity(M ** N, format="csr"), format="csr")


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

NSITES = [3, 4]
SEEDS = [0, 1]
M_TRUNC = 16  # phonon truncation per site (need ~16 to converge the
              # truncated position eigenstate <n|X> below the test
              # tolerance for X ~ 0.5 and beta ~ 0.4).


def _bz(N):
    return [2 * np.pi * n / N for n in range(N)]


def _params():
    return [(s, n, K) for s in SEEDS for n in NSITES for K in _bz(n)]


def _make_random_trial_and_walker(seed, N, K, g=0.5, t=1.0, w0=1.0):
    """Build a Holstein system, a Toyozawa trial with REAL beta and
    complex alpha, and one walker with a random phia and random real X.
    Returns (system, hamiltonian, trial, walkers, alpha, beta).
    """
    from ipie.systems import Generic
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
    from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

    rng = np.random.RandomState(seed)
    alpha = (rng.randn(N) + 1j * rng.randn(N)).astype(np.complex128)
    # Real beta: this is the canonical-convention beta. The trial code
    # internally rescales it to <X> via beta_shift = beta * sqrt(2/(mw)).
    beta = 0.4 * rng.randn(N).astype(np.complex128)

    sys_ = Generic((1, 0))
    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=N, pbc=True)
    ham.build()

    # Walker: random phia (single column) and random real X.
    phia_w = (rng.randn(N, 1) + 1j * rng.randn(N, 1)).astype(np.complex128)
    X_w = 0.5 * rng.randn(N).astype(np.float64)

    # Trial: ToyozawaTrial expects wavefunction = column_stack([beta, alpha]).
    wfn_T = np.column_stack([beta[:, None], alpha[:, None]])
    trial = ToyozawaTrial(wavefunction=wfn_T, w0=w0, num_elec=(1, 0),
                          num_basis=N, K=K)

    # Walker: build EPhWalkers with TWO walkers (the class squeezes
    # nwalkers=1 and breaks). We only use walker index 0 in tests.
    init_walker = np.column_stack([np.zeros(N)[:, None], phia_w])
    walkers = EPhWalkers(init_walker, nup=1, ndown=0, nbasis=N, nwalkers=2,
                         verbose=False)
    walkers.build(trial)
    # Override walker 0 with the random (phia_w, X_w). Walker 1 keeps
    # whatever the constructor produced — irrelevant for the test.
    walkers.phonon_disp[0] = X_w.astype(np.complex128)
    walkers.phia[0] = phia_w.copy()

    return sys_, ham, trial, walkers, alpha, beta


@pytest.mark.unit
@pytest.mark.parametrize("seed,N,K", _params())
def test_calc_overlap_matches_fock(seed, N, K):
    pytest.importorskip("scipy.sparse")
    sys_, ham, trial, walkers, alpha, beta = _make_random_trial_and_walker(
        seed, N, K
    )
    M = M_TRUNC

    # Code's overlap (single walker)
    ovlp_code = complex(trial.calc_overlap(walkers)[0])

    # Fock-basis overlap. The trial's beta_shift internally is
    # beta_shift = wavefunction[:, 0] * sqrt(2/(mw)) = beta * sqrt(2/(mw)).
    # In our convention, beta is the canonical coherent-state amplitude
    # used by _build_K_state (which calls _coh(beta) -> exp(beta a^dag - ...)).
    psi_K = _build_K_state(alpha, beta, K, N, M)
    walker_state = _walker_state(walkers.phia[0], walkers.phonon_disp[0],
                                 N, M, mw=ham.m * ham.w0)
    ovlp_fock = complex(np.vdot(psi_K, walker_state))

    # The QMC code drops the (mw/pi)^{N/4} prefactor of <coh|X>, which
    # appears once per *site* in the phonon overlap. Strip it before
    # comparing.
    prefactor = (ham.m * ham.w0 / np.pi) ** (N / 4)
    ovlp_fock_stripped = ovlp_fock / prefactor

    rel = abs(ovlp_code - ovlp_fock_stripped) / max(abs(ovlp_fock_stripped), 1e-12)
    assert rel < 1e-5, (
        f"calc_overlap mismatch at (seed={seed}, N={N}, K={K:.4f}):\n"
        f"  code       = {ovlp_code}\n"
        f"  fock (raw) = {ovlp_fock}\n"
        f"  fock - pre = {ovlp_fock_stripped}\n"
        f"  rel        = {rel:.3e}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("seed,N,K", _params())
def test_calc_greens_function_matches_fock(seed, N, K):
    pytest.importorskip("scipy.sparse")
    sys_, ham, trial, walkers, alpha, beta = _make_random_trial_and_walker(
        seed, N, K
    )
    M = M_TRUNC

    # `calc_greens_function` defensively refreshes walkers.ovlp_perm
    # internally, so we don't need to call calc_overlap here.
    Ga_code, _Gb_code = trial.calc_greens_function(walkers)
    Ga_code = np.asarray(Ga_code[0])  # (N, N)

    # Fock-basis reference: G[p, q] = <Psi_T | c_q^dag c_p | walker>
    #                              / <Psi_T | walker>
    psi_K = _build_K_state(alpha, beta, K, N, M)
    walker_state = _walker_state(walkers.phia[0], walkers.phonon_disp[0],
                                 N, M, mw=ham.m * ham.w0)
    denom = np.vdot(psi_K, walker_state)
    G_fock = np.zeros((N, N), dtype=np.complex128)
    for p in range(N):
        for q in range(N):
            cdag_c = _build_cdag_c(p, q, N, M)
            G_fock[p, q] = np.vdot(psi_K, cdag_c @ walker_state) / denom

    rel = (np.linalg.norm(Ga_code - G_fock)
           / max(np.linalg.norm(G_fock), 1e-12))
    abs_max = np.max(np.abs(Ga_code - G_fock))
    assert rel < 1e-5 and abs_max < 1e-5, (
        f"calc_greens_function mismatch at (seed={seed}, N={N}, K={K:.4f}):\n"
        f"  rel  = {rel:.3e}\n"
        f"  amax = {abs_max:.3e}"
    )

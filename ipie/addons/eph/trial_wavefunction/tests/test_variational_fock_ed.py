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
Fock-basis ED check for the K-projected Toyozawa trial.

Builds the trial state |Psi_K> *explicitly* in the basis
    {|i_el> ⊗ |n_0, n_1, ..., n_{N-1}>}
by expanding each on-site coherent state |coh(beta_k)> in the truncated
number basis, taking the tensor product over phonon sites, and applying
the K-projector
    |Psi_K> = sum_m e^{i K m} T_m ( |alpha> ⊗ |coh(beta)> )
where T_m translates BOTH the electron and the phonon-site labels.

Then E_K = <Psi_K | H | Psi_K> / <Psi_K | Psi_K> is computed by
sandwiching a sparse Hamiltonian in this basis. We compare against the
analytic ToyozawaVariational.objective_function.

Convention
----------
After the fix to `circ_perm_1D` (which now uses np.roll(sites, +shift)),
`perms[ip]` corresponds to T_{+ip}, and combined with
`kcoeffs[ip] = exp(+1j K ip)` the trial labelled "K" really is the
physical-momentum-K state
    |Psi_K> = sum_n e^{i K n} T_n |alpha, beta>.
This test verifies that by comparing the analytic objective to the
energy of |Psi_K> built explicitly in the truncated Fock basis at the
*same* K.

Sign convention for H_eph: matches the variational code, i.e.
    H_eph = +g sum_i n_i (b_i + b_i^dag).
This differs in sign from HolsteinModel's docstring; the two are unitary
equivalents (X -> -X), so the *energy* is the same in either convention,
but to get bit-for-bit agreement we use the code's convention.
"""

import math

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# Single-site bosonic operators (truncated)
# ----------------------------------------------------------------------------

def _boson_ops(M):
    import scipy.sparse as sp
    diag = np.sqrt(np.arange(1, M))
    b = sp.diags(diag, offsets=1, format="csr", shape=(M, M))
    bd = sp.diags(diag, offsets=-1, format="csr", shape=(M, M))
    n = sp.diags(np.arange(M), offsets=0, format="csr", shape=(M, M))
    return b, bd, n


def _site_op_in_chain(op, k, N, M):
    import scipy.sparse as sp
    out = None
    for s in range(N):
        factor = op if s == k else sp.identity(M, format="csr")
        out = factor if out is None else sp.kron(out, factor, format="csr")
    return out


def _build_holstein_H(N, M, t, g, w0, pbc=True):
    import scipy.sparse as sp
    dim_ph = M ** N
    I_el = sp.identity(N, format="csr")
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
        H_phon_only = H_phon_only + _site_op_in_chain(n1, k, N, M)
    H_phon_only *= w0
    H_phon = sp.kron(I_el, H_phon_only, format="csr")

    H_eph = sp.csr_matrix((N * dim_ph, N * dim_ph))
    for k in range(N):
        proj = sp.csr_matrix(([1.0], ([k], [k])), shape=(N, N))
        x_k = _site_op_in_chain(b1 + bd1, k, N, M)
        H_eph = H_eph + g * sp.kron(proj, x_k, format="csr")

    return (H_kin + H_phon + H_eph).tocsr()


# ----------------------------------------------------------------------------
# K-projected trial in (lattice ⊗ Fock) basis
# ----------------------------------------------------------------------------

def _coh(beta, M):
    v = np.zeros(M, dtype=np.complex128)
    log_norm = -0.5 * np.abs(beta) ** 2
    for n in range(M):
        if beta == 0:
            v[n] = 1.0 if n == 0 else 0.0
        else:
            log_amp = n * np.log(beta) - 0.5 * math.lgamma(n + 1)
            v[n] = np.exp(log_norm + log_amp)
    return v


def _phonon_product(beta, M):
    out = np.array([1.0 + 0j])
    for b in beta:
        out = np.kron(out, _coh(b, M))
    return out


def _T_m(state, m, N, M):
    """Apply T_m to a (lattice ⊗ Fock) state.

    Phonon part: new site k corresponds to old site (k - m) mod N (rename
    site labels). Electron part: roll by +m (electron at site i becomes
    at site i+m).
    """
    arr = state.reshape((N,) + (M,) * N)
    arr = np.roll(arr, shift=m, axis=0)
    ph_axes = list(range(1, N + 1))
    new_ph_order = [ph_axes[(k - m) % N] for k in range(N)]
    arr = arr.transpose([0] + new_ph_order)
    return arr.reshape(state.shape)


def _build_K_state(alpha, beta, K, N, M):
    psi0 = np.kron(alpha.astype(np.complex128), _phonon_product(beta, M))
    out = np.zeros_like(psi0)
    for m in range(N):
        out = out + np.exp(1j * K * m) * _T_m(psi0, m, N, M)
    return out


def _fock_energy(alpha, beta, K, N, M, t, g, w0, pbc=True):
    psi_K = _build_K_state(alpha, beta, K, N, M)
    H = _build_holstein_H(N, M, t, g, w0, pbc)
    num = np.vdot(psi_K, H @ psi_K)
    den = np.vdot(psi_K, psi_K)
    return (num / den).real


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

SEEDS = [0, 1]
NSITES = [3, 4]
M_TRUNC = 10  # phonon occupation cutoff per site


def _params():
    """Test only at BZ-allowed momenta K = 2 pi n / N. Outside the BZ,
    the K-projection sum doesn't collapse to a translation eigenstate,
    so the analytic single-loop formula and the explicit Fock-basis
    construction would differ by finite-N edge terms unrelated to the
    physics."""
    out = []
    for seed in SEEDS:
        for N in NSITES:
            for n in range(N):
                out.append((seed, N, 2 * np.pi * n / N))
    return out


@pytest.mark.unit
@pytest.mark.parametrize("seed,N,K", _params())
def test_toyozawa_objective_matches_fock_ED(seed, N, K):
    """Trial energy from the analytic objective_function matches the
    energy of the same Toyozawa state built explicitly in Fock basis,
    up to the phonon-truncation error.

    Both sides evaluate at the same physical K. If the convention in
    `circ_perm_1D` is ever flipped back, this test will fail at
    K not in {0, pi}, where E(K) != E(-K) for an asymmetric (alpha, beta).
    """
    pytest.importorskip("scipy.sparse")
    from ipie.systems import Generic
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.variational.toyozawa import (
        ToyozawaVariational,
    )

    np.random.seed(seed)
    alpha = np.random.randn(N) + 1j * np.random.randn(N)
    # Keep |beta| ~ 0.5 so M=10 truncation is essentially exact.
    beta = (np.random.randn(N) + 1j * np.random.randn(N)) * 0.4

    g, t, w0 = 0.5, 1.0, 1.0
    sys_ = Generic((1, 0))
    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=N, pbc=True)
    ham.build()

    var = ToyozawaVariational(beta, alpha[:, None], ham, sys_, K=K, cplx=True)
    e_anal = var.objective_function(var.pack_x())

    e_fock = _fock_energy(alpha, beta, K, N, M_TRUNC, t, g, w0, pbc=True)

    assert abs(e_anal - e_fock) < 1e-5, (
        f"Toyozawa trial energy disagrees with Fock-basis ED at "
        f"K={K}, N={N}, seed={seed}: analytic={e_anal:.10g}, "
        f"fock={e_fock:.10g}, diff={abs(e_anal - e_fock):.3e}."
    )

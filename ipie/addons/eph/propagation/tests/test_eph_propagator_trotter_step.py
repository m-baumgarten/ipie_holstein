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
Statistical validation of one full Trotter step of `EPhPropagatorFree`
against an explicit Fock-basis matrix exponential of the same Trotter
factorization.

The DMC consistency identity
----------------------------
For an unbiased DMC sampler of the imaginary-time evolution, given a
walker initially at (phi, X) with weight 1,
    E[ w_final * <Psi_T | walker_final> ]
        = <Psi_T | U_Trotter | phi ⊗ X >,
where
    U_Trotter = e^{-Δτ H_ph/2} e^{-Δτ T/2} e^{-Δτ H_eph} e^{-Δτ T/2} e^{-Δτ H_ph/2}.

We test the ratio
    R = E[w_final <Psi_T|w_final>] / <Psi_T | walker_init>
against
    R_Fock = <Psi_T | U_Trotter | walker_init>_Fock / <Psi_T | walker_init>_Fock,
which removes the (mω/π)^{N/4} prefactor convention difference between
QMC and Fock representations.

What this test does NOT cover
-----------------------------
The cos-phase clamp in `EPhPropagatorFree.update_weight` (the
fixed-phase approximation) is intentionally bypassed: we call the
phonon/electron/phonon pieces of `EPhPropagatorFree` directly and skip
`update_weight`. The clamp is a separate algorithmic approximation
which gets validated by long-time imaginary-time evolution and the
free-projection comparison, not by a single-step Fock-ED check.
"""

import math

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# Fock-basis pieces (re-used machinery from the trial Fock-ED test)
# ----------------------------------------------------------------------------

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


def _build_H_pieces(N, M, t, g, w0):
    """Returns (H_T, H_eph, H_ph) as dense matrices on (N) ⊗ (M^N) basis,
    each in the SAME sign convention as the QMC code.

    H_T:  -t Σ_<ij> c_i^† c_j  (electron hopping, real symmetric)
    H_eph: +g Σ_i n_i (b_i + b_i^†)
    H_ph:  +w0 Σ_i b_i^† b_i  (free harmonic; no zero-point shift here
                                — the QMC weight-update zero-point factor
                                handles that separately)
    """
    import scipy.sparse as sp
    dim_ph = M ** N
    I_ph = sp.identity(dim_ph, format="csr")

    # Hopping (electron)
    rows, cols, data = [], [], []
    for i in range(N - 1):
        rows += [i, i + 1]; cols += [i + 1, i]; data += [-t, -t]
    if N > 1:
        rows += [0, N - 1]; cols += [N - 1, 0]; data += [-t, -t]
    T_el = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    H_T = sp.kron(T_el, I_ph, format="csr")

    # Phonon harmonic
    b1, bd1, n1 = _boson_ops(M)
    H_phon_only = sp.csr_matrix((dim_ph, dim_ph))
    for k in range(N):
        H_phon_only = H_phon_only + _site_op(n1, k, N, M)
    H_phon_only *= w0
    H_ph = sp.kron(sp.identity(N, format="csr"), H_phon_only, format="csr")

    # Electron-phonon
    H_eph = sp.csr_matrix((N * dim_ph, N * dim_ph))
    for k in range(N):
        proj = sp.csr_matrix(([1.0], ([k], [k])), shape=(N, N))
        x_k = _site_op(b1 + bd1, k, N, M)
        H_eph = H_eph + g * sp.kron(proj, x_k, format="csr")

    return H_T.toarray(), H_eph.toarray(), H_ph.toarray()


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


def _ho_position_state(X, N, M, mw):
    out = np.array([1.0 + 0j])
    factor = (mw / np.pi) ** 0.25
    sqrt_mw = np.sqrt(mw)
    for k in range(N):
        y = sqrt_mw * X[k]
        H = np.zeros(M, dtype=np.float64)
        H[0] = 1.0
        if M > 1:
            H[1] = 2 * y
            for n in range(1, M - 1):
                H[n + 1] = 2 * y * H[n] - 2 * n * H[n - 1]
        psi_n = np.zeros(M, dtype=np.complex128)
        norm_log = np.log(factor) - 0.5 * mw * X[k] ** 2
        for n in range(M):
            log_amp = -0.5 * (n * np.log(2.0) + math.lgamma(n + 1))
            psi_n[n] = H[n] * np.exp(norm_log + log_amp)
        out = np.kron(out, psi_n)
    return out


def _walker_state(phia, X, N, M, mw):
    el = phia[:, 0].astype(np.complex128)
    ph = _ho_position_state(X.real, N, M, mw=mw)
    return np.kron(el, ph)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("N,M,dt,g,n_walkers,seed", [
    (3, 6, 0.01, 0.3, 80000, 0),
])
def test_trotter_step_matches_fock_average(N, M, dt, g, n_walkers, seed):
    """E[w_final * <Psi_T|walker_final>] = <Psi_T | U_Trotter | walker_init>
    in the limit of infinite walkers, for small enough Δτ. Use ratio to
    cancel the (mω/π)^{N/4} prefactor convention difference."""
    pytest.importorskip("scipy.linalg")
    import scipy.linalg as la
    from ipie.systems import Generic                                  # noqa: F401
    from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
    from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
    from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
    from ipie.addons.eph.propagation.eph_propagator import EPhPropagatorFree

    K = 0.0
    t, w0 = 1.0, 1.0
    rng = np.random.RandomState(seed)
    # Real trial parameters, real initial walker -> all overlaps real
    # positive, cos-phase clamp is a no-op.
    alpha = rng.randn(N).astype(np.complex128)
    beta = (0.3 * rng.randn(N)).astype(np.complex128)
    phia0 = rng.randn(N, 1).astype(np.complex128)
    X0 = (0.4 * rng.randn(N)).astype(np.float64)

    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=N, pbc=True)
    ham.build()

    wfn_T = np.column_stack([beta[:, None], alpha[:, None]])
    trial = ToyozawaTrial(wavefunction=wfn_T, w0=w0, num_elec=(1, 0),
                          num_basis=N, K=K)

    # --- Fock-basis reference ---
    H_T, H_eph, H_ph = _build_H_pieces(N, M, t, g, w0)
    expT_half = la.expm(-0.5 * dt * H_T)
    expEph    = la.expm(-1.0 * dt * H_eph)
    expPh_half = la.expm(-0.5 * dt * H_ph)
    # Symmetric Trotter: same factorization the QMC code uses.
    U = expPh_half @ expT_half @ expEph @ expT_half @ expPh_half

    # The QMC's effective Hamiltonian is H_QMC = mω²X²/2 + p²/(2m) - w0/2
    # per mode. In Fock-basis units that's H_ph = w0 b†b - N w0/2. The
    # QMC propagate_phonons multiplies weight by exp(dt × N w0/2) per
    # step, which exactly cancels the -N w0/2 shift, so the *net*
    # effective propagator in walker space is e^{-dt × w0 b†b}.
    # Our `_build_H_pieces` already uses H_ph = w0 b†b (no shift), so
    # U_Trotter built above is precisely the propagator the QMC
    # implements: NO extra zpe factor needed.
    psi_T_fock = _build_K_state(alpha, beta, K, N, M)
    walker_init_fock = _walker_state(phia0, X0, N, M, mw=ham.m * ham.w0)

    num_fock = np.vdot(psi_T_fock, U @ walker_init_fock)
    den_fock = np.vdot(psi_T_fock, walker_init_fock)
    ratio_fock = complex(num_fock / den_fock)

    # --- QMC: many walkers, one step each, average w * <Psi_T|.> ---
    init_walker = np.column_stack([np.zeros(N)[:, None], phia0])
    walkers = EPhWalkers(init_walker, nup=1, ndown=0, nbasis=N,
                         nwalkers=n_walkers, verbose=False)
    walkers.build(trial)
    # Reset all walkers to the SAME initial (phia0, X0, w=1).
    walkers.phonon_disp[:] = X0[None, :].astype(np.complex128)
    walkers.phia[:] = phia0[None, :, :]
    walkers.weight[:] = 1.0

    # Sanity: <Psi_T | walker_init>_code should be real positive.
    ovlp_init_code = trial.calc_overlap(walkers)[0]
    assert abs(ovlp_init_code.imag) < 1e-10, \
        f"Initial overlap not real: {ovlp_init_code}"

    prop = EPhPropagatorFree(time_step=dt)
    prop.build(ham, trial=trial, walkers=walkers)

    # Apply the Trotter pieces directly (skipping update_weight, which
    # is the fixed-phase cos-clamp — orthogonal to what we're testing).
    np.random.seed(2024)
    prop.propagate_phonons(walkers, ham, trial)
    prop.propagate_electron(walkers, ham, trial)
    prop.propagate_phonons(walkers, ham, trial)

    # E[w_final * <Psi_T | walker_final>] / <Psi_T | walker_init>
    ovlp_final = trial.calc_overlap(walkers)
    estimator_per_walker = walkers.weight * ovlp_final
    mean = np.mean(estimator_per_walker)
    sem = np.std(estimator_per_walker, ddof=1) / np.sqrt(n_walkers)

    ratio_qmc = mean / ovlp_init_code
    ratio_qmc_err = abs(sem / ovlp_init_code)

    # The Trotter error is O(Δτ^3) ~ (0.01)^3 = 1e-6, plus the explicit
    # cos-clamp side effects (none here at K=0 real). Allow 4 sigma plus
    # Trotter error tolerance.
    diff = abs(ratio_qmc - ratio_fock)
    tol = 4 * ratio_qmc_err + 1e-4

    print(f"\n  ratio_fock = {ratio_fock}")
    print(f"  ratio_qmc  = {ratio_qmc}  +/- {ratio_qmc_err:.3e}")
    print(f"  |diff|     = {diff:.3e}  (tol {tol:.3e})")

    assert diff < tol, (
        f"Trotter step QMC vs Fock disagree:\n"
        f"  ratio_fock = {ratio_fock}\n"
        f"  ratio_qmc  = {ratio_qmc}  +/- {ratio_qmc_err:.3e}\n"
        f"  |diff|     = {diff:.3e}\n"
        f"  tol        = {tol:.3e}\n"
    )

"""Exact Fock-space imaginary-time reference for the coherent-state FP QMC.

Builds the Holstein Hamiltonian in a truncated Fock basis (electron site (x) 4
phonon modes, each 0..nmax), constructs the *same* K Toyozawa coherent-state
trial as an explicit Fock vector, and computes the deterministic mixed
imaginary-time curve

    E(tau) = <Psi_T| H e^{-tau H} |Psi_T> / <Psi_T| e^{-tau H} |Psi_T>

via scipy expm_multiply. This is exactly what the stochastic FP samples (no
Trotter error: O(dtau^2) ~ 1e-9 here, negligible). As tau->inf, E(tau) -> the
lowest eigenvalue in the sector that Psi_T overlaps (here K=pi/2).

Usage: python fock_imag_time.py --w0 1 --g 1 --K 1.5707963 --tag w0_1.0_Kpi2 \
           --nmax 10 --tau-max 8 --num 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel  # noqa: E402


def single_mode_ops(nmax):
    M = nmax + 1
    n = np.arange(M)
    a = sp.diags(np.sqrt(n[1:]), 1, format="csr")        # annihilation
    num = sp.diags(n, 0, format="csr")                   # number
    x = a + a.conj().T                                   # b + b^dagger
    return a, num, x, M


def build_hamiltonian(ham, nmax):
    """Sparse Holstein H on (4 electron sites) x (4 modes, 0..nmax)."""
    ns = int(ham.N)
    T = np.asarray(ham.T[0], dtype=np.complex128)        # electronic hopping
    g_tensor = np.asarray(ham.g_tensor, dtype=np.complex128)
    w0 = float(np.asarray(ham.w0).reshape(-1)[0])
    _, num, x, M = single_mode_ops(nmax)
    Iph_1 = sp.identity(M, format="csr")
    Iel = sp.identity(ns, format="csr")

    def mode_op(op, k):
        """op acting on mode k, identity on the other ns-1 modes."""
        mats = [Iph_1] * ns
        mats[k] = op
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format="csr")
        return out

    Iph = sp.identity(M ** ns, format="csr")
    # electron hopping
    H = sp.kron(T, Iph, format="csr")
    # phonon energy w0 * sum_k n_k
    Hph = mode_op(num, 0)
    for k in range(1, ns):
        Hph = Hph + mode_op(num, k)
    H = H + sp.kron(Iel, w0 * Hph, format="csr")
    # e-ph: sum_{ijk} g_tensor[i,j,k] c_i^dag c_j (b_k + b_k^dag)
    for k in range(ns):
        gk = g_tensor[:, :, k]                            # ns x ns electronic matrix
        if np.allclose(gk, 0.0):
            continue
        H = H + sp.kron(sp.csr_matrix(gk), mode_op(x, k), format="csr")
    return H.tocsr(), M


def coherent_vec(beta, nmax):
    """Normalized single-mode coherent state in the 0..nmax Fock basis."""
    n = np.arange(nmax + 1)
    logc = -0.5 * np.abs(beta) ** 2 + n * np.log(beta + 0j) - 0.5 * _logfact(n)
    v = np.exp(logc)
    return v / np.linalg.norm(v)


def _logfact(n):
    from scipy.special import gammaln
    return gammaln(n + 1.0)


def build_trial(beta, psi, K, nmax, ns):
    """Toyozawa K-projected trial |Psi_T> = sum_j e^{iKj} T_j(|phi> x |beta>)."""
    M = nmax + 1
    psiT = np.zeros(ns * M ** ns, dtype=np.complex128)
    for j in range(ns):
        phi_j = np.roll(psi, j)                  # electron orbital translated by j
        beta_j = np.roll(beta, j)                # phonon displacements translated by j
        # phonon product coherent state (kron over modes)
        ph = coherent_vec(beta_j[0], nmax)
        for i in range(1, ns):
            ph = np.kron(ph, coherent_vec(beta_j[i], nmax))
        comp = np.kron(phi_j, ph)                # electron (x) phonon
        psiT += np.exp(1j * K * j) * comp
    return psiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w0", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=np.pi / 2)
    ap.add_argument("--tag", default="w0_1.0_Kpi2")
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--tau-max", type=float, default=8.0)
    ap.add_argument("--num", type=int, default=200)
    a = ap.parse_args()

    ham = HolsteinModel(g=a.g, t=a.t, w0=a.w0, nsites=4, pbc=True)
    ham.build()
    ns = int(ham.N)

    wf = np.load(HERE / f"trial_{a.tag}.npz")["wavefunction"]
    beta, psi = wf[:, 0].astype(np.complex128), wf[:, 1].astype(np.complex128)

    H, M = build_hamiltonian(ham, a.nmax)
    psiT = build_trial(beta, psi, a.K, a.nmax, ns)
    dim = H.shape[0]

    # trial energy (consistency with the QMC trial)
    HpsiT = H @ psiT
    e_trial = (psiT.conj() @ HpsiT) / (psiT.conj() @ psiT)

    # imaginary-time mixed estimator on a tau grid
    series = expm_multiply(-H, psiT, start=0.0, stop=a.tau_max, num=a.num, endpoint=True)
    taus = np.linspace(0.0, a.tau_max, a.num)
    E = np.empty(a.num)
    norm = np.empty(a.num)
    for i in range(a.num):
        v = series[i]
        den = psiT.conj() @ v
        numr = psiT.conj() @ (H @ v)
        E[i] = (numr / den).real
        norm[i] = np.abs(den)

    out = HERE / f"fock_imag_time_{a.tag}_nmax{a.nmax}.npz"
    np.savez(out, tau=taus, E=E, norm=norm, e_trial=np.real(e_trial),
             e_inf=E[-1], nmax=a.nmax, dim=dim)
    print(f"nmax={a.nmax} dim={dim}")
    print(f"trial energy (Fock)  = {e_trial.real:.6f}   (QMC trial ~ -1.7181)")
    for tt in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, a.tau_max]:
        i = int(np.argmin(np.abs(taus - tt)))
        print(f"  E(tau={taus[i]:.2f}) = {E[i]:.6f}")
    print(f"E(tau_max={a.tau_max}) = {E[-1]:.6f}   (ED K=pi/2 ~ -1.771432)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Variationally optimize the Toyozawa coherent-state trial for the q stress test.

Saves a (nsites, 2) wavefunction [beta | orbital] consumable by fixture.build_workflow,
and verifies the loaded trial's variational energy matches the optimizer.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipie.systems.generic import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import ToyozawaVariational
from ipie.addons.eph.trial_wavefunction.toyozawa_cs_unnormalized import (
    ToyozawaTrialUnnormalizedCoherentState,
)

import argparse

NSITES, T, SEED = 4, 1.0, 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w0", type=float, default=0.1)
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--K", type=float, default=0.0, help="crystal momentum (e.g. pi/2)")
    ap.add_argument("--tag", default="w0_0.1")
    a = ap.parse_args()
    W0, G, K = a.w0, a.g, a.K
    OUT = Path(__file__).resolve().parent / f"trial_{a.tag}.npz"
    rng = np.random.default_rng(SEED)
    sys_ = Generic((1, 0))
    ham = HolsteinModel(g=G, t=T, w0=W0, nsites=NSITES, pbc=True)
    ham.build()

    # Physically motivated seed: localized polaron + matching displacement.
    alpha = np.exp(-0.5 * np.arange(NSITES)) + 0.1 * rng.standard_normal(NSITES)
    alpha = alpha.astype(np.complex128)
    if abs(K) > 1e-12:
        # Seed a Bloch-like phase so the K!=0 optimizer doesn't start at a real saddle.
        alpha *= np.exp(1j * K * np.arange(NSITES))
        alpha += 0.05 * (rng.standard_normal(NSITES) + 1j * rng.standard_normal(NSITES))
    alpha /= np.linalg.norm(alpha)
    beta = (-G / W0) * np.abs(alpha) ** 2  # mean-field-ish displacement seed
    beta = beta.astype(np.complex128)

    var = ToyozawaVariational(beta, alpha[:, None], ham, sys_, K=K, cplx=True)
    etrial, beta_opt, psi_opt = var.run()
    beta_opt = np.squeeze(np.asarray(beta_opt)).astype(np.complex128)
    psi_opt = np.asarray(psi_opt).reshape(NSITES).astype(np.complex128)

    wavefunction = np.column_stack([beta_opt, psi_opt]).astype(np.complex128)
    np.savez(OUT, wavefunction=wavefunction, etrial=np.asarray(etrial))

    # Verify the FP trial reproduces the variational energy.
    trial = ToyozawaTrialUnnormalizedCoherentState(
        wavefunction=wavefunction, w0=ham.w0, num_elec=(1, 0), num_basis=NSITES, K=K
    )
    e_fp = float(np.real(trial.calc_energy(ham)[0]))
    print(f"optimizer etrial   = {etrial:.10f}")
    print(f"FP trial calc_energy = {e_fp:.10f}")
    print(f"|diff|             = {abs(e_fp - etrial):.3e}")
    print(f"saved -> {OUT}")
    print("beta_opt =", np.round(beta_opt, 4))
    print("psi_opt  =", np.round(psi_opt, 4))


if __name__ == "__main__":
    main()

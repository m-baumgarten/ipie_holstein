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
"""Site-basis JAX optimizer for a one-electron Toyozawa Holstein trial."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel

jax.config.update("jax_enable_x64", True)


@dataclass
class ToyozawaSiteOptimizationResult:
    energy: float
    initial_energy: float
    beta: np.ndarray
    orbital: np.ndarray
    x: np.ndarray
    success: bool
    message: str
    nit: int
    nfev: int

    @property
    def wavefunction(self) -> np.ndarray:
        return np.column_stack([self.beta, self.orbital])


def cyclic_permutations(nsites: int) -> np.ndarray:
    sites = np.arange(nsites, dtype=np.int32)
    return np.vstack([np.roll(sites, shift) for shift in range(nsites)])


def pack_toyozawa_params(beta: np.ndarray, orbital: np.ndarray, complex_params: bool) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.complex128).reshape(-1)
    orbital = np.asarray(orbital, dtype=np.complex128).reshape(-1)
    if complex_params:
        return np.hstack([beta.real, beta.imag, orbital.real, orbital.imag]).astype(np.float64)
    return np.hstack([beta.real, orbital.real]).astype(np.float64)


def unpack_toyozawa_params(x, nsites: int, complex_params: bool):
    if complex_params:
        beta = x[:nsites] + 1j * x[nsites : 2 * nsites]
        orbital = x[2 * nsites : 3 * nsites] + 1j * x[3 * nsites : 4 * nsites]
    else:
        beta = x[:nsites]
        orbital = x[nsites : 2 * nsites]
    norm = jnp.sqrt(jnp.vdot(orbital, orbital).real)
    orbital = orbital / jnp.maximum(norm, 1.0e-14)
    return beta, orbital


def toyozawa_site_energy(
    x,
    T,
    g_tensor,
    w0,
    perms,
    kcoeffs,
    complex_params: bool = False,
    denom_floor: float = 1.0e-12,
):
    """Evaluate the site-basis Toyozawa Rayleigh quotient.

    The optimized seed is |phi> tensor |beta>.  The projected trial is
    sum_R exp(i K R) T_R |phi,beta>.  This function evaluates the double
    sum after translational reduction to a single relative translation R.
    """
    nsites = T.shape[0]
    beta, orbital = unpack_toyozawa_params(x, nsites, complex_params)

    numer = 0.0 + 0.0j
    denom = 0.0 + 0.0j
    for perm, coeff in zip(perms, kcoeffs):
        beta_r = beta[perm]
        orbital_r = orbital[perm]

        ph_ovlp = jnp.exp(
            jnp.sum(-0.5 * (jnp.abs(beta) ** 2 + jnp.abs(beta_r) ** 2) + beta.conj() * beta_r)
        )
        el_ovlp = jnp.vdot(orbital, orbital_r)

        kinetic = jnp.vdot(orbital, T @ orbital_r)
        el_ph = jnp.einsum(
            "i,j,ijk,k->",
            orbital.conj(),
            orbital_r,
            g_tensor,
            beta.conj() + beta_r,
        )
        phonon = el_ovlp * w0 * jnp.vdot(beta, beta_r)
        matrix_element = ph_ovlp * (kinetic + el_ph + phonon)

        numer = numer + coeff * matrix_element
        denom = denom + coeff * ph_ovlp * el_ovlp

    denom_abs = jnp.abs(denom)
    denom_safe = jnp.where(denom_abs > denom_floor, denom, denom_floor + 0.0j)
    energy = jnp.real(numer / denom_safe)
    penalty = jnp.where(denom_abs > denom_floor, 0.0, 1.0e6 * (denom_floor - denom_abs))
    return energy + penalty


def _normalise_orbital(orbital: np.ndarray) -> np.ndarray:
    orbital = np.asarray(orbital, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(orbital)
    if norm == 0.0:
        raise ValueError("Initial Toyozawa electron orbital has zero norm.")
    return orbital / norm


def optimise_toyozawa_site_jax(
    hamiltonian: HolsteinModel,
    beta_init: np.ndarray,
    orbital_init: np.ndarray,
    K: float = 0.0,
    complex_params: bool = False,
    maxiter: int = 1000,
    gtol: float = 1.0e-9,
    ftol: float = 1.0e-12,
) -> ToyozawaSiteOptimizationResult:
    """Optimize a one-electron Toyozawa seed in the site basis."""
    if hamiltonian.N != hamiltonian.nsites[0]:
        raise NotImplementedError("site_toyozawa currently supports 1D Holstein chains.")

    beta_init = np.asarray(beta_init, dtype=np.complex128).reshape(hamiltonian.N)
    orbital_init = _normalise_orbital(orbital_init)

    perms_np = cyclic_permutations(hamiltonian.N)
    kcoeffs_np = np.exp(1j * K * np.arange(hamiltonian.N))

    T = jnp.asarray(hamiltonian.T[0], dtype=jnp.complex128)
    g_tensor = jnp.asarray(hamiltonian.g_tensor, dtype=jnp.complex128)
    perms = jnp.asarray(perms_np)
    kcoeffs = jnp.asarray(kcoeffs_np, dtype=jnp.complex128)

    objective = jax.jit(
        jax.value_and_grad(
            lambda params: toyozawa_site_energy(
                params,
                T,
                g_tensor,
                hamiltonian.w0,
                perms,
                kcoeffs,
                complex_params=complex_params,
            )
        )
    )

    def value_and_grad(params):
        value, grad = objective(jnp.asarray(params, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    x0 = pack_toyozawa_params(beta_init, orbital_init, complex_params)
    initial_energy, _ = value_and_grad(x0)
    result = minimize(
        value_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "gtol": gtol,
            "ftol": ftol,
            "maxls": 50,
        },
    )

    beta, orbital = unpack_toyozawa_params(
        jnp.asarray(result.x, dtype=jnp.float64),
        hamiltonian.N,
        complex_params,
    )
    return ToyozawaSiteOptimizationResult(
        energy=float(result.fun),
        initial_energy=initial_energy,
        beta=np.asarray(beta, dtype=np.complex128),
        orbital=np.asarray(orbital, dtype=np.complex128)[:, None],
        x=np.asarray(result.x, dtype=np.float64),
        success=bool(result.success),
        message=str(result.message),
        nit=int(result.nit),
        nfev=int(result.nfev),
    )


def optimise_toyozawa_site_jax_restarts(
    hamiltonian: HolsteinModel,
    beta_init: np.ndarray,
    orbital_init: np.ndarray,
    K: float = 0.0,
    complex_params: bool = False,
    maxiter: int = 1000,
    nstarts: int = 1,
    seed: int = 7,
    noise: float = 0.05,
) -> ToyozawaSiteOptimizationResult:
    rng = np.random.default_rng(seed)
    beta_init = np.asarray(beta_init, dtype=np.complex128).reshape(hamiltonian.N)
    orbital_init = _normalise_orbital(orbital_init).reshape(hamiltonian.N)

    starts = [(beta_init, orbital_init)]
    for _ in range(max(0, nstarts - 1)):
        beta = beta_init + noise * rng.normal(size=hamiltonian.N)
        orbital = orbital_init + noise * rng.normal(size=hamiltonian.N)
        if complex_params:
            beta = beta + 1j * noise * rng.normal(size=hamiltonian.N)
            orbital = orbital + 1j * noise * rng.normal(size=hamiltonian.N)
        starts.append((beta, orbital))

    best: Optional[ToyozawaSiteOptimizationResult] = None
    for beta, orbital in starts:
        result = optimise_toyozawa_site_jax(
            hamiltonian,
            beta,
            orbital,
            K=K,
            complex_params=complex_params,
            maxiter=maxiter,
        )
        if best is None or result.energy < best.energy:
            best = result
    return best


optimize_toyozawa_site_jax = optimise_toyozawa_site_jax
optimize_toyozawa_site_jax_restarts = optimise_toyozawa_site_jax_restarts

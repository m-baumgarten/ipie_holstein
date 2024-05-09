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

import numpy as np
import plum
from typing import Tuple
from scipy.optimize import minimize, basinhopping
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
import jax
import jax.numpy as npj

from abc import abstractmethod, ABCMeta


class Variational(metaclass=ABCMeta):
    def __init__(
        self,
        shift_init: np.ndarray,
        electron_init: np.ndarray,
        hamiltonian,
        system,
        cplx: bool = True,
    ):
        self.psi = electron_init.T.ravel()
        self.shift = shift_init.T.ravel()

        self.ham = hamiltonian
        self.sys = system

        if cplx:
            self.pack_x = self.pack_x_complex
            self.unpack_x = self.unpack_x_complex
        else:
            self.pack_x = self.pack_x_real
            self.unpack_x = self.unpack_x_real

    @abstractmethod
    def objective_function(self) -> float: ...

    @abstractmethod
    def get_args(self) -> Tuple:
        """Sets args used by objective_funciton."""
        ...

    def gradient(self, x, *args) -> np.ndarray:
        """"""
        grad = np.array(jax.grad(self.objective_function)(x, *args), dtype=np.float64)
        return grad

    def run(self) -> np.ndarray:

        x = self.pack_x()

        res = minimize(
            self.objective_function,
            x,
            args=self.get_args(),
            jac=self.gradient,
            tol=1e-10,
            method="L-BFGS-B",
            options={
                "maxls": 20,
                "gtol": 1e-10,
                "eps": 1e-10,
                "maxiter": 15000,
                "ftol": 1.0e-10,
                "maxcor": 1000,
                "maxfun": 15000,
                "disp": True,
            },
        )

        etrial = res.fun

        beta_shift, psia, psib = self.unpack_x(res.x)
        if self.sys.ndown > 0:
            psi = np.column_stack([psia, psib])
        else:
            psi = psia

        return etrial, beta_shift, psi

    def pack_x_complex(self) -> np.ndarray:
        x = np.zeros(2 * (self.sys.nup + self.sys.ndown + self.ham.dim) * self.ham.N)
        x[: self.ham.N * self.ham.dim] = self.shift.real
        x[self.ham.N * self.ham.dim : 2 * self.ham.N * self.ham.dim] = self.shift.imag
        x[2 * self.ham.N * self.ham.dim : (2 * self.ham.dim + self.sys.ndown + self.sys.nup) * self.ham.N] = (
            self.psi.real
        )
        x[(2 * self.ham.dim + self.sys.ndown + self.sys.nup) * self.ham.N :] = self.psi.imag
        return x

    def unpack_x_complex(self, x: np.ndarray) -> Tuple:
        """Extracts shift and Slater determinants from x array, which is passed to
        the objective function."""
        shift_real = x[0 : self.ham.N * self.ham.dim].copy()
        shift_real = jax.numpy.reshape(shift_real, (self.ham.dim, self.ham.N)).T
        shift_real = shift_real.astype(np.float64)
        shift_complex = x[self.ham.N * self.ham.dim : 2 * self.ham.N * self.ham.dim].copy()
        shift_complex = jax.numpy.reshape(shift_complex, (self.ham.dim, self.ham.N)).T
        shift_complex = shift_complex.astype(np.float64)

        c0a_real = x[2 * self.ham.N * self.ham.dim : (self.sys.nup + 2 * self.ham.dim) * self.ham.N].copy()
        c0a_real = jax.numpy.reshape(c0a_real, (self.sys.nup, self.ham.N)).T
        c0a_real = c0a_real.astype(np.float64)

        if self.sys.ndown > 0:
            c0b_real = x[
                (self.sys.nup + 2 * self.ham.dim)
                * self.ham.N : (self.sys.nup + 2 * self.ham.dim + self.sys.ndown)
                * self.ham.N
            ].copy()
            c0b_real = jax.numpy.reshape(c0b_real, (self.sys.ndown, self.ham.N)).T
            c0b_real = c0b_real.astype(np.float64)

            c0a_complex = x[
                (self.sys.nup + 2 * self.ham.dim + self.sys.ndown)
                * self.ham.N : (2 * self.sys.nup + 2 * self.ham.dim + self.sys.ndown)
                * self.ham.N
            ].copy()
            c0a_complex = jax.numpy.reshape(c0a_complex, (self.sys.nup, self.ham.N)).T
            c0a_complex = c0a_complex.astype(np.float64)

            c0b_complex = x[(2 * self.sys.nup + 2 * self.ham.dim + self.sys.ndown) * self.ham.N :].copy()
            c0b_complex = jax.numpy.reshape(c0b_complex, (self.sys.ndown, self.ham.N)).T
            c0b_complex = c0b_complex.astype(np.float64)

        else:
            c0a_complex = x[
                (self.sys.nup + 2 * self.ham.dim) * self.ham.N : 2 * (self.sys.nup + self.ham.dim) * self.ham.N
            ].copy()
            c0a_complex = jax.numpy.reshape(c0a_complex, (self.sys.nup, self.ham.N)).T
            c0a_complex = c0a_complex.astype(np.float64)

            c0b_real = npj.zeros_like(c0a_real, dtype=np.float64)
            c0b_complex = npj.zeros_like(c0a_complex, dtype=np.float64)

        shift = shift_real + 1j * shift_complex
        c0a = c0a_real + 1j * c0a_complex
        c0b = c0b_real + 1j * c0b_complex

        return shift, c0a, c0b

    def pack_x_real(self) -> np.ndarray:
        x = np.zeros((self.sys.nup + self.sys.ndown + self.ham.dim) * self.ham.N)
        x[: self.ham.N * self.ham.dim] = self.shift.copy()
        x[self.ham.N * self.ham.dim :] = self.psi.copy()
        return x

    def unpack_x_real(self, x: np.ndarray) -> Tuple:
        """Extracts shift and Slater determinants from x array, which is passed to
        the objective function."""
        shift = x[0 : self.ham.N * self.ham.dim].copy()
        shift = jax.numpy.reshape(shift, (self.ham.dim, self.ham.N)).T
        shift = shift.astype(np.float64)

        c0a = x[self.ham.N * self.ham.dim : (self.sys.nup + self.ham.dim) * self.ham.N].copy()
        c0a = jax.numpy.reshape(c0a, (self.sys.nup, self.ham.N)).T
        c0a = c0a.astype(np.float64)

        if self.sys.ndown > 0:
            c0b = x[(self.sys.nup + self.ham.dim) * self.ham.N :].copy()
            c0b = jax.numpy.reshape(c0b, (self.sys.ndown, self.ham.N)).T
            c0b = c0b.astype(np.float64)
        else:
            c0b = npj.zeros_like(c0a, dtype=np.float64)

        return shift, c0a, c0b

    @plum.dispatch
    def initial_guess(self, ham: HolsteinModel):
        r"""Initial guess for the global optimization for the Holstein Model. 
        We assume the shift to be real. Initial electronic degrees of freedom
        are obtained from diagonalizing the one-body electronic operator T."""
        _, elec_eigvecs_a = np.linalg.eigh(ham.T[0])
        psia = elec_eigvecs_a[:, :self.sys.nup]
        if self.sys.ndown > 0:
            _, elec_eigvecs_a = np.linalg.eigh(ham.T[1])
            psib = elec_eigvecs_a[:, :self.sys.ndown]
            psi = np.column_stack([psia, psib])
        else:
            psi = psia

        Ga = gab(psia, psia)
        if self.sys.ndown > 0:
            Gb = gab(psib, psib)
        else:
            Gb = np.zeros_like(Ga)
        G = [Ga, Gb]

        rho = G[0].diagonal() + G[1].diagonal()
        shift = self.ham.g * rho / self.ham.w0

        self.psi = psi.T.ravel()
        self.shift = shift

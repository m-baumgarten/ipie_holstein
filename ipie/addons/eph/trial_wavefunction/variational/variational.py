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
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
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
        self.shift = shift_init.ravel()

        self.ham = hamiltonian
        self.sys = system

        self.shift_params_rows = 1 # D2 and Toyozawa Ansatz

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
        print('jax grad:    ', grad)
        exit()
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

    def pack_x_complex(self, shift: np.ndarray=None, psi: np.ndarray=None) -> np.ndarray:
        if shift is None:
            shift = self.shift
        if psi is None:
            psi = self.psi

        nparams = 2 * (self.sys.nup + self.sys.ndown + self.ham.dim * self.shift_params_rows) * self.ham.N
        index_shift_real = self.ham.N * self.ham.dim * self.shift_params_rows
        index_shift_complex = 2 * index_shift_real
        index_psi_real = index_shift_complex + (self.sys.ndown + self.sys.nup) * self.ham.N
        index_psi_complex = index_psi_real + (self.sys.ndown + self.sys.nup) * self.ham.N

        x = np.zeros(nparams)
        x[: index_shift_real] = shift.real
        x[index_shift_real : index_shift_complex] = shift.imag
        x[index_shift_complex : index_psi_real] = (
            psi.real
        )
        x[index_psi_real : index_psi_complex] = psi.imag
        return x

    def unpack_x_complex(self, x: np.ndarray) -> Tuple:
        """Extracts shift and Slater determinants from x array, which is passed to
        the objective function."""
        # Make these class attributes s.t. we only need to change these for defining D1 or D2 trial
        index_shift_real = self.ham.N * self.ham.dim * self.shift_params_rows
        index_shift_complex = 2 * index_shift_real
        index_c0a_real = index_shift_complex + self.sys.nup * self.ham.N

        shift_real = x[ : index_shift_real].copy()
        shift_real = jax.numpy.reshape(shift_real, (self.ham.dim, self.ham.N, self.shift_params_rows))
        shift_real = shift_real.astype(np.float64)

        shift_complex = x[index_shift_real : index_shift_complex].copy()
        shift_complex = jax.numpy.reshape(shift_complex, (self.ham.dim, self.ham.N, self.shift_params_rows))
        shift_complex = shift_complex.astype(np.float64)

        c0a_real = x[index_shift_complex : index_c0a_real].copy()
        c0a_real = jax.numpy.reshape(c0a_real, (self.sys.nup, self.ham.N)).T
        c0a_real = c0a_real.astype(np.float64)

        if self.sys.ndown > 0:
            index_c0b_real = index_c0a_real + self.sys.ndown * self.ham.N
            index_c0a_complex = index_c0b_real + self.sys.nup * self.ham.N

            c0b_real = x[index_c0a_real : index_c0b_real].copy()
            c0b_real = jax.numpy.reshape(c0b_real, (self.sys.ndown, self.ham.N)).T
            c0b_real = c0b_real.astype(np.float64)

            c0a_complex = x[index_c0b_real : index_c0a_complex].copy()
            c0a_complex = jax.numpy.reshape(c0a_complex, (self.sys.nup, self.ham.N)).T
            c0a_complex = c0a_complex.astype(np.float64)

            c0b_complex = x[index_c0a_complex:].copy()
            c0b_complex = jax.numpy.reshape(c0b_complex, (self.sys.ndown, self.ham.N)).T
            c0b_complex = c0b_complex.astype(np.float64)

        else:

            c0a_complex = x[index_c0a_real : ].copy()
            c0a_complex = jax.numpy.reshape(c0a_complex, (self.sys.nup, self.ham.N)).T
            c0a_complex = c0a_complex.astype(np.float64)

            c0b_real = npj.zeros_like(c0a_real, dtype=np.float64)
            c0b_complex = npj.zeros_like(c0a_complex, dtype=np.float64)

        shift = shift_real + 1j * shift_complex
        c0a = c0a_real + 1j * c0a_complex
        c0b = c0b_real + 1j * c0b_complex

        return shift, c0a, c0b

    def pack_x_real(self) -> np.ndarray:
        nparams = (self.sys.nup + self.sys.ndown + self.ham.dim * self.shift_params_rows) * self.ham.N
        index_shift_real = self.ham.N * self.ham.dim

        x = np.zeros(nparms)
        x[: index_shift_real] = self.shift.copy()
        x[index_shift_real :] = self.psi.copy()
        return x

    def unpack_x_real(self, x: np.ndarray) -> Tuple:
        """Extracts shift and Slater determinants from x array, which is passed to
        the objective function."""
        index_shift_real = self.ham.N * self.ham.dim * self.shift_params_rows
        index_c0a_real = index_shift_real + self.ham.N * self.sys.nup

        shift = x[ : index_shift_real].copy()
        shift = jax.numpy.reshape(shift, (self.ham.dim, self.ham.N, self.shift_params_rows))
        shift = shift.astype(np.float64)

        c0a = x[index_shift_real : index_c0a_real].copy()
        c0a = jax.numpy.reshape(c0a, (self.sys.nup, self.ham.N)).T
        c0a = c0a.astype(np.float64)

        if self.sys.ndown > 0:
            c0b = x[index_c0a_real :].copy()
            c0b = jax.numpy.reshape(c0b, (self.sys.ndown, self.ham.N)).T
            c0b = c0b.astype(np.float64)
        else:
            c0b = npj.zeros_like(c0a, dtype=np.float64)

        return shift, c0a, c0b

    @plum.dispatch
    def initial_guess(self, ham: GenericEPhModel) -> None:
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

        shift = np.einsum('ijk,ij->k', ham.g_tensor, G[0] + G[1]) / self.ham.w0        
        self.psi = psi.T.ravel()
        self.shift = shift

#    @plum.dispatch
#    def initial_guess(self, ham: HolsteinModel):
#        r"""Initial guess for the global optimization for the Holstein Model. 
#        We assume the shift to be real. Initial electronic degrees of freedom
#        are obtained from diagonalizing the one-body electronic operator T."""
#        _, elec_eigvecs_a = np.linalg.eigh(ham.T[0])
#        psia = elec_eigvecs_a[:, :self.sys.nup]
#        if self.sys.ndown > 0:
#            _, elec_eigvecs_a = np.linalg.eigh(ham.T[1])
#            psib = elec_eigvecs_a[:, :self.sys.ndown]
#            psi = np.column_stack([psia, psib])
#        else:
#            psi = psia
#
#        Ga = gab(psia, psia)
#        if self.sys.ndown > 0:
#            Gb = gab(psib, psib)
#        else:
#            Gb = np.zeros_like(Ga)
#        G = [Ga, Gb]
#
#        rho = G[0].diagonal() + G[1].diagonal()
#        shift = self.ham.g * rho / self.ham.w0
#
#        self.psi = psi.T.ravel()
#        self.shift = shift

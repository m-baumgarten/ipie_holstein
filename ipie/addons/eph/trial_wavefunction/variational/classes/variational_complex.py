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
from typing import Tuple
from scipy.optimize import minimize, basinhopping
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab

import jax
import jax.numpy as npj

from abc import abstractmethod, ABCMeta


class VariationalComplex(metaclass=ABCMeta):
    def __init__(self, shift_init: np.ndarray, electron_init: np.ndarray, hamiltonian, system):
        self.psi = electron_init.T.ravel()
        self.psi_real = self.psi.real
        self.psi_complex = self.psi.imag

        self.shift = shift_init
        self.shift_real = self.shift.real
        self.shift_complex = self.shift.imag

        self.ham = hamiltonian
        self.sys = system

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

        x = np.zeros(2 * (self.sys.nup + self.sys.ndown + 1) * self.ham.nsites)
        x[: self.ham.nsites] = self.shift_real
        x[self.ham.nsites : 2 * self.ham.nsites] = self.shift_complex
        x[2 * self.ham.nsites : (2 + self.sys.ndown + self.sys.nup) * self.ham.nsites] = self.psi_real
        x[(2 + self.sys.ndown + self.sys.nup) * self.ham.nsites : ] = self.psi_complex

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
        beta_shift = res.x[: self.ham.nsites] + 1j * res.x[self.ham.nsites : 2 * self.ham.nsites]

        if self.sys.ndown > 0:
            psia_real = res.x[2 * self.ham.nsites : self.ham.nsites * (self.sys.nup + 2)]
            psia_real = psia_real.reshape((self.sys.nup, self.ham.nsites)).T
            psib_real = res.x[self.ham.nsites * (self.sys.nup + 2) : self.ham.nsites * (self.sys.nup + self.sys.ndown + 2)]
            psib_real = psib_real.reshape((self.sys.ndown, self.ham.nsites)).T
            
            psia_complex = res.x[self.ham.nsites * (self.sys.nup + self.sys.ndown + 2) : self.ham.nsites * (2 * self.sys.nup + self.sys.ndown + 2)]
            psia_complex = psia_complex.reshape((self.sys.nup, self.ham.nsites)).T  
            psib_complex = res.x[self.ham.nsites * (2 * self.sys.nup + self.sys.ndown + 2) : ]
            psib_complex = psib_complex.reshape((self.sys.ndown, self.ham.nsites)).T 
            
            psia = psia_real + 1j * psia_complex
            psib = psib_real + 1j * psib_complex

            psi = np.column_stack([psia, psib])
        else:
            psia_real = res.x[2 * self.ham.nsites : self.ham.nsites * (self.sys.nup + 2)]
            psia_real = psia_real.reshape((self.sys.nup, self.ham.nsites)).T
            psia_complex = res.x[self.ham.nsites * (self.sys.nup + 2) : ]
            psia_complex = psia_complex.reshape((self.sys.nup, self.ham.nsites)).T

            psi = psia_real + 1j * psia_complex

        return etrial, beta_shift, psi

    def unpack_x(self, x: np.ndarray) -> Tuple:
        """Extracts shift and Slater determinants from x array, which is passed to
        the objective function."""
        shift_real = x[0 : self.ham.nsites].copy()
        shift_real = shift_real.astype(np.float64)
        shift_complex = x[self.ham.nsites : 2 * self.ham.nsites].copy()
        shift_complex = shift_complex.astype(np.float64)

        c0a_real = x[2 * self.ham.nsites : (self.sys.nup + 2) * self.ham.nsites].copy()
        c0a_real = jax.numpy.reshape(c0a_real, (self.sys.nup, self.ham.nsites)).T
        c0a_real = c0a_real.astype(np.float64)
        
        if self.sys.ndown > 0:
            c0b_real = x[(self.sys.nup + 2) * self.ham.nsites : (self.sys.nup + 2 + self.sys.ndown) * self.ham.nsites].copy()
            c0b_real = jax.numpy.reshape(c0b_real, (self.sys.ndown, self.ham.nsites)).T
            c0b_real = c0b_real.astype(np.float64)
        
            c0a_complex = x[(self.sys.nup + 2 + self.sys.ndown) * self.ham.nsites : (2 * self.sys.nup + 2 + self.sys.ndown) * self.ham.nsites].copy()
            c0a_complex = jax.numpy.reshape(c0a_complex, (self.sys.nup, self.ham.nsites)).T
            c0a_complex = c0a_complex.astype(np.float64)

            c0b_complex = x[(2 * self.sys.nup + 2 + self.sys.ndown) * self.ham.nsites : ].copy()
            c0b_complex = jax.numpy.reshape(c0b_complex, (self.sys.ndown, self.ham.nsites)).T
            c0b_complex = c0b_complex.astype(np.float64)

        else:
            c0a_complex = x[(self.sys.nup + 2) * self.ham.nsites : 2 * (self.sys.nup + 1) * self.ham.nsites].copy()
            c0a_complex = jax.numpy.reshape(c0a_complex, (self.sys.nup, self.ham.nsites)).T
            c0a_complex = c0a_complex.astype(np.float64)

            c0b_real = npj.zeros_like(c0a_real, dtype=np.float64)
            c0b_complex = npj.zeros_like(c0a_complex, dtype=np.float64)

        shift = shift_real + 1j * shift_complex
        c0a = c0a_real + 1j * c0a_complex
        c0b = c0b_real + 1j * c0b_complex

        return shift, c0a, c0b

#        return shift_real, shift_complex, c0a_real, c0a_complex, c0b_real, c0b_complex

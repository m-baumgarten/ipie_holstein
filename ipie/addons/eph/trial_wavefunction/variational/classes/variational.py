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


class Variational(metaclass=ABCMeta):
    def __init__(self, shift_init: np.ndarray, electron_init: np.ndarray, hamiltonian, system):
        self.psi = electron_init.T.real.ravel()
        self.shift = shift_init.real
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

        x = np.zeros((self.sys.nup + self.sys.ndown + 1) * self.ham.nsites)
        x[: self.ham.nsites] = self.shift.copy()
        x[self.ham.nsites :] = self.psi.copy()

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
        beta_shift = res.x[: self.ham.nsites]

        if self.sys.ndown > 0:
            psia = res.x[self.ham.nsites : self.ham.nsites * (self.sys.nup + 1)]
            psia = psia.reshape((self.sys.nup, self.ham.nsites)).T
            psib = res.x[self.ham.nsites * (self.sys.nup + 1) :]
            psib = psib.reshape((self.sys.ndown, self.ham.nsites)).T
            psi = np.column_stack([psia, psib])
        else:
            psia = res.x[self.ham.nsites :].reshape((self.sys.nup, self.ham.nsites)).T
            psi = psia

        return etrial, beta_shift, psi

    def unpack_x(self, x: np.ndarray) -> Tuple:
        """Extracts shift and Slater determinants from x array, which is passed to
        the objective function."""
        shift = x[0 : self.ham.nsites].copy()
        shift = shift.astype(np.float64)

        c0a = x[self.ham.nsites : (self.sys.nup + 1) * self.ham.nsites].copy()
        c0a = jax.numpy.reshape(c0a, (self.sys.nup, self.ham.nsites)).T
        c0a = c0a.astype(np.float64)

        if self.sys.ndown > 0:
            c0b = x[(self.sys.nup + 1) * self.ham.nsites :].copy()
            c0b = jax.numpy.reshape(c0b, (self.sys.ndown, self.ham.nsites)).T
            c0b = c0b.astype(np.float64)
        else:
            c0b = npj.zeros_like(c0a, dtype=np.float64)

        return shift, c0a, c0b

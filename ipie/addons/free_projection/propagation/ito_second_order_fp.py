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

import time

import numpy

from scipy.linalg import expm
from ipie.propagation.operations import apply_exponential_batch_vectorized
from ipie.utils.backend import synchronize
from ipie.addons.free_projection.propagation.ito_propagator_fp import EulerItoPropagatorFP


class ItoSymmSplitPropagatorFP(EulerItoPropagatorFP):
    r"""
    Symmetric split free-projection coherent-state Ito propagator.

    The implemented ordering is

        exp(-dt H_ph/2) exp(-dt H_ann+el/2)
        exp(-dt H_+) exp(-dt H_ann+el/2) exp(-dt H_ph/2).

    The stochastic creation factors are applied with the exact finite-step
    exponential by default. For small generator norms, the same action can be
    approximated by a batched Taylor series without explicitly building the
    matrix exponential.
    """

    def __init__(
        self,
        time_step,
        verbose=False,
        mean_field_subtraction=False,
        mean_field_shift=None,
        reference_energy=None,
        exponential_action="expm",
        exponential_taylor_order=6,
    ):
        if exponential_action not in ("expm", "taylor"):
            raise ValueError("exponential_action must be 'expm' or 'taylor'.")
        exponential_taylor_order = int(exponential_taylor_order)
        if exponential_taylor_order < 1:
            raise ValueError("exponential_taylor_order must be positive.")
        super().__init__(
            time_step,
            verbose=verbose,
            mean_field_subtraction=mean_field_subtraction,
            mean_field_shift=mean_field_shift,
            reference_energy=reference_energy,
        )
        self.exponential_action = exponential_action
        self.exponential_taylor_order = exponential_taylor_order

    def propagate(self, walkers, hamiltonian, trial) -> None:
        r"""Apply one symmetric split step with static mean-field subtraction."""
        start_time = time.time()

        self.apply_phonon_damping(walkers, hamiltonian, 0.5 * self.dt)
        self.apply_annihilation_step(walkers, hamiltonian, 0.5 * self.dt)
        self.apply_creation_step(walkers, hamiltonian, self.dt)
        self.apply_annihilation_step(walkers, hamiltonian, 0.5 * self.dt)
        self.apply_phonon_damping(walkers, hamiltonian, 0.5 * self.dt)

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def apply_phonon_damping(self, walkers, hamiltonian, step_size):
        walkers.coherent_state_shift *= numpy.exp(-step_size * hamiltonian.w0)

    def apply_annihilation_step(self, walkers, hamiltonian, step_size):
        h_eff = self.construct_annihilation_matrix(
            walkers.coherent_state_shift, hamiltonian
        )
        walkers.phia = self.apply_exponential(walkers.phia, -step_size * h_eff[0])
        if walkers.ndown > 0:
            walkers.phib = self.apply_exponential(walkers.phib, -step_size * h_eff[1])

    def apply_creation_step(
        self,
        walkers,
        hamiltonian,
        step_size,
        dZ=None,
        delta_lambda=None,
        dZ_creation=None,
    ):
        if delta_lambda is None:
            delta_lambda = numpy.zeros(
                (walkers.nwalkers, hamiltonian.N), dtype=numpy.complex128
            )
        else:
            delta_lambda = numpy.asarray(delta_lambda, dtype=numpy.complex128)

        if dZ is None:
            if self.creation_coefficients_are_zero(delta_lambda):
                walkers.coherent_state_shift += -step_size * (
                    self.eph_mean_field.conj() + delta_lambda
                )
                return None
            dZ = self.sample_complex_noise_with_step(walkers, hamiltonian, step_size)
        else:
            dZ = numpy.asarray(dZ, dtype=numpy.complex128)
        if dZ_creation is None:
            dZ_creation = dZ
        else:
            dZ_creation = numpy.asarray(dZ_creation, dtype=numpy.complex128)

        creation = self.construct_creation_matrix(
            dZ_creation, hamiltonian, delta_lambda=delta_lambda
        )
        walkers.coherent_state_shift += (
            -step_size * (self.eph_mean_field.conj() + delta_lambda)
            + dZ
        )

        walkers.phia = self.apply_exponential(walkers.phia, -creation)
        if walkers.ndown > 0:
            walkers.phib = self.apply_exponential(walkers.phib, -creation)
        return dZ

    def apply_exponential(self, phi, generators):
        r"""Apply the configured exponential action to each walker determinant."""
        if self.exponential_action == "taylor":
            return apply_exponential_batch_vectorized(
                phi.copy(), generators, self.exponential_taylor_order
            )
        propagators = numpy.stack([expm(generator) for generator in generators])
        return numpy.einsum("nij,nje->nie", propagators, phi)

    def construct_creation_matrix(self, dZ, hamiltonian, delta_lambda=None):
        r"""Build ``sum_mu dZ_mu^* (G_mu^\dagger - delta_lambda_mu I)``."""
        creation = numpy.einsum("ijk,nk->nij", self.g_tensor_residual_dagger, dZ.conj())
        if delta_lambda is not None:
            delta_lambda = numpy.asarray(delta_lambda, dtype=numpy.complex128)
            if numpy.any(delta_lambda):
                scalar_shift = numpy.einsum("nm,nm->n", delta_lambda, dZ.conj())
                identity = numpy.eye(creation.shape[1], dtype=numpy.complex128)
                creation -= scalar_shift[:, None, None] * identity[None, :, :]
        return creation

    def sample_complex_noise_with_step(self, walkers, hamiltonian, step_size):
        r"""Draw proper complex Gaussian increments with variance ``step_size``."""
        gaussian = numpy.random.normal(
            loc=0.0, scale=1.0, size=(2, walkers.nwalkers, hamiltonian.N)
        )
        return numpy.sqrt(0.5 * step_size) * (gaussian[0] + 1j * gaussian[1])

    def creation_coefficients_are_zero(self, delta_lambda=None):
        residual_zero = numpy.allclose(self.g_tensor_residual_dagger, 0.0)
        if delta_lambda is None:
            return residual_zero
        return residual_zero and numpy.allclose(delta_lambda, 0.0)

    def update_weight(self, walkers, ovlp=None, ovlp_new=None, eshift=None) -> None:
        """Apply only the scalar reference-energy shift inherited from FP Ito."""
        return super().update_weight(walkers, ovlp=ovlp, ovlp_new=ovlp_new, eshift=eshift)


class ItoSymmSplitImportancePropagatorFP(ItoSymmSplitPropagatorFP):
    r"""
    Importance-sampled symmetric split coherent-state Ito propagator.

    This class keeps the same split ordering as :class:`ItoSymmSplitPropagatorFP`,
    but proposes the stochastic creation step from a drifted proper-complex
    Gaussian and compensates exactly with Gaussian likelihood and trial-overlap
    ratios.
    """

    supports_importance_sampling = True

    def __init__(
        self,
        time_step,
        verbose=False,
        mean_field_subtraction=False,
        mean_field_shift=None,
        reference_energy=None,
        force_bias="overlap",
        force_bias_scale=1.0,
        zero_overlap_threshold=1.0e-14,
        split_gauge="static",
        split_gauge_scale=1.0,
        split_gauge_max_norm=None,
        split_gauge_q=1.0,
        exponential_action="expm",
        exponential_taylor_order=6,
    ):
        if force_bias not in ("overlap", "zero"):
            raise ValueError("force_bias must be 'overlap' or 'zero'.")
        if split_gauge not in ("static", "phase_cancel"):
            raise ValueError("split_gauge must be 'static' or 'phase_cancel'.")
        if not 0.0 <= float(split_gauge_scale) <= 1.0:
            raise ValueError("split_gauge_scale must be between 0 and 1.")
        split_gauge_q = float(split_gauge_q)
        if not numpy.isfinite(split_gauge_q) or split_gauge_q <= 0.0:
            raise ValueError("split_gauge_q must be finite and positive.")
        if split_gauge != "phase_cancel" and split_gauge_q != 1.0:
            raise ValueError("split_gauge_q != 1 requires split_gauge='phase_cancel'.")
        super().__init__(
            time_step,
            verbose=verbose,
            mean_field_subtraction=mean_field_subtraction,
            mean_field_shift=mean_field_shift,
            reference_energy=reference_energy,
            exponential_action=exponential_action,
            exponential_taylor_order=exponential_taylor_order,
        )
        self.force_bias = force_bias
        self.force_bias_scale = float(force_bias_scale)
        self.zero_overlap_threshold = float(zero_overlap_threshold)
        self.split_gauge = split_gauge
        self.split_gauge_scale = float(split_gauge_scale)
        self.split_gauge_max_norm = (
            None if split_gauge_max_norm is None else float(split_gauge_max_norm)
        )
        self.split_gauge_q = split_gauge_q
        self.split_gauge_sqrt_q = numpy.sqrt(self.split_gauge_q)
        if self.split_gauge_max_norm is not None and self.split_gauge_max_norm < 0.0:
            raise ValueError("split_gauge_max_norm must be non-negative.")

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=None) -> None:
        start_time = time.time()
        ovlp = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp
        synchronize()
        self.timer.tovlp += time.time() - start_time

        log_likelihood = self.propagate(walkers, hamiltonian, trial)

        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp_new
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(
            walkers,
            ovlp=ovlp,
            ovlp_new=ovlp_new,
            log_likelihood=log_likelihood,
            eshift=eshift,
        )
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def propagate(self, walkers, hamiltonian, trial):
        r"""Apply one importance-sampled symmetric split step."""
        start_time = time.time()
        self.apply_phonon_damping(walkers, hamiltonian, 0.5 * self.dt)
        self.apply_annihilation_step(walkers, hamiltonian, 0.5 * self.dt)

        dZ, dZ_creation, log_likelihood, delta_lambda = self.sample_biased_complex_noise(
            walkers, hamiltonian, trial, self.dt
        )
        self.apply_creation_step(
            walkers,
            hamiltonian,
            self.dt,
            dZ=dZ,
            delta_lambda=delta_lambda,
            dZ_creation=dZ_creation,
        )

        self.apply_annihilation_step(walkers, hamiltonian, 0.5 * self.dt)
        self.apply_phonon_damping(walkers, hamiltonian, 0.5 * self.dt)

        synchronize()
        self.timer.tgemm += time.time() - start_time
        return log_likelihood

    def sample_biased_complex_noise(self, walkers, hamiltonian, trial, step_size):
        r"""Draw ``dZ = a * step_size + dW`` and return its log likelihood ratio."""
        delta_lambda, A, B = self.construct_split_gauge(walkers, hamiltonian, trial)
        if self.creation_coefficients_are_zero(delta_lambda):
            dZ = numpy.zeros((walkers.nwalkers, hamiltonian.N), dtype=numpy.complex128)
            log_likelihood = numpy.zeros(walkers.nwalkers, dtype=numpy.float64)
            return dZ, dZ, log_likelihood, delta_lambda

        dW = self.sample_complex_noise_with_step(walkers, hamiltonian, step_size)
        drift = self.construct_force_bias(
            walkers,
            hamiltonian,
            trial,
            A=A,
            B=B,
            noise_scale=self.split_gauge_sqrt_q,
        )
        dZ_base = drift * step_size + dW
        dZ = self.split_gauge_sqrt_q * dZ_base
        dZ_creation = dZ_base / self.split_gauge_sqrt_q
        log_likelihood = self.gaussian_log_likelihood_ratio(drift, dW, step_size)
        return dZ, dZ_creation, log_likelihood, delta_lambda

    def construct_split_gauge(self, walkers, hamiltonian, trial):
        zeros = numpy.zeros((walkers.nwalkers, hamiltonian.N), dtype=numpy.complex128)
        if self.split_gauge == "static":
            return zeros, None, None

        A, B = self.construct_ito_log_derivatives(walkers, trial)
        delta_lambda = self.split_gauge_scale * (B + self.split_gauge_q * A.conj())
        delta_lambda = self.apply_split_gauge_bound(delta_lambda)
        B_gauged = B - delta_lambda
        return delta_lambda, A, B_gauged

    def construct_ito_log_derivatives(self, walkers, trial):
        if not hasattr(trial, "calc_ito_log_derivatives"):
            raise TypeError(
                "Dynamic split-gauge propagation requires trial.calc_ito_log_derivatives."
            )
        A, B = trial.calc_ito_log_derivatives(
            walkers,
            self.g_tensor_residual_dagger,
            zero_overlap_threshold=self.zero_overlap_threshold,
        )
        return (
            numpy.asarray(A, dtype=numpy.complex128),
            numpy.asarray(B, dtype=numpy.complex128),
        )

    def apply_split_gauge_bound(self, delta_lambda):
        if self.split_gauge_max_norm is None:
            return delta_lambda
        norms = numpy.linalg.norm(delta_lambda, axis=1)
        active = norms > self.split_gauge_max_norm
        if numpy.any(active):
            scale = self.split_gauge_max_norm / norms[active]
            delta_lambda = delta_lambda.copy()
            delta_lambda[active] *= scale[:, None]
        return delta_lambda

    def construct_force_bias(
        self, walkers, hamiltonian, trial, A=None, B=None, noise_scale=1.0
    ):
        if self.force_bias == "zero" or self.force_bias_scale == 0.0:
            return numpy.zeros((walkers.nwalkers, hamiltonian.N), dtype=numpy.complex128)
        if A is not None and B is not None:
            drift = 0.5 * (noise_scale * A.conj() - B / noise_scale)
            return self.force_bias_scale * numpy.asarray(drift, dtype=numpy.complex128)

        if hasattr(trial, "calc_ito_log_derivatives"):
            A, B = self.construct_ito_log_derivatives(walkers, trial)
            drift = 0.5 * (A.conj() - B)
        elif hasattr(trial, "calc_ito_force_bias"):
            drift = trial.calc_ito_force_bias(
                walkers,
                self.g_tensor_residual_dagger,
                zero_overlap_threshold=self.zero_overlap_threshold,
            )
        else:
            raise TypeError(
                "Importance-sampled Ito propagation requires trial.calc_ito_force_bias."
            )
        return self.force_bias_scale * numpy.asarray(drift, dtype=numpy.complex128)

    @staticmethod
    def gaussian_log_likelihood_ratio(drift, dW, step_size):
        r"""Return log[p(dW + a dt) / q(dW)] for a proper-complex drift proposal."""
        return (
            -2.0 * numpy.real(numpy.einsum("nm,nm->n", drift.conj(), dW))
            - step_size * numpy.sum(numpy.abs(drift) ** 2, axis=1)
        )

    def initialize_importance_weights(self, walkers, trial) -> None:
        r"""Fold the initial trial overlap into ``weight * phase`` once."""
        if getattr(walkers, "_importance_sampling_initialized", False):
            return None

        ovlp = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp
        log_abs = numpy.full(walkers.nwalkers, -numpy.inf, dtype=numpy.float64)
        active = numpy.isfinite(ovlp.real) & numpy.isfinite(ovlp.imag)
        active &= numpy.abs(ovlp) > self.zero_overlap_threshold
        log_abs[active] = numpy.log(numpy.abs(ovlp[active]))
        self._apply_complex_factor(walkers, ovlp, log_abs)
        walkers._importance_sampling_initialized = True
        return None

    def update_weight(
        self,
        walkers,
        ovlp=None,
        ovlp_new=None,
        log_likelihood=None,
        eshift=None,
    ) -> None:
        r"""Apply likelihood, trial-overlap, and reference-energy factors."""
        if ovlp is None or ovlp_new is None:
            raise ValueError("Importance-sampled propagation requires old and new overlaps.")

        if log_likelihood is None:
            log_likelihood = numpy.zeros(walkers.nwalkers, dtype=numpy.float64)
        else:
            log_likelihood = numpy.asarray(log_likelihood, dtype=numpy.float64)

        reference_energy = self._resolve_reference_energy(eshift)
        log_scalar = log_likelihood.copy()
        if reference_energy is not None and reference_energy != 0.0:
            log_scalar += self.dt * reference_energy

        ratio = numpy.zeros_like(ovlp_new, dtype=numpy.complex128)
        active = numpy.isfinite(ovlp.real) & numpy.isfinite(ovlp.imag)
        active &= numpy.isfinite(ovlp_new.real) & numpy.isfinite(ovlp_new.imag)
        active &= numpy.isfinite(log_scalar)
        active &= numpy.abs(ovlp) > self.zero_overlap_threshold
        ratio[active] = ovlp_new[active] / ovlp[active]

        log_abs = numpy.full(walkers.nwalkers, -numpy.inf, dtype=numpy.float64)
        ratio_abs = numpy.abs(ratio)
        nonzero = active & (ratio_abs > self.zero_overlap_threshold)
        log_abs[nonzero] = log_scalar[nonzero] + numpy.log(ratio_abs[nonzero])
        self._apply_complex_factor(walkers, ratio, log_abs)
        walkers.ovlp = ovlp_new
        return None

    def _apply_complex_factor(self, walkers, factor, log_abs_multiplier) -> None:
        factor = numpy.asarray(factor, dtype=numpy.complex128)
        log_abs_multiplier = numpy.asarray(log_abs_multiplier, dtype=numpy.float64)
        factor_abs = numpy.abs(factor)
        active = numpy.isfinite(factor.real) & numpy.isfinite(factor.imag)
        active &= numpy.isfinite(log_abs_multiplier)
        active &= factor_abs > self.zero_overlap_threshold

        inactive = ~active
        if numpy.any(inactive):
            walkers.weight[inactive] = 0.0
            if hasattr(walkers, "weight_log"):
                walkers.weight_log[inactive] = -numpy.inf

        if numpy.any(active):
            walkers.weight[active] *= numpy.exp(log_abs_multiplier[active])
            walkers.phase[active] *= factor[active] / factor_abs[active]
            if hasattr(walkers, "weight_log"):
                walkers.weight_log[active] += log_abs_multiplier[active]

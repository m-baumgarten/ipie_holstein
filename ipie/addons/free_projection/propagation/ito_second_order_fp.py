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

from pathlib import Path
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
        self.diagnostics_enabled = False
        self._diagnostic_filename = None
        self._diagnostic_stride = 1
        self._diagnostic_max_records = None
        self._diagnostic_steps_per_block = None
        self._diagnostic_rank = 0
        self._diagnostic_size = 1
        self._diagnostic_step = 0
        self._diagnostic_records = []
        self._diagnostic_current = None
        self._diagnostic_failure = {}
        self._diagnostic_last_arrays = {}
        self._diagnostic_annihilation_count = 0

    def configure_diagnostics(
        self,
        filename=None,
        mpi_handler=None,
        stride=1,
        max_records=None,
        steps_per_block=None,
    ):
        r"""Enable compact per-step diagnostics for unstable importance runs."""
        if filename is None:
            self.diagnostics_enabled = False
            self._diagnostic_filename = None
            return None

        stride = int(stride)
        if stride < 1:
            raise ValueError("diagnostics stride must be positive.")
        if max_records is not None:
            max_records = int(max_records)
            if max_records < 1:
                raise ValueError("diagnostics max records must be positive.")

        rank = 0
        size = 1
        if mpi_handler is not None and hasattr(mpi_handler, "comm"):
            comm = mpi_handler.comm
            rank = getattr(comm, "rank", None)
            if rank is None:
                rank = comm.Get_rank()
            size = getattr(comm, "size", None)
            if size is None:
                size = comm.Get_size()

        path = Path(filename)
        if size > 1:
            path = path.with_name(f"{path.stem}.rank{rank:04d}{path.suffix}")

        self.diagnostics_enabled = True
        self._diagnostic_filename = path
        self._diagnostic_stride = stride
        self._diagnostic_max_records = max_records
        self._diagnostic_steps_per_block = (
            None if steps_per_block is None else int(steps_per_block)
        )
        self._diagnostic_rank = int(rank)
        self._diagnostic_size = int(size)
        self._diagnostic_records = []
        self._diagnostic_current = None
        self._diagnostic_failure = {}
        self._diagnostic_last_arrays = {}
        self._diagnostic_step = 0
        return path

    def save_diagnostics(self, reason="manual"):
        if not self.diagnostics_enabled or self._diagnostic_filename is None:
            return None

        fields = self._diagnostic_fields()
        records = numpy.full(
            (len(self._diagnostic_records), len(fields)), numpy.nan, dtype=numpy.float64
        )
        for irec, record in enumerate(self._diagnostic_records):
            for ifield, field in enumerate(fields):
                if field in record:
                    records[irec, ifield] = record[field]

        arrays = {
            "fields": numpy.asarray(fields, dtype="<U96"),
            "records": records,
            "save_reason": numpy.asarray(str(reason)),
            "failure_stage": numpy.asarray(str(self._diagnostic_failure.get("stage", ""))),
            "failure_exception": numpy.asarray(
                str(self._diagnostic_failure.get("exception", ""))
            ),
            "failure_step": numpy.asarray(
                self._diagnostic_failure.get("step", -1), dtype=numpy.int64
            ),
            "failure_walker_index": numpy.asarray(
                self._diagnostic_failure.get("walker_index", -1), dtype=numpy.int64
            ),
            "rank": numpy.asarray(self._diagnostic_rank, dtype=numpy.int64),
            "size": numpy.asarray(self._diagnostic_size, dtype=numpy.int64),
        }
        arrays.update(self._diagnostic_last_arrays)

        path = Path(self._diagnostic_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez(path, **arrays)
        return path

    def _diagnostic_fields(self):
        preferred = [
            "rank",
            "absolute_step",
            "block",
            "step_in_block",
            "old_overlap_min_abs",
            "old_overlap_max_abs",
            "new_overlap_min_abs",
            "new_overlap_max_abs",
            "log_likelihood_min",
            "log_likelihood_max",
            "overlap_ratio_log_min",
            "overlap_ratio_log_max",
            "weight_increment_log_min",
            "weight_increment_log_max",
            "final_weight_log_min",
            "final_weight_log_max",
            "final_weight_log_spread",
            "final_raw_weight_max_abs",
            "final_phase_resultant",
            "final_coherent_shift_max_norm",
        ]
        keys = set(preferred)
        for record in self._diagnostic_records:
            keys.update(record)
        return [key for key in preferred if key in keys] + sorted(keys - set(preferred))

    def _diagnostic_begin_step(self, walkers):
        if not self.diagnostics_enabled:
            return None
        self._diagnostic_current = None
        self._diagnostic_annihilation_count = 0
        if self._diagnostic_step % self._diagnostic_stride != 0:
            return None
        if (
            self._diagnostic_max_records is not None
            and len(self._diagnostic_records) >= self._diagnostic_max_records
        ):
            return None

        record = {
            "rank": float(self._diagnostic_rank),
            "absolute_step": float(self._diagnostic_step),
        }
        if self._diagnostic_steps_per_block is not None:
            record["block"] = float(self._diagnostic_step // self._diagnostic_steps_per_block)
            record["step_in_block"] = float(
                self._diagnostic_step % self._diagnostic_steps_per_block
            )
        self._diagnostic_current = record
        self._diagnostic_record_walker_state("initial", walkers)
        self._diagnostic_check_finite("initial", walkers)
        return None

    def _diagnostic_finish_step(self, walkers):
        if not self.diagnostics_enabled:
            return None
        self._diagnostic_record_walker_state("final", walkers)
        self._diagnostic_check_finite("final", walkers)
        if self._diagnostic_current is not None:
            self._diagnostic_records.append(self._diagnostic_current)
            self._diagnostic_current = None
            self.save_diagnostics(reason="step")
        self._diagnostic_step += 1
        return None

    def _diagnostic_record_array(self, prefix, values):
        if not self.diagnostics_enabled or self._diagnostic_current is None:
            return None
        if values is None:
            return None
        array = numpy.asarray(values)
        if array.size == 0:
            return None
        finite = numpy.isfinite(array)
        self._diagnostic_current[f"{prefix}_finite_fraction"] = float(
            numpy.count_nonzero(finite) / finite.size
        )
        abs_values = numpy.abs(array)
        finite_abs = numpy.isfinite(abs_values)
        if numpy.any(finite_abs):
            self._diagnostic_current[f"{prefix}_mean_abs"] = float(
                numpy.mean(abs_values[finite_abs])
            )
            self._diagnostic_current[f"{prefix}_max_abs"] = float(
                numpy.max(abs_values[finite_abs])
            )
            self._diagnostic_current[f"{prefix}_min_abs"] = float(
                numpy.min(abs_values[finite_abs])
            )
        if array.ndim >= 2:
            norms = numpy.linalg.norm(array.reshape(array.shape[0], -1), axis=1)
        else:
            norms = abs_values.reshape(-1)
        finite_norms = numpy.isfinite(norms)
        if numpy.any(finite_norms):
            self._diagnostic_current[f"{prefix}_mean_norm"] = float(
                numpy.mean(norms[finite_norms])
            )
            self._diagnostic_current[f"{prefix}_max_norm"] = float(
                numpy.max(norms[finite_norms])
            )
            self._diagnostic_current[f"{prefix}_argmax_norm"] = float(
                numpy.argmax(numpy.where(finite_norms, norms, -numpy.inf))
            )
        return None

    def _diagnostic_record_real(self, prefix, values):
        if not self.diagnostics_enabled or self._diagnostic_current is None:
            return None
        array = numpy.asarray(values, dtype=numpy.float64)
        if array.size == 0:
            return None
        finite = numpy.isfinite(array)
        self._diagnostic_current[f"{prefix}_finite_fraction"] = float(
            numpy.count_nonzero(finite) / finite.size
        )
        if numpy.any(finite):
            self._diagnostic_current[f"{prefix}_min"] = float(numpy.min(array[finite]))
            self._diagnostic_current[f"{prefix}_max"] = float(numpy.max(array[finite]))
            self._diagnostic_current[f"{prefix}_mean"] = float(numpy.mean(array[finite]))
        return None

    def _diagnostic_record_matrix(self, prefix, matrices):
        if not self.diagnostics_enabled or self._diagnostic_current is None:
            return None
        mats = numpy.asarray(matrices)
        self._diagnostic_record_array(prefix, mats)
        if mats.ndim != 3:
            return None
        max_real_eigs = numpy.full(mats.shape[0], numpy.nan, dtype=numpy.float64)
        for iw, matrix in enumerate(mats):
            if numpy.all(numpy.isfinite(matrix)):
                try:
                    max_real_eigs[iw] = numpy.max(numpy.linalg.eigvals(matrix).real)
                except numpy.linalg.LinAlgError:
                    pass
        finite = numpy.isfinite(max_real_eigs)
        if numpy.any(finite):
            self._diagnostic_current[f"{prefix}_max_real_eig"] = float(
                numpy.max(max_real_eigs[finite])
            )
            self._diagnostic_current[f"{prefix}_mean_max_real_eig"] = float(
                numpy.mean(max_real_eigs[finite])
            )
            self._diagnostic_current[f"{prefix}_argmax_real_eig"] = float(
                numpy.argmax(numpy.where(finite, max_real_eigs, -numpy.inf))
            )
        return None

    def _diagnostic_record_walker_state(self, prefix, walkers):
        self._diagnostic_record_array(f"{prefix}_coherent_shift", walkers.coherent_state_shift)
        self._diagnostic_record_array(f"{prefix}_phia", walkers.phia)
        if getattr(walkers, "ndown", 0) > 0:
            self._diagnostic_record_array(f"{prefix}_phib", walkers.phib)
        if hasattr(walkers, "weight"):
            self._diagnostic_record_array(f"{prefix}_raw_weight", walkers.weight)
        if hasattr(walkers, "weight_log"):
            weight_log = numpy.asarray(walkers.weight_log, dtype=numpy.complex128)
            real_log = weight_log.real
            self._diagnostic_record_real(f"{prefix}_weight_log", real_log)
            finite = numpy.isfinite(real_log)
            if self._diagnostic_current is not None and numpy.any(finite):
                self._diagnostic_current[f"{prefix}_weight_log_spread"] = float(
                    numpy.max(real_log[finite]) - numpy.min(real_log[finite])
                )
        if hasattr(walkers, "phase"):
            phase = numpy.asarray(walkers.phase)
            active = numpy.isfinite(phase) & (numpy.abs(phase) > 0.0)
            if self._diagnostic_current is not None and numpy.any(active):
                unit_phase = phase[active] / numpy.abs(phase[active])
                self._diagnostic_current[f"{prefix}_phase_resultant"] = float(
                    numpy.abs(numpy.mean(unit_phase))
                )
        if hasattr(walkers, "ovlp"):
            self._diagnostic_record_array(f"{prefix}_overlap", walkers.ovlp)
        return None

    def _diagnostic_check_finite(self, stage, walkers):
        if not self.diagnostics_enabled or self._diagnostic_failure:
            return None
        walker_index = self._diagnostic_first_nonfinite_walker(walkers)
        if walker_index is not None:
            self._diagnostic_record_failure(stage, None, walkers, walker_index)
        return None

    def _diagnostic_first_nonfinite_walker(self, walkers):
        for name in ("coherent_state_shift", "phia", "phib", "weight", "phase", "weight_log", "ovlp"):
            if not hasattr(walkers, name):
                continue
            array = numpy.asarray(getattr(walkers, name))
            if array.size == 0:
                continue
            finite = numpy.isfinite(array)
            if not numpy.all(finite):
                if array.ndim == 0:
                    return 0
                per_walker = finite.reshape(array.shape[0], -1)
                bad = numpy.where(~numpy.all(per_walker, axis=1))[0]
                if bad.size > 0:
                    return int(bad[0])
        return None

    def _diagnostic_record_failure(self, stage, exc=None, walkers=None, walker_index=None):
        if not self.diagnostics_enabled or self._diagnostic_failure:
            return None
        if walker_index is None and walkers is not None:
            walker_index = self._diagnostic_first_nonfinite_walker(walkers)
        self._diagnostic_failure = {
            "stage": stage,
            "exception": "" if exc is None else repr(exc),
            "step": int(self._diagnostic_step),
            "walker_index": -1 if walker_index is None else int(walker_index),
        }
        if walkers is not None:
            self._diagnostic_last_arrays.update(
                {
                    "failure_coherent_state_shift": numpy.asarray(
                        walkers.coherent_state_shift
                    ).copy(),
                    "failure_phia": numpy.asarray(walkers.phia).copy(),
                }
            )
            for name in ("phib", "weight", "phase", "weight_log", "ovlp"):
                if hasattr(walkers, name):
                    self._diagnostic_last_arrays[f"failure_{name}"] = numpy.asarray(
                        getattr(walkers, name)
                    ).copy()
        if self._diagnostic_current is not None:
            self._diagnostic_records.append(self._diagnostic_current)
            self._diagnostic_current = None
        self.save_diagnostics(reason="failure")
        return None

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
        step_index = self._diagnostic_annihilation_count
        self._diagnostic_record_matrix(
            f"annihilation_{step_index}_up_generator", -step_size * h_eff[0]
        )
        if walkers.ndown > 0:
            self._diagnostic_record_matrix(
                f"annihilation_{step_index}_down_generator", -step_size * h_eff[1]
            )
        self._diagnostic_annihilation_count += 1
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
        self._diagnostic_record_array("dZ", dZ)
        self._diagnostic_record_array("dZ_creation", dZ_creation)
        self._diagnostic_record_array("delta_lambda", delta_lambda)
        self._diagnostic_record_matrix("creation_generator", -creation)
        if self.diagnostics_enabled:
            self._diagnostic_last_arrays.update(
                {
                    "last_delta_lambda": delta_lambda.copy(),
                    "last_dZ": dZ.copy(),
                    "last_dZ_creation": dZ_creation.copy(),
                    "last_creation_generator": (-creation).copy(),
                }
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
        self._diagnostic_begin_step(walkers)
        try:
            start_time = time.time()
            ovlp = trial.calc_overlap(walkers)
            walkers.ovlp = ovlp
            self._diagnostic_record_array("old_overlap", ovlp)
            self._diagnostic_check_finite("old_overlap", walkers)
            synchronize()
            self.timer.tovlp += time.time() - start_time

            log_likelihood = self.propagate(walkers, hamiltonian, trial)

            start_time = time.time()
            ovlp_new = trial.calc_overlap(walkers)
            walkers.ovlp = ovlp_new
            self._diagnostic_record_array("new_overlap", ovlp_new)
            self._diagnostic_check_finite("new_overlap", walkers)
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
            self._diagnostic_check_finite("weight_update", walkers)
            synchronize()
            self.timer.tupdate += time.time() - start_time
            self._diagnostic_finish_step(walkers)
        except Exception as exc:
            self._diagnostic_record_failure("propagate_walkers_exception", exc, walkers)
            raise

    def propagate(self, walkers, hamiltonian, trial):
        r"""Apply one importance-sampled symmetric split step."""
        start_time = time.time()
        self.apply_phonon_damping(walkers, hamiltonian, 0.5 * self.dt)
        self._diagnostic_check_finite("after_first_phonon_half", walkers)
        self.apply_annihilation_step(walkers, hamiltonian, 0.5 * self.dt)
        self._diagnostic_check_finite("after_first_annihilation_half", walkers)

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
        self._diagnostic_check_finite("after_creation", walkers)

        self.apply_annihilation_step(walkers, hamiltonian, 0.5 * self.dt)
        self._diagnostic_check_finite("after_second_annihilation_half", walkers)
        self.apply_phonon_damping(walkers, hamiltonian, 0.5 * self.dt)
        self._diagnostic_check_finite("after_second_phonon_half", walkers)

        synchronize()
        self.timer.tgemm += time.time() - start_time
        return log_likelihood

    def sample_biased_complex_noise(self, walkers, hamiltonian, trial, step_size):
        r"""Draw ``dZ = a * step_size + dW`` and return its log likelihood ratio."""
        delta_lambda, A, B = self.construct_split_gauge(walkers, hamiltonian, trial)
        if self.creation_coefficients_are_zero(delta_lambda):
            dZ = numpy.zeros((walkers.nwalkers, hamiltonian.N), dtype=numpy.complex128)
            log_likelihood = numpy.zeros(walkers.nwalkers, dtype=numpy.float64)
            self._diagnostic_record_array("dZ", dZ)
            self._diagnostic_record_array("dZ_creation", dZ)
            self._diagnostic_record_real("log_likelihood", log_likelihood)
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
        self._diagnostic_record_array("drift", drift)
        self._diagnostic_record_array("dW", dW)
        self._diagnostic_record_array("dZ_base", dZ_base)
        self._diagnostic_record_array("dZ", dZ)
        self._diagnostic_record_array("dZ_creation", dZ_creation)
        self._diagnostic_record_real("log_likelihood", log_likelihood)
        if self.diagnostics_enabled:
            self._diagnostic_last_arrays.update(
                {
                    "last_drift": drift.copy(),
                    "last_dW": dW.copy(),
                    "last_dZ_base": dZ_base.copy(),
                    "last_log_likelihood": log_likelihood.copy(),
                }
            )
        return dZ, dZ_creation, log_likelihood, delta_lambda

    def construct_split_gauge(self, walkers, hamiltonian, trial):
        zeros = numpy.zeros((walkers.nwalkers, hamiltonian.N), dtype=numpy.complex128)
        if self.split_gauge == "static":
            self._diagnostic_record_array("delta_lambda", zeros)
            return zeros, None, None

        A, B = self.construct_ito_log_derivatives(walkers, trial)
        delta_lambda = self.split_gauge_scale * (B + self.split_gauge_q * A.conj())
        delta_lambda = self.apply_split_gauge_bound(delta_lambda)
        B_gauged = B - delta_lambda
        self._diagnostic_record_array("A", A)
        self._diagnostic_record_array("B", B)
        self._diagnostic_record_array("delta_lambda", delta_lambda)
        self._diagnostic_record_array("B_gauged", B_gauged)
        if self.diagnostics_enabled:
            self._diagnostic_last_arrays.update(
                {
                    "last_A": A.copy(),
                    "last_B": B.copy(),
                    "last_B_gauged": B_gauged.copy(),
                    "last_delta_lambda": delta_lambda.copy(),
                }
            )
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
        overlap_ratio_log = numpy.full(walkers.nwalkers, numpy.nan, dtype=numpy.float64)
        overlap_ratio_log[nonzero] = numpy.log(ratio_abs[nonzero])
        self._diagnostic_record_real("log_likelihood", log_likelihood)
        self._diagnostic_record_array("overlap_ratio", ratio)
        self._diagnostic_record_real("overlap_ratio_log", overlap_ratio_log)
        self._diagnostic_record_real("weight_increment_log", log_abs)
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

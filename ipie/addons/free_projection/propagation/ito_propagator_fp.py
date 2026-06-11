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

from ipie.addons.eph.propagation.ito_propagator import EulerItoPropagator
from ipie.estimators.greens_function_single_det import gab_mod_ovlp
from ipie.utils.backend import synchronize


def _coherent_state_overlap(beta_bra, beta_ket, convention):
    if convention == "normalized":
        exponent = -0.5 * (
            numpy.sum(numpy.abs(beta_bra) ** 2) + numpy.sum(numpy.abs(beta_ket) ** 2)
        )
        exponent += numpy.sum(beta_bra.conj() * beta_ket)
        return numpy.exp(exponent)
    return numpy.exp(numpy.sum(beta_bra.conj() * beta_ket))


def _electronic_overlap(psia_bra, psia_ket, psib_bra=None, psib_ket=None):
    overlap = numpy.linalg.det(psia_bra.conj().T.dot(psia_ket))
    if psib_bra is not None and psib_ket is not None:
        overlap *= numpy.linalg.det(psib_bra.conj().T.dot(psib_ket))
    return overlap


def _contract_eph_mean_field(g_tensor, ga, gb):
    return numpy.einsum("ijk,ij->k", g_tensor, ga + gb)


def _coerce_real_scalar(value, name):
    if value is None:
        return None
    scalar = numpy.asarray(value)
    if scalar.shape != ():
        raise ValueError(f"{name} must be a scalar, not shape {scalar.shape}.")

    scalar = scalar.item()
    imag = numpy.imag(scalar)
    real = numpy.real(scalar)
    if abs(imag) > 1e-10 * max(1.0, abs(real)):
        raise ValueError(f"{name} must be real, not {scalar}.")
    return float(real)


def construct_trial_eph_mean_field(hamiltonian, trial, zero_threshold=1e-12):
    r"""Compute ``<trial|G_mu|trial>/<trial|trial>`` for static subtraction."""
    required = ("g_tensor", "N")
    if any(not hasattr(hamiltonian, attr) for attr in required):
        raise TypeError("Mean-field subtraction requires an electron-phonon Hamiltonian.")

    required = ("psia", "beta_shift", "ndown")
    if any(not hasattr(trial, attr) for attr in required):
        raise TypeError("Mean-field subtraction requires an electron-phonon trial state.")

    g_tensor = numpy.asarray(hamiltonian.g_tensor, dtype=numpy.complex128)
    convention = getattr(trial, "coherent_state_convention", "unnormalized")
    beta = numpy.asarray(trial.beta_shift, dtype=numpy.complex128)
    psia = numpy.asarray(trial.psia, dtype=numpy.complex128)
    psib = numpy.asarray(
        getattr(trial, "psib", numpy.empty((psia.shape[0], 0))), dtype=numpy.complex128
    )

    numerator = numpy.zeros(hamiltonian.N, dtype=numpy.complex128)
    denominator = 0.0j

    if hasattr(trial, "perms") and hasattr(trial, "kcoeffs"):
        perms = numpy.asarray(trial.perms)
        kcoeffs = numpy.asarray(trial.kcoeffs, dtype=numpy.complex128)
        for ibra, perm_bra in enumerate(perms):
            psia_bra = psia[perm_bra, :]
            beta_bra = beta[perm_bra]
            psib_bra = psib[perm_bra, :] if trial.ndown > 0 else None

            for iket, perm_ket in enumerate(perms):
                psia_ket = psia[perm_ket, :]
                beta_ket = beta[perm_ket]
                psib_ket = psib[perm_ket, :] if trial.ndown > 0 else None

                overlap = _electronic_overlap(psia_bra, psia_ket, psib_bra, psib_ket)
                overlap *= _coherent_state_overlap(beta_bra, beta_ket, convention)
                overlap *= kcoeffs[ibra].conj() * kcoeffs[iket]
                if numpy.abs(overlap) < zero_threshold:
                    continue

                ga, _, _ = gab_mod_ovlp(psia_bra, psia_ket)
                if trial.ndown > 0:
                    gb, _, _ = gab_mod_ovlp(psib_bra, psib_ket)
                else:
                    gb = numpy.zeros_like(ga)

                numerator += overlap * _contract_eph_mean_field(g_tensor, ga, gb)
                denominator += overlap
    else:
        psib_bra = psib if trial.ndown > 0 else None
        overlap = _electronic_overlap(psia, psia, psib_bra, psib_bra)
        overlap *= _coherent_state_overlap(beta, beta, convention)
        if numpy.abs(overlap) >= zero_threshold:
            ga, _, _ = gab_mod_ovlp(psia, psia)
            if trial.ndown > 0:
                gb, _, _ = gab_mod_ovlp(psib, psib)
            else:
                gb = numpy.zeros_like(ga)
            numerator += overlap * _contract_eph_mean_field(g_tensor, ga, gb)
            denominator += overlap

    if numpy.abs(denominator) < zero_threshold:
        raise ValueError("Cannot construct mean-field subtraction from a zero-overlap trial.")
    return numerator / denominator


class EulerItoPropagatorFP(EulerItoPropagator):
    r"""Bare Euler--Maruyama coherent-state propagator.

    This implements the proper-complex-noise version of the D2 Ito update

    .. math::
        df_\mu = -\omega_\mu f_\mu d\tau + dZ_\mu,

    .. math::
        d\phi = -\left(h+\sum_\mu f_\mu G_\mu\right)\phi d\tau
                -\sum_\mu G_\mu^\dagger\phi\,dZ_\mu^*,

    with :math:`E[dZ_\mu dZ_\nu]=0` and
    :math:`E[dZ_\mu^* dZ_\nu]=\delta_{\mu\nu}d\tau`.
    """

    def __init__(
        self,
        time_step,
        verbose=False,
        mean_field_subtraction=False,
        mean_field_shift=None,
        reference_energy=None,
    ):
        super().__init__(time_step, verbose=verbose)
        self.mean_field_subtraction = mean_field_subtraction or mean_field_shift is not None
        self._input_mean_field_shift = mean_field_shift
        self.reference_energy = _coerce_real_scalar(reference_energy, "reference_energy")
        self.eph_mean_field = None
        self.g_tensor_residual = None
        self.g_tensor_residual_dagger = None

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None) -> None:
        super().build(hamiltonian, trial=trial, walkers=walkers, mpi_handler=mpi_handler)

        if self._input_mean_field_shift is not None:
            mean_field = numpy.asarray(self._input_mean_field_shift, dtype=numpy.complex128)
            if mean_field.shape != (hamiltonian.N,):
                raise ValueError(
                    "mean_field_shift must have shape "
                    f"({hamiltonian.N},), not {mean_field.shape}."
                )
        elif self.mean_field_subtraction:
            mean_field = construct_trial_eph_mean_field(hamiltonian, trial)
        else:
            mean_field = numpy.zeros(hamiltonian.N, dtype=numpy.complex128)

        self.eph_mean_field = mean_field
        identity = numpy.eye(hamiltonian.g_tensor.shape[0], dtype=numpy.complex128)
        self.g_tensor_residual = numpy.asarray(
            hamiltonian.g_tensor, dtype=numpy.complex128
        ).copy()
        self.g_tensor_residual -= identity[:, :, None] * self.eph_mean_field[None, None, :]
        self.g_tensor_residual_dagger = numpy.swapaxes(self.g_tensor_residual.conj(), 0, 1)

    def propagate(self, walkers, hamiltonian, trial) -> None:
        r"""Apply one Euler--Maruyama step with static mean-field subtraction."""
        start_time = time.time()

        shift_old = walkers.coherent_state_shift.copy()
        dZ = self.sample_complex_noise(walkers, hamiltonian)

        h_eff = self.construct_annihilation_matrix(shift_old, hamiltonian)
        creation = self.construct_creation_matrix(dZ, hamiltonian)

        walkers.phia = self.apply_electronic_euler_step(walkers.phia, h_eff[0], creation)
        if walkers.ndown > 0:
            walkers.phib = self.apply_electronic_euler_step(walkers.phib, h_eff[1], creation)

        walkers.coherent_state_shift += (
            -self.dt * hamiltonian.w0 * shift_old
            - self.dt * self.eph_mean_field.conj()
            + dZ
        )

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def construct_creation_matrix(self, dZ, hamiltonian):
        r"""Build ``sum_mu dZ_mu^* (G_mu^\dagger - gbar_mu^* I)``."""
        return numpy.einsum("ijk,nk->nij", self.g_tensor_residual_dagger, dZ.conj())

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=None) -> None:
        self.propagate(walkers, hamiltonian, trial)

        start_time = time.time()
        walkers.ovlp = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp=None, ovlp_new=walkers.ovlp, eshift=eshift)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def set_reference_energy(self, reference_energy) -> None:
        """Set the constant energy subtracted from the projector Hamiltonian."""
        self.reference_energy = _coerce_real_scalar(reference_energy, "reference_energy")

    def _resolve_reference_energy(self, eshift):
        if self.reference_energy is not None:
            return self.reference_energy
        return _coerce_real_scalar(eshift, "eshift")

    def update_weight(self, walkers, ovlp=None, ovlp_new=None, eshift=None) -> None:
        """Apply the scalar factor from projecting with ``H - E_ref``."""
        reference_energy = self._resolve_reference_energy(eshift)
        if reference_energy is None or reference_energy == 0.0:
            return None

        log_factor = self.dt * reference_energy
        walkers.weight *= numpy.exp(log_factor)
        if hasattr(walkers, "weight_log"):
            walkers.weight_log += log_factor
        return None

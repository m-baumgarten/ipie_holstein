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
from typing import Tuple, Optional, List

from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import circ_perm
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp

import jax
import jax.numpy as npj
from jax import jit


def _calc_greens_function(
    nwalkers: int,
    nsites: int,
    ndown: int,
    ovlp_perm: np.ndarray,
    ovlp_tot: np.ndarray,
    perms: np.ndarray,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: np.ndarray,
    phib: np.ndarray,
) -> List[np.ndarray]:

    Ga = npj.zeros((nwalkers, nsites, nsites), dtype=np.complex128)
    Gb = npj.zeros_like(Ga)

    for ovlp, perm in zip(ovlp_perm.T, perms):
        inv_Oa = npj.linalg.inv(npj.einsum("ie,nif->nef", psia[perm, :].conj(), phia))
        Ga = Ga + npj.einsum("nie,nef,jf,n->nji", phia, inv_Oa, psia[perm].conj(), ovlp)

        if ndown > 0:
            inv_Ob = npj.linalg.inv(npj.einsum("ie,nif->nef", psib[perm, :].conj(), phib))
            Gb += npj.einsum("nie,nef,jf,n->nji", phib, inv_Ob, psib[perm].conj(), ovlp)

    Ga = npj.einsum("nij,n->nij", Ga, 1 / ovlp_tot)
    if ndown > 0:
        Gb = npj.einsum("nij,n->nij", Gb, 1 / ovlp_tot)

    return [Ga, Gb]


def _calc_electronic_overlap_perms(
    nwalkers: int,
    kcoeffs: np.ndarray,
    ndown: int,
    nperms: int,
    perms: np.ndarray,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: np.ndarray,
    phib: np.ndarray,
) -> np.ndarray:
    el_ovlp = npj.zeros((nwalkers, nperms), dtype=npj.complex128)
    for ip, (coeff, perm) in enumerate(zip(kcoeffs, perms)):
        ovlp_a = npj.einsum("mi,wmj->wij", psia[perm, :].conj(), phia, optimize=True)
        sign_a, log_ovlp_a = npj.linalg.slogdet(ovlp_a)

        if ndown > 0:
            ovlp_b = npj.einsum("mi,wmj->wij", psib[perm, :].conj(), phib, optimize=True)
            sign_b, log_ovlp_b = npj.linalg.slogdet(ovlp_b)
            ot = sign_a * sign_b * npj.exp(log_ovlp_a + log_ovlp_b)
        else:
            ot = sign_a * npj.exp(log_ovlp_a)

        ot *= coeff.conj()
        el_ovlp = el_ovlp.at[:, ip].set(ot)

    return el_ovlp


def _calc_electronic_overlap_perms_single_walker(
    kcoeffs: np.ndarray,
    ndown: int,
    nperms: int,
    perms: np.ndarray,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: np.ndarray,
    phib: np.ndarray,
) -> np.ndarray:
    el_ovlp = npj.zeros((nperms), dtype=npj.complex128)
    for ip, (coeff, perm) in enumerate(zip(kcoeffs, perms)):
        ovlp_a = npj.einsum("mi,mj->ij", psia[perm, :].conj(), phia, optimize=True)
        sign_a, log_ovlp_a = npj.linalg.slogdet(ovlp_a)

        if ndown > 0:
            ovlp_b = npj.einsum("mi,mj->ij", psib[perm, :].conj(), phib, optimize=True)
            sign_b, log_ovlp_b = npj.linalg.slogdet(ovlp_b)
            ot = sign_a * sign_b * npj.exp(log_ovlp_a + log_ovlp_b)
        else:
            ot = sign_a * npj.exp(log_ovlp_a)

        ot *= coeff.conj()
        el_ovlp = el_ovlp.at[ip].set(ot)

    return el_ovlp


def _calc_phonon_overlap_perms(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
) -> np.ndarray:
    ph_ovlp = npj.zeros((phonon_disp.shape[0], nperms), dtype=npj.complex128)
    for ip, perm in enumerate(perms):
        ph_ov = npj.exp(
            -(0.5 * m * w0) * (phonon_disp - beta_shift[perm].real) ** 2
            - 1j * m * w0 * phonon_disp * beta_shift[perm].imag
            + 1j * beta_shift[perm].real * beta_shift[perm].imag
        )
        ph_ovlp = ph_ovlp.at[:, ip].set(npj.prod(ph_ov, axis=1))
    return ph_ovlp


def _calc_phonon_overlap_perms_single_walker(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
) -> np.ndarray:
    ph_ovlp = npj.zeros((nperms), dtype=npj.complex128)
    for ip, perm in enumerate(perms):
        ph_ov = npj.exp(
            -(0.5 * m * w0) * (phonon_disp - beta_shift[perm].real) ** 2
            - 1j * m * w0 * phonon_disp * beta_shift[perm].imag
            + 1j * beta_shift[perm].real * beta_shift[perm].imag
        )
        ph_ovlp = ph_ovlp.at[ip].set(npj.prod(ph_ov))
    return ph_ovlp


def _calc_overlap_perms(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    nwalkers: int,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    ph_ovlp_perm = _calc_phonon_overlap_perms(phonon_disp, nperms, perms, m, w0, beta_shift)
    el_ovlp_perm = _calc_electronic_overlap_perms(
        nwalkers, kcoeffs, ndown, nperms, perms, psia, phia, psib, phib
    )
    ovlp_perm = ph_ovlp_perm * el_ovlp_perm
    return ovlp_perm


def _calc_overlap_perms_single_walker(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    ph_ovlp_perm = _calc_phonon_overlap_perms_single_walker(
        phonon_disp, nperms, perms, m, w0, beta_shift
    )
    el_ovlp_perm = _calc_electronic_overlap_perms_single_walker(
        kcoeffs, ndown, nperms, perms, psia, phia, psib, phib
    )
    ovlp_perm = ph_ovlp_perm * el_ovlp_perm
    return ovlp_perm


def _calc_overlap(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    nwalkers: int,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    ovlp_perms = _calc_overlap_perms(
        phonon_disp,
        nperms,
        perms,
        m,
        w0,
        beta_shift,
        nwalkers,
        kcoeffs,
        ndown,
        psia,
        phia,
        psib,
        phib,
    )
    return np.sum(ovlp_perms, axis=1)


def _calc_overlap_single_walker(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    ovlp_perms = _calc_overlap_perms_single_walker(
        phonon_disp,
        nperms,
        perms,
        m,
        w0,
        beta_shift,
        kcoeffs,
        ndown,
        psia,
        phia,
        psib,
        phib,
    )
    return np.sum(ovlp_perms)

def _calc_overlap_abs_single_walker(
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    ovlp_perms = _calc_overlap_perms_single_walker(
        phonon_disp,
        nperms,
        perms,
        m,
        w0,
        beta_shift,
        kcoeffs,
        ndown,
        psia,
        phia,
        psib,
        phib,
    )
    return npj.abs(np.sum(ovlp_perms)).astype(npj.complex128)

jit_grad = jax.jit(
#    jax.grad(_calc_overlap_single_walker, holomorphic=True), #static_argnums=(0, 2, 4, 5, 7, 9)
    jax.grad(_calc_overlap_abs_single_walker, holomorphic=True),
    static_argnums=(1, 3, 4, 7),
)
#_calc_overlap_jitted = jit(_calc_overlap, static_argnums=(1, 3, 4, 6, 8))
_calc_overlap_jitted = jit(_calc_overlap_, static_argnums=(1, 3, 4, 6, 8))

def _calc_phonon_gradient(
    nsites: int,
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    nwalkers: int,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    grad = npj.zeros((nwalkers, nsites), dtype=np.complex128)
    for w in range(nwalkers):
        grad = grad.at[w, :].set(
            npj.array(
                #            jax.grad(_calc_overlap_single_walker, holomorphic=True)(
                jit_grad(
                    phonon_disp[w],
                    nperms,
                    perms,
                    m,
                    w0,
                    beta_shift,
                    kcoeffs,
                    ndown,
                    psia,
                    phia[w],
                    psib,
                    phib[w],
                ),
                dtype=npj.complex128,
            )
        )
    ovlp = _calc_overlap_jitted(
        phonon_disp,
        nperms,
        perms,
        m,
        w0,
        beta_shift,
        nwalkers,
        kcoeffs,
        ndown,
        psia,
        phia,
        psib,
        phib,
    )
    #    grad = grad.at[w, :].set(grad[w, :] / ovlp)
    grad = npj.einsum("ni,n->ni", grad, 1 / ovlp)
    return grad


jit_hess = jax.jit(
    jax.hessian(_calc_overlap_single_walker, holomorphic=True), static_argnums=(1, 3, 4, 7)
)


def _calc_phonon_laplacian(
    nsites: int,
    phonon_disp: np.ndarray,
    nperms: int,
    perms: np.ndarray,
    m: float,
    w0: float,
    beta_shift: np.ndarray,
    nwalkers: int,
    kcoeffs: np.ndarray,
    ndown: int,
    psia: np.ndarray,
    phia: np.ndarray,
    psib: Optional[np.ndarray] = None,
    phib: Optional[np.ndarray] = None,
) -> np.ndarray:
    laplacian = npj.zeros((nwalkers), dtype=np.complex128)  # npj
    for w in range(nwalkers):
        hessian = npj.array(
            #            jax.hessian(_calc_overlap_single_walker, holomorphic=True)(
            jit_hess(
                phonon_disp[w],
                nperms,
                perms,
                m,
                w0,
                beta_shift,
                kcoeffs,
                ndown,
                psia,
                phia[w],
                psib,
                phib[w],
            ),
            dtype=npj.complex128,
        )
        laplacian = laplacian.at[w].set(npj.sum(hessian.diagonal()))

    ovlp = _calc_overlap(
        phonon_disp,
        nperms,
        perms,
        m,
        w0,
        beta_shift,
        nwalkers,
        kcoeffs,
        ndown,
        psia,
        phia,
        psib,
        phib,
    )

    laplacian = laplacian / ovlp
    #        laplacian = laplacian.at[w].set(laplacian[w] / ovlp)
    return laplacian


_calc_greens_function_jitted = jit(_calc_greens_function, static_argnums=(0, 1, 2))
_calc_electronic_overlap_perms_jitted = jit(
    _calc_electronic_overlap_perms, static_argnums=(0, 2, 3)
)
_calc_phonon_overlap_perms_jitted = jit(_calc_phonon_overlap_perms, static_argnums=(1, 3, 4))
_calc_phonon_gradient_jitted = jit(_calc_phonon_gradient, static_argnums=(0, 2, 4, 5, 7, 9))
_calc_phonon_laplacian_jitted = jit(_calc_phonon_laplacian, static_argnums=(0, 2, 4, 5, 7, 9))
_calc_overlap_perms_jitted = jit(_calc_overlap_perms, static_argnums=(1, 3, 4, 6, 8))
#_calc_overlap_jitted = jit(_calc_overlap, static_argnums=(1, 3, 4, 6, 8))

_calc_electronic_overlap_perms_single_walker_jitted = jit(
    _calc_electronic_overlap_perms, static_argnums=(1, 2)
)
_calc_phonon_overlap_perms_single_walker_jitted = jit(
    _calc_phonon_overlap_perms_single_walker, static_argnums=(1, 3, 4)
)
_calc_overlap_perms_single_walker_jitted = jit(
    _calc_overlap_perms_single_walker, static_argnums=(1, 3, 4, 7)
)
_calc_overlap_single_walker_jitted = jit(_calc_overlap_single_walker, static_argnums=(1, 3, 4, 7))


class ToyozawaTrial(CoherentStateTrial):
    r"""The Toyozawa trial

    .. math::
        |\Psi(\kappa)\rangle = \sum_n e^{i \kappa n} \sum_{n_1} \alpha_{n_1}^{\kappa}
        a_{n_1}^{\dagger} \exp(-\sum_{n_2} (\beta^\kappa_{n_2 - n} b_{n_2}^{\dagger}
        - \beta^{\kappa^*}_{n_2 - n} b_{n_2}))|0\rangle

    developed by `Toyozawa <https://doi.org/10.1143/PTP.26.29>`_ is translationally
    invariant and reliable offers a good approximation to the polaron ground state
    for most parameter regimes of the Holstein Model. Here, :math:`\alpha,\beta`are
    varaitional parameters, and :math:`|0\rangle` is the total vacuum state.
    For a 1D Holstein chain this reduces to a superposition of cyclically `CoherentState`
    type trials.
    More details may be found in `Zhao et al. <https://doi.org/10.1063/1.474667>`_.

    Attributes
    ----------
    perms : :class:`np.ndarray`
        Rows of this matrix corresponds to cyclic permutations of `range(nsites)`
    nperms : :class:`int`
        Number of permutations in `perms`
    """

    def __init__(
        self,
        wavefunction: np.ndarray,
        w0: float,
        num_elec: Tuple[int, int],
        num_basis: int,
        K: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(wavefunction, w0, num_elec, num_basis, verbose=verbose)
        self.perms = circ_perm(np.arange(self.nbasis))
        self.nperms = self.perms.shape[0]
        self.kcoeffs = np.exp(1j * K * np.arange(self.nbasis))
        self.K = K
        self.phase = np.exp(1j * np.angle(self.beta_shift))

    def calc_energy(self, ham, zero_th=1e-12):
        r"""Computes the variational energy of the trial, i.e.

        .. math::
            E_T = \frac{\langle\Psi_T|\hat{H}|\Psi_T\rangle}{\langle\Psi_T|\Psi_T\rangle}.

        As the Toyozawa trial wavefunction is a superposition of coherent state trials
        the evaluation of :math:`E_T` a naive implementation would scale quadratically
        with the number of sites. Here, we exploit the translational symmetry of the
        wavefunction to obtain linear scaling.

        Parameters
        ----------
        ham:
            Hamiltonian

        Returns
        -------
        etrial : :class:`float`
            Trial energy
        """
        num_energy = 0.0
        denom = 0.0
        num_ph_energy = 0.0
        # Recover beta from expected position <X> we store as beta_shift
        beta0 = self.beta_shift * np.sqrt(0.5 * ham.m * ham.w0)
        for ip, (coeff, perm) in enumerate(zip(self.kcoeffs, self.perms)):
            psia_i = self.psia[perm, :]
            beta_i = beta0[perm]

            if self.ndown > 0:
                psib_i = self.psib[perm, :]
                ov = (
                    np.linalg.det(self.psia.conj().T.dot(psia_i))
                    * np.linalg.det(self.psib.conj().T.dot(psib_i))
                    * np.prod(
                        np.exp(
                            -0.5 * (np.abs(beta0) ** 2 + np.abs(beta_i) ** 2)
                            + beta0.conj() * beta_i
                        )
                    )
                )
            else:
                ov = np.linalg.det(self.psia.conj().T.dot(psia_i)) * np.prod(
                    np.exp(
                        -0.5 * (np.abs(beta0) ** 2 + np.abs(beta_i) ** 2) + beta0.conj() * beta_i
                    )
                )
            ov *= self.kcoeffs[0].conj() * coeff

            if np.abs(ov) < zero_th:
                continue

            if ip != 0:
                ov = ov * (self.nbasis - ip) * 2
            else:
                ov = ov * self.nbasis

            Ga_i, _, _ = gab_mod_ovlp(self.psia, psia_i)
            if self.ndown > 0:
                Gb_i, _, _ = gab_mod_ovlp(self.psib, psib_i)
            else:
                Gb_i = np.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]

            kinetic = np.sum(ham.T[0] * G_i[0] + ham.T[1] * G_i[1])
            e_ph = ham.w0 * np.sum(beta0.conj() * beta_i)
            rho = ham.g_tensor * (G_i[0] + G_i[1])
            e_eph = np.sum(np.dot(rho, beta0.conj() + beta_i))
            
            num_energy += np.real((kinetic + e_ph + e_eph) * ov)
            num_ph_energy += np.real(e_ph * ov)
            denom += np.real(ov)

        etrial = num_energy / denom
        etrial_ph = num_ph_energy / denom
        return etrial, etrial_ph

    def calc_overlap_perm(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes the product of electron and phonon overlaps for each
        permutation :math:`\sigma`,

        .. math::
            \langle \psi_T(\sigma(r))|\psi(\tau)\rangle
            \langle \phi(\sigma(\beta))|X_{\mathrm{w}(\tau)}\rangle.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ovlp_perm : :class:`np.ndarray`
            Product of electron and phonon overlap for each permutation
        """
        ph_ovlp_perm = self.calc_phonon_overlap_perms(walkers)
        el_ovlp_perm = self.calc_electronic_overlap_perms(walkers)
        walkers.ovlp_perm = ph_ovlp_perm * el_ovlp_perm
        return walkers.ovlp_perm

    def calc_overlap(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Sums product of electronic and phonon overlap for each permutation
        over all permutations,

        .. math::
            \sum_\tau \langle \psi_T(\sigma(r))|\psi(\tau)\rangle
            \langle \phi(\sigma(\beta))|X_{\mathrm{w}(\tau)}\rangle.

        Used when evaluating local energy and when updating
        weight.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ovlp: :class:`np.ndarray`
            Sum of product of electron and phonon overlap
        """
        ovlp_perm = np.array(
            _calc_overlap_perms_jitted(
                walkers.phonon_disp,
                self.nperms,
                self.perms,
                self.m,
                self.w0,
                self.beta_shift,
                walkers.nwalkers,
                self.kcoeffs,
                self.ndown,
                self.psia,
                walkers.phia,
                self.psib,
                walkers.phib,
            )
        )
        walkers.ovlp_perm = ovlp_perm
        ovlp = np.sum(ovlp_perm, axis=1)
        return ovlp

    def calc_phonon_overlap_perms(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Updates the walker phonon overlap with each permutation :math:`\tau`,
        i.e. :math:`\langle\phi(\tau(\beta))|X_{\mathrm{w}}\rangle` and stores
        it in `walkers.ph_ovlp`.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_ovlp_perm : :class:`np.ndarray`
            Overlap of walker with permuted coherent states
        """
        return np.array(
            _calc_phonon_overlap_perms_jitted(
                walkers.phonon_disp, self.nperms, self.perms, self.m, self.w0, self.beta_shift
            )
        )

    def calc_phonon_overlap(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Sums walker phonon overlaps with permuted coherent states over all
        permutations,

        .. math::
            \sum_\tau \langle \phi(\tau(\beta)) | X_{\mathrm{w}} \rangle

        to get total phonon overlap. This is only used to correct
        for the importance sampling in propagate_phonons.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_ovlp : :class:`np.ndarray`
            Total walker phonon overlap
        """
        ph_ovlp_perm = np.array(
            _calc_phonon_overlap_perms_jitted(
                walkers.phonon_disp, self.nperms, self.perms, self.m, self.w0, self.beta_shift
            )
        )
        walkers.ph_ovlp = ph_ovlp_perm
        ph_ovlp = np.sum(ph_ovlp_perm, axis=1)
        return ph_ovlp

    def calc_phonon_gradient(self, walkers) -> np.ndarray:
        """"""
        return np.array(_calc_phonon_gradient(
            self.nsites,
            walkers.phonon_disp,
            self.nperms,
            self.perms,
            self.m,
            self.w0,
            self.beta_shift,
            walkers.nwalkers,
            self.kcoeffs,
            self.ndown,
            self.psia,
            walkers.phia,
            self.psib,
            walkers.phib,
        ))

    def calc_phonon_laplacian(
        self, walkers: EPhWalkers, dummy: None
    ) -> np.ndarray:  # TODO remove dummy if works
        """"""
        laplacian = _calc_phonon_laplacian(
            self.nsites,
            walkers.phonon_disp,
            self.nperms,
            self.perms,
            self.m,
            self.w0,
            self.beta_shift,
            walkers.nwalkers,
            self.kcoeffs,
            self.ndown,
            self.psia,
            walkers.phia,
            self.psib,
            walkers.phib,
        )
        return laplacian

    def calc_phonon_laplacian_importance(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes phonon Laplacian via `calc_phonon_laplacian` with weighting
        by pure phonon overlap. This is only utilized in the importance sampling
        of the DMC procedure.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_lapl : :class:`np.ndarray`
            Phonon Laplacian weigthed by phonon overlaps
        """
        return self.calc_phonon_laplacian(walkers, walkers.ovlp_perm)

    def calc_phonon_laplacian_locenergy(self, walkers: EPhWalkers) -> np.ndarray:
        """Computes phonon Laplacian using total overlap weights as required in
        local energy evaluation.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_lapl : :class:`np.ndarray`
            Phonon Laplacian weigthed by total overlaps
        """
        return self.calc_phonon_laplacian(walkers, walkers.ovlp_perm)

    def calc_electronic_overlap_perms(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Calculates the electronic overlap of each walker with each permuted
        Slater determinant :math:`|\Phi_T(\tau(r_i))\rangle` of the trial,

        .. math::
            \langle \Phi_T(\tau(r_i))|\psi_w\rangle = \mathrm{det(U^{\dagger}V)},

        where :math:`U,V` parametrized the two Slater determinants.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        el_ovlp_perm : :class:`np.ndarray`
            Electronic overlap of each permuted Slater Determiant with walkers
        """
        el_ovlp = _calc_electronic_overlap_perms_jitted(
            walkers.nwalkers,
            self.kcoeffs,
            self.ndown,
            self.nperms,
            self.perms,
            self.psia,
            walkers.phia,
            self.psib,
            walkers.phib,
        )
        walkers.el_ovlp = el_ovlp
        return el_ovlp

    def calc_electronic_overlap(self, walkers: EPhWalkers) -> np.ndarray:
        """Sums walkers.el_ovlp over permutations to obtain total electronic
        overlap of trial with walkers.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        el_ovlp : :class:`np.ndarray`
            Electronic overlap of trial with walkers
        """
        el_ovlp_perms = _calc_electronic_overlap_perms_jitted(
            walkers.nwalkers,
            self.kcoeffs,
            self.ndown,
            self.nperms,
            self.perms,
            self.psia,
            walkers.phia,
            self.psib,
            walkers.phib,
        )
        el_ovlp = np.sum(el_ovlp_perms, axis=1)
        return el_ovlp

    def calc_greens_function(self, walkers: EPhWalkers, build_full=True) -> np.ndarray:
        greens_function = _calc_greens_function_jitted(
            walkers.nwalkers,
            self.nsites,
            self.ndown,
            walkers.ovlp_perm,
            walkers.ovlp,
            self.perms,
            self.psia,
            walkers.phia,
            self.psib,
            walkers.phib,
        )
        return greens_function

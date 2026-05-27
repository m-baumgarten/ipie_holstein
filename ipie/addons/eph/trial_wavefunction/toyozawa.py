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

#from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.toyozawa import circ_perm, circ_perm_1D
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp

class EPhWalkers: ...


def _normalise_by_nonzero_overlap(values: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Divide walker-wise data by overlap, leaving zero-overlap walkers finite."""
    result = np.zeros_like(values)
    mask = np.abs(overlap) > 0.0
    if not np.any(mask):
        return result
    if values.ndim == 1:
        result[mask] = values[mask] / overlap[mask]
    else:
        shape = (-1,) + (1,) * (values.ndim - 1)
        result[mask] = values[mask] / overlap[mask].reshape(shape)
    return result


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
        self.perms = circ_perm_1D(self.nbasis)
        self.nperms = self.perms.shape[0]
        self.kcoeffs = np.exp(1j * K * np.arange(self.nbasis))
        self.K = K

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
        num_ph_energy = 0.0
        denom = 0.0
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
                    * np.prod(np.exp(-0.5 * (np.abs(beta0)**2 + np.abs(beta_i)**2) + beta0.conj() * beta_i))
                )
            else:
                ov = np.linalg.det(self.psia.conj().T.dot(psia_i)) * np.prod(
                    np.exp(-0.5 * (np.abs(beta0)**2 + np.abs(beta_i)**2) + beta0.conj() * beta_i)
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
            e_eph = np.einsum('ijk,ij,k->', ham.g_tensor, G_i[0], beta0.conj() + beta_i)
            #if ip != 0:
            if self.ndown > 0:    
                e_eph += np.einsum('ijk,ij,k->', ham.g_tensor, G_i[1], beta0.conj() + beta_i)

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
        ovlp_perm = self.calc_overlap_perm(walkers)
        ovlp = np.sum(ovlp_perm, axis=1)
#        print('tyoy ovlp:    ', ovlp[966])
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
        for ip, perm in enumerate(self.perms):
            ph_ov = np.exp(
                -(0.5 * self.m * self.w0) * (walkers.phonon_disp - self.beta_shift[perm].real) ** 2
                - 1j * self.m * self.w0 * walkers.phonon_disp * self.beta_shift[perm].imag
                + 1j * self.beta_shift[perm].real * self.beta_shift[perm].imag
            )
            walkers.ph_ovlp[:, ip] = np.prod(ph_ov, axis=1)
#        print('toyo ph ovlp: ', walkers.ph_ovlp[966,:])
#        exit()
        return walkers.ph_ovlp

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
        ph_ovlp_perm = self.calc_phonon_overlap_perms(walkers)
        ph_ovlp = np.sum(ph_ovlp_perm, axis=1)
        return ph_ovlp

    def calc_phonon_gradient(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes the phonon drift,

        .. math::
            D_l = \frac{\nabla_{X_l} \langle \Psi_T | \psi, X \rangle}
                       {\langle \Psi_T | \psi, X \rangle}
                = -m\omega \frac{\sum_\sigma o_\sigma\, (X_l - \sigma(\beta)_l)}
                                 {\sum_\sigma o_\sigma},

        where :math:`o_\sigma = e^{-iK\sigma}\,\langle T_\sigma\alpha | \psi\rangle\,
        \langle\mathrm{coh}(T_\sigma\beta) | X\rangle` is the per-permutation
        full-trial weight stored in `walkers.ovlp_perm`. Because the
        electronic factor and K-projector phase don't depend on X, they
        appear only through the per-permutation weighting, so this
        formula correctly differentiates the *full* trial despite only
        the phonon Gaussian being explicitly differentiated.

        This drift is consumed by the importance-sampling DMC propagator.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        grad : :class:`np.ndarray`
            Phonon drift, shape (nwalkers, nbasis).
        """
        # Defensive: refresh walkers.ovlp_perm to be consistent with the
        # current walkers.phia / phonon_disp. The propagator normally
        # calls calc_overlap before this, but recomputing here is cheap
        # and removes a footgun for direct callers.
        self.calc_overlap_perm(walkers)

        grad = np.zeros_like(walkers.phonon_disp, dtype=np.complex128)
        for ovlp, perm in zip(walkers.ovlp_perm.T, self.perms):
            grad += np.einsum(
                "ni,n->ni",
                (walkers.phonon_disp - self.beta_shift[perm].conj()),
                ovlp,
            )
        grad *= -self.m * self.w0
        return _normalise_by_nonzero_overlap(grad, np.sum(walkers.ovlp_perm, axis=1))

#        grad = np.zeros_like(walkers.phonon_disp, dtype=np.complex128)
#        ovlps = walkers.el_ovlp * np.abs(walkers.ph_ovlp)
#        for ovlp, perm in zip(ovlps.T, self.perms):
#            grad += np.einsum("ni,n->ni", (walkers.phonon_disp - self.beta_shift[perm].conj()), ovlp) # TODO conj correct?
#        grad *= -self.m * self.w0
#        grad = np.einsum("ni,n->ni", grad, 1 / np.sum(ovlps, axis=1))
#        return grad

    def calc_phonon_laplacian(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes the phonon Laplacian, which weights coherent state laplacians
        by overlaps :math:`o(\sigma, r, X, \tau)` passed to this function,

        .. math::
            \sum_\sigma \frac{\nabla_X \langle \phi(\sigma(\beta)) | X(\tau) \rangle}
            {\rangle \phi(\sigma(\beta)) | X(\tau) \rangle}
            = \frac{\sum_sigma ((\sum_i (m \omega (X_i(\tau) - \sigma(\beta)_i))^2) - N m \omega) o(\sigma, r, X, \tau)}
            {\sum_\sigma o(\sigma, r, X, \tau)}.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object
        ovlps : :class:`np.ndarray`
            Overlaps weighting contributions from permuted coherent states

        Returns
        -------
        laplacian : :class:`np.ndarray`
            Phonon Laplacian
        """
        # Defensive: refresh walkers.ovlp_perm. See calc_phonon_gradient.
        self.calc_overlap_perm(walkers)

        laplacian = np.zeros(walkers.nwalkers, dtype=np.complex128)
        for ovlp, perm in zip(walkers.ovlp_perm.T, self.perms):
            arg = (walkers.phonon_disp - self.beta_shift[perm].conj()) * self.m * self.w0
            arg2 = arg**2
            laplacian += (np.sum(arg2, axis=1) - self.nsites * self.m * self.w0) * ovlp
        return _normalise_by_nonzero_overlap(laplacian, np.sum(walkers.ovlp_perm, axis=1))
    
#    def calc_phonon_laplacian_imp(self, walkers: EPhWalkers) -> np.ndarray:
#        r"""Computes the phonon Laplacian, which weights coherent state laplacians
#        by overlaps :math:`o(\sigma, r, X, \tau)` passed to this function,
#
#        .. math::
#            \sum_\sigma \frac{\nabla_X \langle \phi(\sigma(\beta)) | X(\tau) \rangle}
#            {\rangle \phi(\sigma(\beta)) | X(\tau) \rangle}
#            = \frac{\sum_sigma ((\sum_i (m \omega (X_i(\tau) - \sigma(\beta)_i))^2) - N m \omega) o(\sigma, r, X, \tau)}
#            {\sum_\sigma o(\sigma, r, X, \tau)}.
#
#        Parameters
#        ----------
#        walkers : :class:`EPhWalkers`
#            EPhWalkers object
#        ovlps : :class:`np.ndarray`
#            Overlaps weighting contributions from permuted coherent states
#
#        Returns
#        -------
#        laplacian : :class:`np.ndarray`
#            Phonon Laplacian
#        """
#        laplacian = np.zeros(walkers.nwalkers, dtype=np.complex128)
#        ovlps = walkers.el_ovlp * np.abs(walkers.ph_ovlp)
#        for ovlp, perm in zip(ovlps.T, self.perms):
#            arg = (walkers.phonon_disp - self.beta_shift[perm].real) * self.m * self.w0 # TODO conj correct?
#            arg2 = arg**2
#            laplacian += (np.sum(arg2, axis=1) - self.nsites * self.m * self.w0) * ovlp
#        laplacian /= np.sum(ovlps, axis=1)
#        return laplacian

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
        for ip, (coeff, perm) in enumerate(zip(self.kcoeffs, self.perms)):
            ovlp_a = xp.einsum(
                "mi,wmj->wij", self.psia[perm, :].conj(), walkers.phia, optimize=True
            )
            sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)
            
#            print(self.psia, walkers.phia[0])
#            exit()

            if self.ndown > 0:
                ovlp_b = xp.einsum(
                    "mi,wmj->wij", self.psib[perm, :].conj(), walkers.phib, optimize=True
                )
                sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)
                ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walkers.log_shift)
            else:
                ot = sign_a * xp.exp(log_ovlp_a - walkers.log_shift)

            ot *= coeff.conj()

            walkers.el_ovlp[:, ip] = ot
#        print('toyo el_ovlp: ', walkers.el_ovlp[966, :])
#        exit()
        return walkers.el_ovlp

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
        el_ovlp_perms = self.calc_electronic_overlap_perms(walkers)
        el_ovlp = np.sum(el_ovlp_perms, axis=1)
        return el_ovlp

    def calc_greens_function(self, walkers: EPhWalkers, build_full=True) -> np.ndarray:
        r"""Calculates Greens functions by

        .. math::
            G^{\Phi \Psi}_{p\alpha, q\beta}
            = \frac{\sum_{\tau} \delta_{\alpha\beta}(U_\alpha(V^{\dagger}_\alhpa(\tau) U_\alpha) V^{\dagger}_\alpha(\tau)) \langle\Phi_T(\tau(r_i))|\psi_w\rangle}
            {\sum_{\tau} \langle\Phi_T(\tau(r_i))|\psi_w\rangle}

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        G : :class:`list`
            List of Greens functions for :math:`\alpha,\beta` spin spaces.
        """
        # Defensive: refresh walkers.ovlp_perm. See calc_phonon_gradient.
        self.calc_overlap_perm(walkers)

        Ga = np.zeros((walkers.nwalkers, self.nsites, self.nsites), dtype=np.complex128)
        Gb = np.zeros_like(Ga)

        for ip, (ovlp, perm) in enumerate(zip(walkers.ovlp_perm.T, self.perms)):
            inv_Oa = xp.linalg.inv(
                xp.einsum("nie,if->nef", walkers.phia, self.psia[perm, :].conj())
            )
#            Ga += xp.einsum("nie,nef,jf,n->nij", walkers.phia, inv_Oa, self.psia[perm].conj(), ovlp)    #flip ij in the end TODO
#            Ga += xp.einsum("ie,nef,njf,n->nij", self.psia[perm].conj(), inv_Oa, walkers.phia, ovlp)
            walkers.Ga_perm[:,:,:,ip] = xp.einsum("ie,nef,njf,n->nji", self.psia[perm].conj(), inv_Oa, walkers.phia, ovlp)
            Ga += walkers.Ga_perm[:,:,:,ip]
#            print('stuff:   ', self.psia.conj()[perm, :], walkers.phia[966], inv_Oa[966], ovlp[966])
#            print(f'toyo Ga perm {ip}:  ', walkers.Ga_perm[966,:,:,ip], ovlp[966] * inv_Oa[966])

            if self.ndown > 0:
                inv_Ob = xp.linalg.inv(
                    xp.einsum("nie,if->nef", walkers.phib, self.psib[perm, :].conj())
                )
                walkers.Gb_perm[:,:,:,ip] = xp.einsum("nie,nef,jf,n->nji", walkers.phib, inv_Ob, self.psib[perm].conj(), ovlp)
                Gb += walkers.Gb_perm[:,:,:,ip] 

#                Gb += xp.einsum(
#                    "ie,nef,njf,n->nij", self.psib[perm].conj(), inv_Ob, walkers.phia, ovlp
#                )
        
#        print('Ga:  ', Ga, '\nGa_swap:  ', np.swapaxes(Ga, 1, 2))
#        assert (np.allclose(Ga, np.swapaxes(Ga, 1, 2)))
#            print(Ga[0,0,0])
#            exit()
#        print('toyo ovlp in gf:  ', np.sum(walkers.ovlp_perm, axis=(1))[966])
        overlap = np.sum(walkers.ovlp_perm, axis=1)
        Ga = _normalise_by_nonzero_overlap(Ga, overlap)
#        print('sum ovlp perm:   ', np.sum(walkers.ovlp_perm, axis=(1)))
        if self.ndown > 0:
            Gb = _normalise_by_nonzero_overlap(Gb, overlap)
#        print('toyo Ga:  ', Ga[966,:,:])        
#        print('walker disp: ', walkers.phonon_disp[966]) 
#        print('walker phia: ', walkers.phia[966])
        return [Ga, Gb]

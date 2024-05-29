import numpy as np
import plum

from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel

from ipie.systems.generic import Generic
from ipie.utils.backend import arraylib as xp 


def local_energy_generic(
    system: Generic, 
    hamiltonian: GenericEPhModel, 
    walkers: EPhWalkers, 
    trial: EPhTrialWavefunctionBase
) -> np.ndarray:
    r"""Computes the local energy for the Holstein model via
    
    .. math::
        \frac{\langle \Psi_\mathrm{T} | \hat{H} | \Phi_\mathrm{w}\rangle}
        {\langle \Psi_\mathrm{T} | \Phi_\mathrm{w}\rangle},

    where :math:`| \Phi_\mathrm{w}\rangle = \sum_{\tau \in cyclic perms} 
    |\phi(\tau(r))\rangle \otimes |\beta(\tau(X))\rangle`. In this ansatz for
    the walkers :math:`|\beta\rangle` is a coherent state, which corresonds to a 
    by :math:`\beta` displaced vacuum state. 
    This would also work for the Holstein Model, but for now we keep it exploiting
    the sparsity the way it is.
    """

    energy = xp.zeros((walkers.nwalkers, 4), dtype=xp.complex128)

    G = trial.calc_greens_function(walkers)
    walkers.Ga, walkers.Gb = G[0], G[1]

    # Hopping Contribution
    energy[:, 1] = np.sum(hamiltonian.T[0] * G[0], axis=(1, 2))
    if system.ndown > 0:
        energy[:, 1] += np.sum(hamiltonian.T[1] * G[1], axis=(1, 2))

    # Electron-Phonon Contribution
    eph = np.einsum('ijk,nij,nk->n', hamiltonian.g_tensor, G[0], walkers.phonon_disp)
    if system.ndown > 0:
        eph += np.einsum('ijk,nij,nk->n', hamiltonian.g_tensor, G[1], walkers.phonon_disp)
    energy[:, 2] = hamiltonian.const * eph

    # Phonon Contribution
    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0 ** 2 * np.sum(walkers.phonon_disp ** 2, axis=1)
    energy[:, 3] -= 0.5 * hamiltonian.nsites * np.sum(hamiltonian.w0)
    energy[:, 3] -= 0.5 * trial.calc_phonon_laplacian(walkers) / hamiltonian.m

    energy[:, 0] = np.sum(energy[:,1:], axis=1)

    return energy

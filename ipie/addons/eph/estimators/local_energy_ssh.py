import numpy as np

from ipie.addons.eph.hamiltonians.ssh import BondSSHModel
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

from ipie.systems.generic import Generic
from ipie.utils.backend import arraylib as xp 


def local_energy_ssh(
    system: Generic, 
    hamiltonian: BondSSHModel, 
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

    gf = trial.calc_greens_function(walkers)
    walkers.Ga, walkers.Gb = gf[0], gf[1]

    # Hopping Contribution
    energy[:, 1] = np.sum(hamiltonian.T[0] * gf[0], axis=(1, 2))
    if system.ndown > 0:
        energy[:, 1] += np.sum(hamiltonian.T[1] * gf[1], axis=(1, 2))

    # Electron-Phonon Contribution
    displ = np.einsum('ij,nj->ni', hamiltonian.X_connectivity, walkers.phonon_disp)
    arg = np.einsum('nij,nj->ni', hamiltonian.g_tensor * gf[0], displ)
    energy[:, 2] = np.sum(arg, axis=1)
    if system.ndown > 0:
        arg = np.einsum('nij,nj->ni', hamiltonian.g_tensor * gf[1], displ)
        energy[:, 2] += np.sum(arg, axis=1)  
    energy[:, 2] *= hamiltonian.const

    # Phonon Contribution
    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0 ** 2 * np.sum(displ ** 2, axis=1)
    energy[:, 3] -= 0.5 * hamiltonian.nsites * hamiltonian.w0
    energy[:, 3] -= 0.5 * trial.calc_phonon_laplacian_locenergy(walkers) / hamiltonian.m
    
    energy[:, 0] = np.sum(energy[:,1:], axis=1)

    return energy




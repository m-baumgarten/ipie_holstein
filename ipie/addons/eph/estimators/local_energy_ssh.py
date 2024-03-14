import numpy as np
import plum

from ipie.addons.eph.hamiltonians.ssh import BondSSHModel, AcousticSSHModel
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

    G = trial.calc_greens_function(walkers)
    walkers.Ga, walkers.Gb = G[0], G[1]

    # Hopping Contribution
    energy[:, 1] = np.sum(hamiltonian.T[0] * G[0], axis=(1, 2))
    if system.ndown > 0:
        energy[:, 1] += np.sum(hamiltonian.T[1] * G[1], axis=(1, 2))

    # Electron-Phonon Contribution
    offdiags = construct_position(walkers, hamiltonian) 
    eph = hamiltonian.g_tensor * G[0] * offdiags
    if system.ndown > 0:
        eph += hamiltonian.g_tensor * G[1] * offdiags
    energy[:, 2] = np.sum(eph, axis=(1,2))
    energy[:, 2] *= hamiltonian.const

    # Phonon Contribution
    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0 ** 2 * np.sum(walkers.phonon_disp ** 2, axis=1)
    energy[:, 3] -= 0.5 * hamiltonian.nsites * hamiltonian.w0
    energy[:, 3] -= 0.5 * trial.calc_phonon_laplacian_locenergy(walkers) / hamiltonian.m
    
    energy[:, 0] = np.sum(energy[:,1:], axis=1)

#    print(energy)
#    print(walkers.phonon_disp)
#    print('offdiags:    ', offdiags)
#    exit()
    return energy

@plum.dispatch
def construct_position(walkers: EPhWalkers, hamiltonian: BondSSHModel) -> np.ndarray: 
    offdiags = np.zeros((walkers.nwalkers, hamiltonian.nsites, hamiltonian.nsites), dtype=np.complex128)
    for w, disp in enumerate(walkers.phonon_disp):
        offdiags[w,:,:] = np.diag(disp[:-1], -1)
        offdiags[w,:,:] += offdiags[w,:,:].T
    if hamiltonian.pbc:
        offdiags[:,-1,0] = offdiags[:,0,-1] = walkers.phonon_disp[:,-1]
    return offdiags

@plum.dispatch
def construct_position(walkers: EPhWalkers, hamiltonian: AcousticSSHModel) -> np.ndarray:
    displacement = np.einsum('ij,nj->ni', hamiltonian.X_connectivity, walkers.phonon_disp)
    offdiags = np.zeros((walkers.nwalkers, hamiltonian.nsites, hamiltonian.nsites), dtype=np.complex128)
    for w, disp in enumerate(displacement):
        offdiags[w,:,:] = np.diag(disp[:-1], -1)
        offdiags[w,:,:] += offdiags[w,:,:].T
    if hamiltonian.pbc:
        offdiags[:,-1,0] = offdiags[:,0,-1] = displacement[:,-1]
    return offdiags

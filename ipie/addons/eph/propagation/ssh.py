import numpy
import time
import scipy.linalg
import plum

from ipie.addons.eph.hamiltonians.ssh import BondSSHModel, AcousticSSHModel
from ipie.addons.eph.propagation.holstein import HolsteinPropagator
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import synchronize

# TODO change propagte_walkers such that propagate_phonons and electrons
# takes walkers, trial, hamiltonian to just use HolsteinPropagatorFree / other
# Base class

class SSHPropagator(HolsteinPropagator):
    """"""

    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def propagate_electron(
        self, walkers: EPhWalkers, hamiltonian: BondSSHModel, trial: EPhTrialWavefunctionBase
    ) -> None:
        r"""Propagates electronic degrees of freedom via

        .. math::
            \mathrm{e}^{-\Delta \tau (\hat{H}_{\mathrm{el}} \otimes \hat{I}_{\mathrm{ph}} + \hat{H}_{\mathrm{el-ph}})}
            \approx \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2}
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el-ph}}}
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2}.

        This acts on walkers of the form :math:`|\phi(\tau)\rangle \otimes |X(\tau)\rangle`.


        Parameters
        ----------
        walkers :
            Walkers class
        trial :
            Trial class
        """
        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time

        EPh = self.construct_EPh(walkers, hamiltonian)
        expEph = scipy.linalg.expm(self.const * EPh)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum("nij,nje->nie", expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])

        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum("nij,nje->nie", expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])


    @plum.dispatch
    def construct_EPh(self, walkers: EPhWalkers, hamiltonian: BondSSHModel) -> numpy.ndarray:
       
        offdiags = numpy.zeros((walkers.nwalkers, self.nsites, self.nsites), dtype=numpy.complex128)
        for w,disp in enumerate(walkers.phonon_disp): 
            offdiags[w,:,:] = numpy.diag(disp[:-1], -1)
            offdiags[w,:,:] += offdiags[w,:,:].T
        
        if hamiltonian.pbc:
            offdiags[:,-1,0] = offdiags[:,0,-1] = walkers.phonon_disp[:,-1]

        EPh = hamiltonian.g_tensor * offdiags 
        return EPh

    @plum.dispatch
    def construct_EPh(self, walkers: EPhWalkers, hamiltonian: AcousticSSHModel) -> numpy.ndarray:
        
        displacement = numpy.einsum('ij,nj->ni', hamiltonian.X_connectivity, walkers.phonon_disp)
        offdiags = numpy.zeros((walkers.nwalkers, self.nsites, self.nsites), dtype=numpy.complex128)
        for w,disp in enumerate(displacement):
            offdiags[w,:,:] = numpy.diag(disp[:-1], -1)
            offdiags[w,:,:] += offdiags[w,:,:].T

        if hamiltonian.pbc:
            offdiags[:,-1,0] = offdiags[:,0,-1] = displacement[:,-1]

        # Shapes:   (nsites, nsites) * (nwalkers, nsites, nsites) -> (nwalkers, nsites, nsites)
        EPh = hamiltonian.g_tensor * offdiags
        return EPh


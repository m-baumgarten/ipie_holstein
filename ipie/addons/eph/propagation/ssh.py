import numpy 
import time
import scipy.linalg

from ipie.addons.eph.hamiltonians.ssh import SSHModel
from ipie.addons.eph.propagation.holstein import (
        HolsteinPropagatorFree,
        construct_one_body_propagator,
)
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import synchronize, cast_to_device
from ipie.propagation.continuous_base import PropagatorTimer

# TODO change propaagte_walkers such that propagate_phonons and electrons
# takes walkers, trial, hamiltonian to just use HolsteinPropagatorFree / other
# Base class 

class BondSSHPropagatorFree(HolsteinPropagatorFree):
    """"""
    def __init__(self, time_step: float, verbose: bool = False):
        super().__init__(time_step)

    def build(self, hamiltonian: SSHModel, trial=None, walkers=None, mpi_handler=None):   
        self.expH1 = construct_one_body_propagator(hamiltonian, self.dt)
        self.const = hamiltonian.g * numpy.sqrt(2. * hamiltonian.m * hamiltonian.w0) * self.dt
        self.w0 = hamiltonian.w0
        self.m = hamiltonian.m
        self.scale = numpy.sqrt(self.dt_ph / self.m)
        self.nsites = hamiltonian.nsites

    def propagate_phonons(self, walkers):
        start_time = time.time()

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot)

        N = numpy.random.normal(loc=0.0, scale=self.scale, 
                             size=(walkers.nwalkers, self.nsites))        
        walkers.x = walkers.x + N 

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot)
            
        walkers.weight *= numpy.exp(self.dt_ph * self.nsites * self.w0 / 2) #doesnt matter for estimators

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_electron(self, walkers, hamiltonian, trial):
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers) 
        synchronize()
        self.timer.tgf += time.time() - start_time

        Eph = numpy.einsum('ij,ni->nij', hamiltonian.hop[0], walkers.x)
        expEph = scipy.linalg.expm(self.const * Eph)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum('nij,nje->nie', expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        
        if walkers.ndown > 0:
            Eph = numpy.einsum('ij,ni->nij', hamiltonian.hop[1], walkers.x)
            expEph = scipy.linalg.expm(self.const * Eph)

            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum('nij,nje->nie', expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=None):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Walkers
        # 2.a DMC for phonon degrees of freedom
        self.propagate_phonons(walkers)

        # 2.b One-body propagation for electrons
        self.propagate_electron(walkers, hamiltonian, trial)

        # 2.c DMC for phonon degrees of freedom
        self.propagate_phonons(walkers)

        # Update weights (and later do phaseless for multi-electron)
        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp, ovlp_new)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    
    def update_weight(self, walkers, ovlp, ovlp_new):
        walkers.weight *= ovlp_new / ovlp


class BondSSHPropagator(BondSSHPropagatorFree):
    """"""
    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def propagate_phonons(self, walkers, hamiltonian, trial):
        """Propagates phonons via Diffusion MC."""
        start_time = time.time()
        
        # No ZPE in pot -> cancels with ZPE of etrial, wouldn't affect estimators anyways
        ph_ovlp_old = trial.calc_phonon_overlap(walkers)
        
        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian_importance(walkers) / hamiltonian.m
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot / 2)

        N = numpy.random.normal(loc=0.0, scale=self.scale, size=(walkers.nwalkers, self.nsites))    
        drift = trial.calc_phonon_gradient(walkers)
        walkers.x = walkers.x + N + self.dt_ph * drift / hamiltonian.m

        ph_ovlp_new = trial.calc_phonon_overlap(walkers)        

        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian_importance(walkers) / hamiltonian.m
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot / 2)

        walkers.weight *= ph_ovlp_old / ph_ovlp_new
        walkers.weight *= numpy.exp(self.dt_ph * trial.energy)

        synchronize()
        self.timer.tgemm += time.time() - start_time
        

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=None):
        """"""
        synchronize()
        start_time = time.time()
        
        ovlp = trial.calc_overlap(walkers).copy()
        
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Walkers
        # 2.a DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)
        
        # 2.b One-body propagation for electrons
        self.propagate_electron(walkers, hamiltonian, trial)
        
        # 2.c DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        start_time = time.time()
        
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        
        self.update_weight(walkers, ovlp, ovlp_new)
        
        synchronize()
        self.timer.tupdate += time.time() - start_time

class OpticalSSHPropagatorFree:
    def __init__(self):
        ...

class OpticalSSHPropagator:
    def __init__(self):
        ...



from pyqumc.qmc.afqmc_batch import AFQMCBatch
import numpy
from pyqumc.qmc.options import QMCOpts
from pyqumc.propagation.utils import get_propagator_driver
from pyqumc.hamiltonians.utils import get_hamiltonian
from pyqumc.trial_wavefunction.utils import get_trial_wavefunction
from pyqumc.estimators.local_energy import local_energy
from pyqumc.systems.generic import Generic
from pyqumc.utils.mpi import get_shared_comm
from pyqumc.utils.io import  get_input_value
from pyqumc.walkers.single_det_batch import SingleDetWalkerBatch
from pyqumc.walkers.single_det import SingleDetWalker

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import cProfile

nelec = (5,5)
nwalkers = 10
nsteps = 10

options = {
    "system": {
        "nup": nelec[0],
        "ndown": nelec[1],
    },
    "hamiltonian": {
        "name": "Generic",
        "integrals": "afqmc.h5"
    },
    "qmc": {
        "dt": 0.01,
        "nsteps": nsteps,
        "nwalkers": nwalkers,
        "blocks": 1,
        "batched": True
    },
    "trial": {
        "filename": "afqmc.h5"
    },
    "estimators": {}
}

numpy.random.seed(7)
sys = Generic(nelec=nelec)
comm = MPI.COMM_WORLD
verbose = True
shared_comm = get_shared_comm(comm, verbose=verbose)

qmc_opts = get_input_value(options, 'qmc',
                           default={},
                           verbose=verbose)
ham_opts = get_input_value(options, 'hamiltonian',
                           default={},
                           verbose=verbose)
twf_opts = get_input_value(options, 'trial',
                           default={},
                           verbose=verbose)
prop_opts = get_input_value(options, 'propoagator',
                           default={},
                           verbose=verbose)
qmc = QMCOpts(qmc_opts, sys,verbose=True)
ham = get_hamiltonian (sys, ham_opts, verbose = True, comm=shared_comm)

trial = ( get_trial_wavefunction(sys, ham, options=twf_opts,
                       comm=comm,
                       scomm=shared_comm,
                       verbose=verbose) )
trial.psi = trial.psi[0] # Super hacky thing to do; this needs to be fixed...
trial.calculate_energy(sys, ham) # this is to get the energy shift

prop = get_propagator_driver(sys, ham, trial, qmc, options=prop_opts,verbose=verbose)

walker_batch = SingleDetWalkerBatch(sys, ham, trial, nwalkers)
for i in range (nsteps):
    prop.propagate_walker_batch(walker_batch, sys, ham, trial, trial.energy)
    walker_batch.reortho()

options = {
    "system": {
        "nup": nelec[0],
        "ndown": nelec[1],
    },
    "hamiltonian": {
        "name": "Generic",
        "integrals": "afqmc.h5"
    },
    "qmc": {
        "dt": 0.01,
        "nsteps": nsteps,
        "nwalkers": nwalkers,
        "blocks": 1,
        "batched": False
    },
    "trial": {
        "filename": "afqmc.h5"
    },
    "estimators": {}
}

numpy.random.seed(7)
sys = Generic(nelec=nelec)
comm = MPI.COMM_WORLD
verbose = True
shared_comm = get_shared_comm(comm, verbose=verbose)

qmc_opts = get_input_value(options, 'qmc',
                           default={},
                           verbose=verbose)
ham_opts = get_input_value(options, 'hamiltonian',
                           default={},
                           verbose=verbose)
twf_opts = get_input_value(options, 'trial',
                           default={},
                           verbose=verbose)
prop_opts = get_input_value(options, 'propoagator',
                           default={},
                           verbose=verbose)
qmc = QMCOpts(qmc_opts, sys,verbose=True)
prop = get_propagator_driver(sys, ham, trial, qmc, options=prop_opts,verbose=verbose)
walkers = [SingleDetWalker(sys, ham, trial) for iw in range(nwalkers)]
for i in range (nsteps):
    for walker in walkers:
        prop.propagate_walker(walker, sys, ham, trial, trial.energy)
        detR = walker.reortho(trial) # reorthogonalizing to stablize

for iw in range(nwalkers):
    assert numpy.allclose(walker_batch.phi[iw], walkers[iw].phi)



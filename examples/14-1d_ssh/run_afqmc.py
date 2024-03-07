import numpy as np
np.random.seed(125)
from mpi4py import MPI

from ipie.qmc.afqmc import AFQMC
from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.ssh import SSHModel
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.coherent_state_variational_ssh import variational_trial
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.estimators.energy import EnergyEstimator
#from ipie.addons.eph.estimators.overlap import OverlapEstimator

#System Parameters
nup = 2
ndown = 2
nelec = (nup, ndown)

#Hamiltonian Parameters
g = 2.
v = 0.5
w = 2.
w0 = 4.
ncell = 3
nsites = ncell*2
pbc = True

#Walker Parameters & Setup
comm = MPI.COMM_WORLD
nwalkers = 1000 // comm.size

#Setup initial guess for variational optimization
initial_electron = np.random.random((nsites, nup + ndown))
initial_phonons = np.ones(nsites) * 0.1

#System and Hamiltonian setup
system = Generic(nelec)
ham = SSHModel(g=g, v=v, w=w, w0=w0, ncell=ncell, pbc=pbc)
ham.build()

#Variational procedure
etrial, beta_shift, el_trial = variational_trial(
        initial_phonons, initial_electron, ham, system
)
wavefunction = np.column_stack([beta_shift, el_trial])

#Setup trial
trial = CoherentStateTrial(
    wavefunction=wavefunction,
    hamiltonian=ham,
    num_elec=[nup, ndown],
    num_basis=nsites
)
trial.set_etrial(etrial)

#Setup walkers
walkers = EPhWalkers(
    initial_walker=wavefunction,
    nup=nup,
    ndown=ndown,
    nbasis=nsites,
    nwalkers=nwalkers
)
walkers.build(trial)

num_steps_per_block = 10
num_blocks = 10000
add_est = {
    "energy": EnergyEstimator(
        system=system, ham=ham, trial=trial
    ),
#    "overlap": OverlapEstimator(
#        system=system, ham=ham, trial=trial
#    )
}

seed = 125

# Note nwalkers specifies the number of walkers on each CPU
ephqmc = AFQMC.build(
        num_elec=nelec, 
        hamiltonian=ham, 
        trial_wavefunction=trial, 
        walkers=walkers, 
        num_walkers=nwalkers, 
        seed=seed, 
        num_steps_per_block=num_steps_per_block, 
        num_blocks=num_blocks,
)
ephqmc.run(additional_estimators=add_est, verbose=False)


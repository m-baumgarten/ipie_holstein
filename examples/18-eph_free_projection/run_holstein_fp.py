import numpy as np
from ipie.config import MPI
from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.free_projection.walkers.eph_walkers import EPhWalkersFP
from ipie.addons.eph.estimators.energy import EnergyEstimator
from ipie.addons.free_projection.propagation.eph_propagator_fp import EPhPropagatorFP, EPhPropagatorFPImportance
comm = MPI.COMM_WORLD

from ipie.addons.free_projection.qmc.options import QMCParamsFP
from ipie.addons.free_projection.qmc.fp_afqmc_eph import FPAFQMC


# System Parameters
nup = 1
ndown = 0
nelec = (nup, ndown)

# Hamiltonian Parameters
g = 3.0 #1.0
t = 1.0
w0 = 3.0
nsites = 5
pbc = True
k = 0
nk = nsites//2 + 1
K = np.linspace(0, np.pi, nk, endpoint=True)[k]
importance = True

# Walker Parameters & Setup
comm = MPI.COMM_WORLD
nwalkers = 200
num_iterations_fp = 150
num_blocks = 200
num_steps_per_block = 10

# System and Hamiltonian setup
system = Generic(nelec)
ham = HolsteinModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
ham.build()

beta_shift = np.load('trial_ph_{:02d}.npy'.format(k))
el_trial = np.load('trial_el_{:02d}.npy'.format(k))
wavefunction = np.column_stack([beta_shift, el_trial])

# Setup trial
trial = ToyozawaTrial(
    wavefunction=wavefunction, w0=ham.w0, num_elec=[nup, ndown], num_basis=nsites, K=K
)
#trial = CoherentStateTrial(
#    wavefunction=wavefunction, w0=ham.w0, num_elec=[nup, ndown], num_basis=nsites
#)
trial.set_etrial(ham)

# Setup walkers
walkers = EPhWalkersFP(
    initial_walker=wavefunction, nup=nup, ndown=ndown, nbasis=nsites, nwalkers=nwalkers
)
walkers.build(trial)

params = QMCParamsFP(
    num_walkers=nwalkers,
    total_num_walkers=nwalkers * comm.size,
    num_blocks=num_blocks,
    num_steps_per_block=num_steps_per_block,
    timestep=0.0025,
    num_stblz=5,
    rng_seed=125,
    pop_control_freq=5,
    num_iterations_fp=num_iterations_fp,
)

# Setup propagator
if importance:
    propagator = EPhPropagatorFPImportance(timestep=params.timestep, verbose=False, exp_nmax=10, ene_0=-2.469)
else:
    propagator = EPhPropagatorFP(timestep=params.timestep, verbose=False, exp_nmax=10, ene_0=-2.469)
propagator.build(ham, trial, walkers)

fpafqmc = FPAFQMC(
    system,
    ham,
    trial,
    walkers,
    propagator,
    params,
    verbose=False,
)
fpafqmc.run(importance_sampling=importance)

# analysis
if comm.rank == 0:
    from ipie.addons.free_projection.analysis.extraction import extract_observable
    from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios

    data = np.zeros((fpafqmc.params.num_blocks, 3), dtype=np.complex128)
    for i in range(fpafqmc.params.num_blocks):
        data[i, 0] = (i+1) * fpafqmc.params.num_steps_per_block * fpafqmc.params.timestep
        print(
            f"\nEnergy statistics at time {(i+1) * fpafqmc.params.num_steps_per_block * fpafqmc.params.timestep}:"
        )
        qmc_data = extract_observable(fpafqmc.estimators[i].filename, "energy")
        energy_mean, energy_err = jackknife_ratios(qmc_data["ENumer"], qmc_data["EDenom"])
        data[i, 1], data[i, 2] = energy_mean, energy_err
        print(f"Energy: {energy_mean:.8e} +/- {energy_err:.8e}")
    np.save('fp_data_3000walkers.npy', data)

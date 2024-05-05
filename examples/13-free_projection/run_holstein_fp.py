import numpy as np
from ipie.config import MPI
from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.estimators.energy import EnergyEstimator
from ipie.addons.eph.propagation.holstein import FreePropagationHolstein
comm = MPI.COMM_WORLD

from ipie.addons.free_projection.qmc.options import QMCParamsFP
from ipie.addons.free_projection.qmc.fp_afqmc import FPAFQMC


# System Parameters
nup = 1
ndown = 0
nelec = (nup, ndown)

# Hamiltonian Parameters
g = 1.0
t = 1.0
w0 = 1.0
nsites = 32
pbc = True
#K = 2 * k * np.pi / nsites
k = 0
nk = nsites//2 + 1 #sum(divmod(nsites+1, 2))
K = np.linspace(0, np.pi, nk, endpoint=True)[k]

# Walker Parameters & Setup
comm = MPI.COMM_WORLD
nwalkers = 10
num_iterations_fp = 100
num_blocks = 15
num_steps_per_block = 40

# System and Hamiltonian setup
system = Generic(nelec)
ham = HolsteinModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
ham.build()

beta_shift = np.load('../14-1d_holstein/bonca_compare/trials_32/trial_ph_{:02d}.npy'.format(k))
el_trial = np.load('../14-1d_holstein/bonca_compare/trials_32/trial_el_{:02d}.npy'.format(k))
wavefunction = np.column_stack([beta_shift, el_trial])

# Setup trial
trial = ToyozawaTrial(
    wavefunction=wavefunction, w0=ham.w0, num_elec=[nup, ndown], num_basis=nsites, K=K
)
trial.set_etrial(ham)

# Setup walkers
walkers = EPhWalkers(
    initial_walker=wavefunction, nup=nup, ndown=ndown, nbasis=nsites, nwalkers=nwalkers
)
walkers.build(trial)

params = QMCParamsFP(
    num_walkers=nwalkers,
    total_num_walkers=nwalkers * comm.size,
    num_blocks=num_blocks,
    num_steps_per_block=num_steps_per_block,
    timestep=0.005,
    num_stblz=5,
    rng_seed=125,
    pop_control_freq=-1,
    num_iterations_fp=num_iterations_fp,
)
propagator = FreePropagationHolstein(timestep=params.timestep, verbose=False, exp_nmax=10, ene_0=0.)
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
fpafqmc.run()

# analysis
if comm.rank == 0:
    from ipie.addons.free_projection.analysis.extraction import extract_observable
    from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios

    for i in range(fpafqmc.params.num_blocks):
        print(
            f"\nEnergy statistics at time {(i+1) * fpafqmc.params.num_steps_per_block * fpafqmc.params.timestep}:"
        )
        qmc_data = extract_observable(fpafqmc.estimators[i].filename, "energy")
        energy_mean, energy_err = jackknife_ratios(qmc_data["ENumer"], qmc_data["EDenom"])
        print(f"Energy: {energy_mean:.8e} +/- {energy_err:.8e}")

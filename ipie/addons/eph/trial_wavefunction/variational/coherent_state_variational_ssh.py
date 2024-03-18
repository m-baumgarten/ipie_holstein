import numpy as np
from scipy.optimize import basinhopping

from ipie.systems.generic import Generic
from ipie.addons.eph.hamiltonians.ssh import BondSSHModel, AcousticSSHModel
from ipie.addons.eph.trial_wavefunction.variational.estimators import gab

import jax
from jax.config import config
import jax.numpy as npj
config.update("jax_enable_x64", True)

# def local_energy(
#        X: np.ndarray,
#        G: np.ndarray,
#        m: float,
#        w: float,
#        g: float,
#        nsites: int,
#        Lap: np.ndarray,
#        T: np.ndarray,
#        nup: int,
#        ndown: int
#    ) -> np.ndarray:


def local_energy_acoustic(X: np.ndarray, G: np.ndarray, hamiltonian, system) -> float:
    g = hamiltonian.g
    w = hamiltonian.w0
    m = hamiltonian.m
    nsites = hamiltonian.nsites

    kinetic = np.sum(hamiltonian.T[0] * G[0] + hamiltonian.T[1] * G[1])
    
    # Make displacement matrix
    displ = hamiltonian.X_connectivity.dot(X)
    displ_mat = npj.diag(displ[:-1], 1) #+ npj.diag(displ[:-1], -1)
    displ_mat += displ_mat.T
    if hamiltonian.pbc:
        displ_mat = displ_mat.at[0,-1].set(displ[-1]) 
        displ_mat = displ_mat.at[-1,0].set(displ[-1])
    
    # Merge Hilbert spaces
    tmp0 = hamiltonian.g_tensor * G[0] * displ_mat
    if system.ndown > 0:
        tmp0 += hamiltonian.g_tensor * G[1] * displ_mat
    el_ph_contrib = 2 * jax.numpy.sum(tmp0)

    phonon_contrib = w * jax.numpy.sum(X * X)
   
#    jax.debug.print('kinetic:   {x}', x=kinetic)
#    jax.debug.print('el_ph_:    {x}', x=el_ph_contrib)
#    jax.debug.print('phonon:    {x}', x=phonon_contrib)

    local_energy = kinetic + el_ph_contrib + phonon_contrib
#    jax.debug.print('total_e:   {x}', x=local_energy)
    return local_energy

def local_energy_bond(X: np.ndarray, G: np.ndarray, hamiltonian, system) -> float:
    g = hamiltonian.g
    w = hamiltonian.w0
    m = hamiltonian.m
    nsites = hamiltonian.nsites

    kinetic = np.sum(hamiltonian.T[0] * G[0] + hamiltonian.T[1] * G[1])

    offdiags = X[:-1]
    displ = npj.diag(offdiags, -1)
    displ += displ.T
    # This does not matter bc it will be multiplied by diagonal g_tensor eventually.
    # Only included for readibility.
    if hamiltonian.pbc:
        displ = displ.at[-1,0].set(X[-1])
        displ = displ.at[0,-1].set(X[-1])

    tmp0 = hamiltonian.g_tensor * G[0] * displ
    if system.ndown > 0:
        tmp0 += hamiltonian.g_tensor * G[1] * displ 
    el_ph_contrib = 2 * jax.numpy.sum(tmp0)

    phonon_contrib = w * jax.numpy.sum(X * X)
    
    local_energy = kinetic + el_ph_contrib + phonon_contrib
    return local_energy

def objective_function(x: np.ndarray, hamiltonian, system):
    nbasis = hamiltonian.nsites
    nup = system.nup
    ndown = system.ndown

    shift = x[0:nbasis].copy()
    shift = shift.astype(np.float64)

    c0a = x[nbasis : (nup + 1) * nbasis].copy()
    c0a = jax.numpy.reshape(c0a, (nbasis, nup))
    c0a = c0a.astype(np.float64)
    Ga = gab(c0a, c0a)

    if ndown > 0:
        c0b = x[(nup + 1) * nbasis :].copy()
        c0b = jax.numpy.reshape(c0b, (nbasis, ndown))
        c0b = c0b.astype(np.float64)
        Gb = gab(c0b, c0b)
    else:
        Gb = jax.numpy.zeros_like(Ga, dtype=np.float64)

    G = [Ga, Gb]

    if type(hamiltonian) is BondSSHModel:
        local_energy = local_energy_bond
    elif type(hamiltonian) is AcousticSSHModel:
        local_energy = local_energy_acoustic

    etot = local_energy(shift, G, hamiltonian, system)
    return etot.real


def gradient(x: np.ndarray, hamiltonian, system):
    grad = np.array(jax.grad(objective_function)(x, hamiltonian, system), dtype=np.float64)
    return grad


def func(x: np.ndarray, hamiltonian, system):
    f = objective_function(x, hamiltonian, system)
    df = gradient(x, hamiltonian, system)
    return f, df


def print_fun(x: np.ndarray, f: float, accepted: bool):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))


def variational_trial(init_phonons: np.ndarray, init_electron: np.ndarray, hamiltonian, system):

    init_phonons = init_phonons.astype(np.float64)
    init_electron = init_electron.astype(np.float64).ravel()

    hamiltonian.X_connectivity = npj.array(hamiltonian.X_connectivity.astype(np.float64))
    x = np.hstack([init_phonons, init_electron])

    maxiter = 500
    minimizer_kwargs = {
        "jac": True,
        #        "args": (hamiltonian.nsites, hamiltonian.T, hamiltonian.g, hamiltonian.m, hamiltonian.w0, system.nup, system.ndown),
        "args": (hamiltonian, system),
        "options": {
            "gtol": 1e-10,
            "eps": 1e-10,
            "maxiter": maxiter,
            "disp": False,
        },
    }

    res = basinhopping(
        func,
        x,
        minimizer_kwargs=minimizer_kwargs,
        callback=print_fun,
        niter=maxiter,
        niter_success=3,
    )

    etrial = res.fun

    beta_shift = res.x[: hamiltonian.nsites]
    if system.ndown > 0:
        psia = res.x[hamiltonian.nsites : hamiltonian.nsites * (system.nup + 1)]
        psia = psia.reshape((hamiltonian.nsites, system.nup))
        psib = res.x[hamiltonian.nsites * (system.nup + 1) :]
        psib = psib.reshape((hamiltonian.nsites, system.ndown))
        psi = np.column_stack([psia, psib])
    else:
        psia = res.x[hamiltonian.nsites :].reshape((hamiltonian.nsites, system.nup))
        psi = psia

    return etrial, beta_shift, psi

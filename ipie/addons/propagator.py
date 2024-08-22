from ipie.addons.eph.propagation.eph_propagator import EPhPropagator, EPhPropagatorFree
from ipie.addons.eph.hamiltonians.eph_generic import GenericEPhModel
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.hamiltonians.ssh import BondSSHModel
from ipie.addons.eph.hamiltonians.ssh import OpticalSSHModel
from ipie.addons.eph.propagation.cs_propagator import CoherentStatePropagator

PropagatorAddons = {GenericEPhModel: EPhPropagator, HolsteinModel: CoherentStatePropagator, BondSSHModel: EPhPropagator, OpticalSSHModel: EPhPropagator}

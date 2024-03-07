from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.addons.eph.propagation.holstein import HolsteinPropagatorImportance
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.propagation.ssh import BondSSHPropagator
from ipie.addons.eph.hamiltonians.ssh import SSHModel

Propagator = {
    GenericRealChol: PhaselessGeneric, 
    GenericComplexChol: PhaselessGeneric, 
    HolsteinModel: HolsteinPropagatorImportance,
    SSHModel: BondSSHPropagator,
}

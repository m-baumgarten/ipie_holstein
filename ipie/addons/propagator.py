from ipie.addons.eph.propagation.holstein import HolsteinPropagator
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.propagation.ssh import SSHPropagator
from ipie.addons.eph.hamiltonians.ssh import BondSSHModel
from ipie.addons.eph.hamiltonians.ssh import AcousticSSHModel

PropagatorAddons = {HolsteinModel: HolsteinPropagator, BondSSHModel: SSHPropagator, AcousticSSHModel: SSHPropagator}

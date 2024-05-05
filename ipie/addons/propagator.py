from ipie.addons.eph.propagation.holstein import HolsteinPropagator, FreePropagationHolstein
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.propagation.ssh import SSHPropagator
from ipie.addons.eph.hamiltonians.ssh import BondSSHModel
from ipie.addons.eph.hamiltonians.ssh import AcousticSSHModel

PropagatorAddons = {HolsteinModel: FreePropagationHolstein, BondSSHModel: SSHPropagator, AcousticSSHModel: SSHPropagator}

import numpy
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp


class SSHModel(GenericBase):
    """Class for SSH model carrying elph tensor and system parameters
    """

    def __init__(self, g: float, v: float, w: float, 
                 w0: float, ncell: int, pbc: bool):
        """Each site corresponds to a unit cell of 2 atoms, 
        one on each sublattice A and B"""
        self.g = g
        self.v = v
        self.w = w
        self.w0 = w0
        self.m = 1/self.w0
        self.nsites = 2*ncell 
        self.pbc = pbc
        self.T = None
        self.const = -self.g * numpy.sqrt(2. * self.m * self.w0)

    def build(self):
        offdiags = numpy.ones(self.nsites-1)
        offdiags[::2] *= self.v
        offdiags[1::2] *= self.w
    

        self.T = numpy.diag(offdiags, 1)
        self.T += numpy.diag(offdiags, -1)
    
        if self.pbc:
            self.T[0,-1] = self.T[-1,0] = self.w

        self.T *= -1.

        self.T = [self.T.copy(), self.T.copy()]

        self.hop = numpy.diag(numpy.ones(self.nsites-1), 1)
        self.hop += numpy.diag(numpy.ones(self.nsites-1), -1)
        self.hop = [self.hop.copy(), self.hop.copy()]


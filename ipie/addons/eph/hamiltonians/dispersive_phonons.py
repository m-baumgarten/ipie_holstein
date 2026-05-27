# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy
from typing import Sequence
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.utils.backend import arraylib as xp

class DispersivePhononModel(HolsteinModel):
    """Class for Holstein Model + dispersive optical phonons, as seen in 
    https://www.osti.gov/servlets/purl/1856155"""
    def __init__(self, g: float, t: float, tph: float, w0: float, nsites: int, pbc: bool):
        super().__init__(g, t, w0, nsites, pbc)
        self.tph = tph
        self.w0 = w0
        self.ph_tensor = self.build_ph_disp()

    def build_ph_disp(self):
        return - super().build_T_1D(self.nsites[0]) * self.tph / self.t



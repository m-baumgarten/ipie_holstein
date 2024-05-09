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

from typing import Union
from abc import ABCMeta, abstractmethod
import numpy as np

class GenericEPhModel(metaclass=ABCMeta):
    def __init__(self, nsites: Union[np.ndarray, int], pbc: bool):
        
        if isinstance(nsites, np.ndarray):
            assert len(nsites.shape) == 1
            assert len(nsites) < 3
            self.dim = len(nsites)
        else:
            self.dim = 1
            nsites = np.array([nsites])
        self.nsites = nsites
        self.N = np.prod(nsites)
        self.pbc = pbc

    @abstractmethod
    def build(self): ...

    @abstractmethod
    def build_T(self): ...

    @abstractmethod
    def build_g(self): ...

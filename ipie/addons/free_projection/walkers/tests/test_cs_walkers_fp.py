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
import pytest

from ipie.addons.free_projection.walkers.cs_walkers import EPhCSWalkersFP
from ipie.walkers.pop_controller import get_buffer, set_buffer


class _FakeMPIHandler:
    pass


@pytest.mark.unit
def test_eph_cs_walkers_fp_stores_mpi_handler_and_buffers_weight_log():
    nsites = 3
    nwalkers = 2
    initial_walker = numpy.column_stack(
        [
            numpy.array([0.1 + 0.2j, -0.3 + 0.1j, 0.4 - 0.2j]),
            numpy.array([1.0, 0.5j, -0.25]),
            numpy.array([0.25j, -0.75, 0.5 + 0.2j]),
        ]
    )
    mpi_handler = _FakeMPIHandler()

    walkers = EPhCSWalkersFP(
        initial_walker,
        nup=1,
        ndown=1,
        nbasis=nsites,
        nwalkers=nwalkers,
        mpi_handler=mpi_handler,
    )

    assert walkers.mpi_handler is mpi_handler
    assert "weight_log" in walkers.buff_names

    walkers.weight[:] = numpy.array([1.5 + 0.0j, 0.2 + 0.0j])
    walkers.phase[:] = numpy.array([0.5 + 0.5j, -0.25 + 0.75j])
    walkers.weight_log[:] = numpy.array([0.25 + 0.1j, -2.0 + 0.4j])
    walkers.phia[:] = numpy.array(
        [
            [[1.0 + 0.2j], [0.3 - 0.1j], [-0.4 + 0.5j]],
            [[-0.2 + 0.7j], [0.6 + 0.1j], [0.8 - 0.3j]],
        ]
    )
    walkers.phib[:] = numpy.array(
        [
            [[-0.5 + 0.1j], [0.2 + 0.3j], [0.7 - 0.6j]],
            [[0.1 - 0.4j], [-0.8 + 0.2j], [0.3 + 0.9j]],
        ]
    )
    walkers.coherent_state_shift[:] = numpy.array(
        [
            [0.1 + 0.2j, -0.3 + 0.4j, 0.5 - 0.1j],
            [-0.6 + 0.3j, 0.2 - 0.5j, -0.7 + 0.8j],
        ]
    )

    source_weight = walkers.weight[0].copy()
    source_phase = walkers.phase[0].copy()
    source_weight_log = walkers.weight_log[0].copy()
    source_phia = walkers.phia[0].copy()
    source_phib = walkers.phib[0].copy()
    source_shift = walkers.coherent_state_shift[0].copy()
    buff = get_buffer(walkers, 0)
    set_buffer(walkers, 1, buff)

    assert numpy.allclose(walkers.weight[1], source_weight)
    assert numpy.allclose(walkers.phase[1], source_phase)
    assert numpy.allclose(walkers.weight_log[1], source_weight_log)
    assert numpy.allclose(walkers.phia[1], source_phia)
    assert numpy.allclose(walkers.phib[1], source_phib)
    assert numpy.allclose(walkers.coherent_state_shift[1], source_shift)

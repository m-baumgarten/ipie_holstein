# Ab-Initio EPh dD2 Fixture Workflow

This directory is a small smoke-test style example for the ab-initio
electron-phonon dD2 stack:

- `AbInitioEPhHamiltonian`
- `AbInitioDD2Trial`
- `AbInitioEPhWalkers`
- `AbInitioEPhPropagator`
- `local_energy_abinitio`

The workflow reads:

- `svd_kq.h5`
- `electron_3.npy`
- `shift_3.npy`

The HDF5 file stores:

- `bands` as `eps_kj` with shape `(nk, nband)`
- `phfreq` as `omega_qnu` with shape `(nq, nmode)`
- `Uk_{real,imag}` with shape `(n_svd, i, j, k)`
- `Vq_{real,imag}` with shape `(nmode, n_svd, i, j, q)`
- `longrange_{real,imag}` with shape `(nq, nmode)`
- `Uk_bloch_{real,imag}` with shape `(nk, norb, nband)`

For now the example explicitly reconstructs
`g_qnu_kmn` with shape `(nq, nmode, nk, nband, nband)`.
The short-range part is contracted as

```python
g_orb[q, nu, k, i, j] = sum_s Vq[nu, s, i, j, q] * Uk[s, i, j, k]
```

then the long-range contribution is added to the diagonal and each matrix is
rotated to the band basis with `Uk_bloch[k+q].conj().T @ g_orb @ Uk_bloch[k]`.
Eventually the Hamiltonian and estimator should consume this low-rank
structure directly; this dense reconstruction is only the clean workflow test.
The resulting `g_qnu_kmn` is passed to the Hamiltonian exactly as reconstructed;
the ab-initio code does not apply an additional `1/sqrt(N)` normalization.

`electron_3.npy` is used as `psi_kj`; `shift_3.npy` is used as `beta_qnu`.

Run the example as a pytest smoke test:

```bash
conda run -n ipie_dev python -m pytest test_abinitio_workflow.py
```

or as a short script:

```bash
conda run -n ipie_dev python test_abinitio_workflow.py
```

Run the same fixture through ipie's AFQMC driver:

```bash
conda run -n ipie_dev python run_afqmc.py --blocks 20 --steps-per-block 10 --nwalkers 200
```

`run_afqmc.py` builds `AbInitioEPhHamiltonian` and `AbInitioDD2Trial`, then lets
`AFQMC.build(...)` select `AbInitioEPhPropagator` from ipie's propagator
registry.  The default outputs are the standard ipie estimator file
`estimates.0.h5`, the ASCII energy table `energy.dat`, and a line-flushed
stdout log `afqmc.out`.  The log receives the same block table printed to the
terminal, so it can be monitored while the run is still active:

```bash
tail -f afqmc.out
```

The HDF5 file stores the usual metadata plus a `block_size_1` estimator group.
The first columns are walker properties, followed by the energy estimator
columns:

```python
import h5py
import numpy as np

with h5py.File("estimates.0.h5", "r") as handle:
    group = handle["block_size_1"]
    walker_names = [
        name.decode() if isinstance(name, bytes) else name
        for name in group["walker_prop_header"][()]
    ]
    energy_names = group["names"]["energy"][()].decode().split()
    columns = walker_names + energy_names
    data = group["data"]["000000000"][: group["max_block"]["0"][()] + 1]

time_step = 1.0e-7
steps_per_block = 10
imag_time = np.arange(data.shape[0]) * steps_per_block * time_step
energy = data[:, columns.index("ETotal")].real
```

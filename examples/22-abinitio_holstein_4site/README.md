# Four-Site Holstein Through Ab-Initio EPh

This example uses the ab-initio EPh Hamiltonian, dD2 trial, walkers,
propagator, and estimator for a simple four-site Holstein model.

The generated tensors are:

- `eps_kj[k, 0] = -2 t cos(k)` with `k = 2 pi n / N`
- `omega_qnu[q, 0] = w0`
- `g_qnu_kmn[q, 0, k, 0, 0] = g / sqrt(N)`

The last point is the main normalization check: the ab-initio Hamiltonian uses
the supplied `g_qnu_kmn` directly, so the Holstein Fourier factor is included in
the fixture.

Run a smoke test:

```bash
conda run -n ipie_dev python -m pytest test_holstein_abinitio_workflow.py
```

Run AFQMC:

```bash
conda run -n ipie_dev python run_afqmc.py --blocks 20 --steps-per-block 10 --nwalkers 200 --timestep 0.005
```

The default outputs are `estimates.0.h5`, `energy.dat`, and the line-flushed
stdout log `afqmc.out`.

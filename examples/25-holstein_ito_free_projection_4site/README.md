# Four-Site Holstein Euler-Ito Free Projection

This example exercises the no-importance free-projection Euler-Ito coherent-state
propagator for a four-site Holstein model.

Run a small smoke calculation from this directory with:

```bash
python run_afqmc.py
```

The script builds:

- `ToyozawaTrialUnnormalizedCoherentState`
- `EPhCSWalkersFP`
- `EulerItoPropagatorFP`
- `FPAFQMC(...).run(importance_sampling=False)`

The estimator files are written as `estimate.h5.0`, `estimate.h5.1`, ... because
the free-projection driver writes one estimator file per imaginary-time block.

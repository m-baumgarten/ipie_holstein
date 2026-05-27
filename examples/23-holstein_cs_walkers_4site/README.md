# Four-Site Holstein With Coherent-State Walkers

This example runs the real-space Holstein model through the updated
coherent-state walker implementation.

The default run is:

- 4 sites, periodic boundary conditions
- one electron, `K = 0` Toyozawa coherent-state trial
- `t = 1`, `g = 1`, `w0 = 1`
- 200 total walkers
- timestep `0.005`
- 20 blocks with 1 step per block
- population control every step

This coherent-state walker path is still a diagnostic target. The unshifted
coherent-state proposal has a very broad phase distribution, so running many
propagation steps before branching can kill essentially all walker weight.
For debugging, keep `--steps-per-block 1 --pop-control-freq 1`.

Run the smoke test:

```bash
conda run -n ipie_dev python -m pytest test_holstein_cs_workflow.py
```

Run AFQMC:

```bash
conda run -n ipie_dev python run_afqmc.py
```

Optimize the localized Toyozawa seed in the site basis with JAX:

```bash
conda run -n ipie_dev python optimize_toyozawa_jax.py --output optimized_toyozawa_trial.npz
```

Use the optimized trial in AFQMC:

```bash
conda run -n ipie_dev python run_afqmc.py --trial-file optimized_toyozawa_trial.npz
```

For a one-shot diagnostic run, `run_afqmc.py` can also optimize before starting:

```bash
conda run -n ipie_dev python run_afqmc.py --optimize-trial --save-trial optimized_toyozawa_trial.npz
```

The default outputs are `estimates.0.h5`, `energy.dat`, and the line-flushed
stdout log `afqmc.out`.

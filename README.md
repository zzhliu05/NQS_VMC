# NQS VMC

The maintained torus JAX package is in
[`code/nqs_torus_laqx_nes`](code/nqs_torus_laqx_nes/README.md).
Its [input-parameter guide](code/nqs_torus_laqx_nes/INPUT_PARAMETERS.md)
lists every training CLI flag, defaults, constraints, performance switches,
and the separate magnetic-roton post-processing inputs.

From the repository root, with a Python environment containing JAX and the
project dependencies:

```bash
cd code
python -m nqs_torus_laqx_nes --help
python -m nqs_torus_laqx_nes.magnetoroton_cli --help
```

The pre-existing top-level scripts and transformer prototype are retained as
historical work. See the package's [code map](code/nqs_torus_laqx_nes/CODE_STRUCTURE.md)
and [local test status](code/nqs_torus_laqx_nes/TEST_STATUS.md) before
reproducing an experiment.

# Local test status (2026-10-22)

Environment: local Python with JAX 0.4.30, CPU.

- `fused4` configuration/CLI selection and numerical value against balanced contraction: passed.
- Contiguous fixed-width row reduction against scatter accumulation: passed.
- Full `unittest discover`: 115 tests, 4 failures and 8 errors. The package is **not** fully regression-clean in this environment.

The full-suite failures/errors group as follows:

| Area | Count | Observed issue |
| --- | ---: | --- |
| Continuous-coordinate tests | 2 failures | Nonfinite sampler log probability or `nan` normalization/energy |
| Tensor-bank tests | 2 errors | NumPy dtype promotion between float16 and bfloat16 |
| Packed-sector state handling | 6 errors | `NetworkConfig.num_sites` absent in bitstring-neighbor and older VMC workflow tests |
| Momentum-scattering sampler | 1 failure | Expected state change was not observed |
| Small-system energy regression | 1 failure | Energy differs from stored reference |

This status records observed local behavior; it does not establish whether
each failure reproduces on the main cluster or which earlier change caused it.
The source copy was SHA256-matched against the working package before this
documentation was added. No historical package or A100 snapshot was modified.

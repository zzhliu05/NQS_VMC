# Code structure

The package is intentionally flat so each module can be imported directly.
The primary training path is:

```text
__main__.py -> cli.py -> config.py -> workflow.py -> training.py
                                              |           |
                                              v           v
                                         physics.py   sampling.py
                                              |           |
                                              +----> optimization.py
                                                        |
                                                        v
                                                    network.py
```

| Module | Responsibility |
| --- | --- |
| `__main__.py`, `cli.py` | CLI entry point, argument parsing, and construction of typed run config |
| `config.py` | System, network, sampler, optimizer, training, and output validation |
| `workflow.py`, `training.py` | Sector orchestration, training loop, history, and checkpoints |
| `physics.py` | Torus Coulomb Hamiltonian, basis, and sector operations |
| `sampling.py` | Markov chains and sector proposals |
| `network.py` | MLP/NES/ViT/MPS wavefunctions; balanced and four-site fused MPS contraction |
| `optimization.py` | Local energy, streamed sparse rows, gradients, and optimizer updates; optional contiguous row reduction |
| `sparse_rows_fast.py`, `stamp_dedup.py`, `stamp_runner.py`, `fast_stamp_runner.py` | Sparse-row construction and experimental deduplication runners |
| `triton_kernels.py` | Opt-in first-layer kernel experiment |
| `continuous.py` | Continuous-coordinate VMC path |
| `spectral.py`, `diagnostics.py` | Spectrum and diagnostic calculations |
| `magnetoroton.py`, `magnetoroton_cli.py` | Monte Carlo magnetic-roton post-processing; separate entry point |
| `tests/` | CLI, physics, workflow, JAX, spectral, and post-processing regression tests |

The maintained source in this repository is `code/nqs_torus_laqx_nes/`.
Earlier local packages and A100 benchmark snapshots are historical material;
they are not edited in place. For input settings, see
[INPUT_PARAMETERS.md](INPUT_PARAMETERS.md).

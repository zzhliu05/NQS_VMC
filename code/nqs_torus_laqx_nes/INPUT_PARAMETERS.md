# Training inputs

训练入口：在仓库的 `code/` 目录运行
`python -m nqs_torus_laqx_nes --help`。磁旋子后处理使用独立入口
`python -m nqs_torus_laqx_nes.magnetoroton_cli --help`，参数见文末。
The CLI defaults below come from `cli.py`; they are not recommended settings
for every system. Flags are grouped by the configuration object they populate.

常用配置先确定三组参数：系统 `Nphi,p,q,tk`，波函数
`network_architecture,mps_tensor_generation,mps_bond_dim,hidden_sizes`，以及
采样/训练 `n_chains,n_burn,n_steps,optimizer,lr`。`forward_batch_size` 控制
波函数前向分块，`vmc_chain_batch_size` 控制训练链的微批；两者不是同一个量。
布尔开关只需写出标志名，未写时为 off。表中的 unset 表示 CLI 默认传入 `None`。

## System and sector

| Flags | Default | Meaning |
| --- | --- | --- |
| `--Nphi`, `--p`, `--q` | `24`, `1`, `3` | Number of orbitals and nominal filling `p/q`; code computes `Ne=(Nphi//q)*p`, so choose `Nphi` divisible by `q` for exact filling |
| `--tk`, `--all_sectors` | `0`, off | Momentum sector or run all sectors |
| `--ratio_rect`, `--landau_level` | `1.0`, `0` | Torus aspect ratio and Landau-level index |
| `--coulomb_cut`, `--disable_numba_coulomb` | automatic, off | Coulomb summation cutoff and CPU implementation switch |

## Network

| Flags | Default | Meaning |
| --- | --- | --- |
| `--k`, `--network_architecture` | `1`, `ace_nes_1d` | Output states and ansatz: `ace_nes_1d`, `mlp`, `mps`, `vit` |
| `--hidden_sizes`, `--activation` | `64 64`, `silu` | Hidden widths and activation (`silu`, `tanh`, `poly`, `relu`, `gelu`, `leaky_relu`) |
| `--no_layer_norm`, `--logphi_real_clip`, `--init_std` | off, `30`, `0.01` | Normalization, log-amplitude clipping, initialization scale |
| `--nes_hidden`, `--nes_layers`, `--nes_mlp_hidden`, `--nes_mlp_layers`, `--nes_cutoff` | `58`, `4`, `58`, `1`, `3` | ACE/NES widths, depths, and cutoff |
| `--vit_patch_size`, `--vit_embed_dim`, `--vit_num_heads`, `--vit_num_layers`, `--vit_mlp_ratio`, `--vit_fourier_modes` | `3`, `256`, `8`, `4`, `4`, `4` | ViT patch, width, attention, MLP, and Fourier settings |
| `--mps_tensor_generation` | `sitewise` | `direct`, `sitewise`, `tensor_bank`, `backflow`, `global_backflow`, `site_head_backflow` |
| `--mps_bond_dim`, `--mps_backflow_embed_dim` | `16`, `128` | MPS bond and embedding dimensions |
| `--mps_contraction` | `sequential` | `sequential`, `balanced`, `balanced_fp64`, `fused4`; `fused4` groups four sites before the balanced tree |
| `--mps_first_layer_backend` | `jax` | `jax` or experimental `triton` for Site Backflow first layer |
| `--mps_residual_blocks`, `--mps_attention_blocks`, `--mps_attention_heads` | `0`, `0`, `4` | Optional sequence blocks and attention heads |
| `--forward_batch_size`, `--forward_parallel_chunks` | `8192`, `1` | Wavefunction evaluation chunk size and parallel chunks |

`balanced_fp64` needs `JAX_ENABLE_X64=true` and is incompatible with
`mps_sitemajor`. `fused4` is selectable in both CLI and `NetworkConfig`.

## Sampling and training

| Flags | Default | Meaning |
| --- | --- | --- |
| `--sampler_mode` | `sector_uniform` | `sector_uniform`, `momentum_scattering`, `laqx_hamiltonian` |
| `--n_chains`, `--n_burn`, `--n_burn_warmup`, `--n_between`, `--n_samples` | `1024`, `20`, unset, `1`, `1` | MC chains, burn-in, first-step burn-in, spacing, samples |
| `--parallel_tempering`, `--tempering_replicas`, `--tempering_beta_min`, `--tempering_swap_interval` | off, `4`, `0.25`, `1` | Replica-exchange sampling; discrete VMC only |
| `--optimize_method`, `--n_steps`, `--min_steps` | `vmc`, `10000`, `20` | `projected`, `vmc`, or `continuous_vmc`; maximum/minimum steps |
| `--converge_tol`, `--converge_window`, `--loss_levels` | `0`, `10`, unset | Convergence control and number of loss levels |
| `--overlap_shift`, `--projection_rcond`, `--seed` | `1e-8`, `1e-10`, `1234` | Numerical regularization and RNG seed |
| `--skip_train`, `--perturb_steps`, `--perturb_scale` | off, `0`, `1e-4` | Training skip and early-step parameter perturbation |
| `--vmc_row_batch_size`, `--vmc_chain_batch_size` | unset, unset | Streamed Hamiltonian-row and chain microbatch sizes |
| `--vmc_fixed_neighbor_row_nnz`, `--vmc_neighbor_padding_bucket_size` | unset, unset | Fixed row width and neighbor-padding bucket |
| `--vmc_prebuild_sparse_rows`, `--vmc_sparse_rows_from_bitstrings` | off, off | Prebuild rows or build streamed rows directly from bitstrings |
| `--vmc_deduplicate_neighbors`, `--vmc_disable_neighbor_padding` | off, off | Neighbor deduplication and dynamic-size forward |
| `--vmc_timing_analysis`, `--vmc_timing_interval` | off, `10` | Extra timing instrumentation |

`--vmc_sparse_rows_from_bitstrings` requires discrete VMC, streamed rows,
`--vmc_chain_batch_size`, `sector_uniform`, `--skip_final_ritz`, and
`--prob_n_samples 0`; it excludes prebuilt rows, timing analysis, and
probability plots/snapshots. `--vmc_disable_neighbor_dedup` is a hidden
compatibility flag.

## Optimizer

| Flags | Default | Meaning |
| --- | --- | --- |
| `--optimizer`, `--lr` | `signed_gradient`, `1e-2` | `natural_gradient`, `signed_gradient`, `muon`, `adam`, `adamw`; base learning rate |
| `--constant_lr`, `--decay_rate`, `--multi_step_lr` | off, `1000`, off | Constant, decayed, or staged learning-rate schedule |
| `--cyclic_lr`, `--cyclic_lr_min`, `--cyclic_lr_period`, `--cyclic_lr_mode` | off, unset, `1000`, `triangular` | Cyclic schedule (`triangular` or `triangular2`) |
| `--sr_diag_shift`, `--cg_tol`, `--cg_maxiter`, `--max_sr_step_norm` | `1e-3`, `1e-8`, `2000`, `10` | Natural-gradient/SR regularization and solver limits |
| `--adam_b1`, `--adam_b2`, `--adam_eps`, `--adam_weight_decay` | `0.9`, `0.999`, `1e-8`, `0` | Adam-family hyperparameters |
| `--muon_beta`, `--muon_ns_steps`, `--muon_weight_decay`, `--muon_adaptive` | `0.95`, `5`, `0`, off | Muon hyperparameters |
| `--report_fisher_condition` | off | Fisher diagnostic |

## Continuous VMC

Only used with `--optimize_method continuous_vmc` (currently `--k 1`):
`--continuous_basis` (`fourier`, default; or `compact_overlap`),
`--continuous_harmonic` (`1`), `--continuous_overlap_epsilon` (`0.1`),
`--continuous_proposal_width` (`0.05`),
`--continuous_target_acceptance` (`0.5`), `--continuous_adapt_rate` (`0.1`),
`--continuous_min_width` (`1e-4`), `--continuous_max_width` (`0.5`), and
`--continuous_node_epsilon` (`1e-14`).

## Outputs and diagnostics

| Flags | Default | Meaning |
| --- | --- | --- |
| `--history_output`, `--params_output` | unset | CSV history and final parameter/config checkpoint |
| `--build_exact`, `--skip_final_ritz` | off, off | Exact-basis build and skip post-training full-sector Ritz evaluation |
| `--state_archive_output` | unset | Save sector Ritz vectors; incompatible with `--skip_final_ritz` |
| `--plot_output`, `--energy_curve_output`, `--prob_plot_output` | unset | Plot paths |
| `--live_plot_output`, `--live_plot_interval` | unset, `0` | Live plot path and update interval; positive interval needs path |
| `--prob_snapshot_dir`, `--prob_snapshot_steps` | unset, empty | Probability diagnostic snapshots |
| `--prob_n_burn`, `--prob_n_between`, `--prob_n_samples` | `2048`, `1`, `1` | Separate probability-diagnostic MC settings |

## Opt-in environment switches

| Variable | Effect |
| --- | --- |
| `NQS_NO_DEDUP=1` | Forward all neighbor entries rather than deduplicating them |
| `NQS_CONTIGUOUS_ROW_REDUCE=1` | Under `NQS_NO_DEDUP`, reshape fixed-width neighbor entries by row and sum; requires a fixed row width |
| `JAX_ENABLE_X64=true` | Enable JAX x64, required for `balanced_fp64` |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Disable default JAX GPU preallocation |

These environment checks use nonempty-string truthiness: unset a switch to
disable it; setting it to `0` is not equivalent to disabling it. The old
baseline script's `NQS_ROW_REDUCE=1` is not the switch used by the current
contiguous reduction code.

固定宽度 row reduction 只适用于 `NQS_NO_DEDUP` 分支，且同一批次中每行邻居
数必须相同；不要仅凭 `NQS_ROW_REDUCE` 以为它已启用。`fused4` 是 MPS 收缩方式，
不会自动打开 row reduction，也不保证在所有 GPU/批量上更快。

## Site Backflow example

```powershell
Set-Location code
python -m nqs_torus_laqx_nes --Nphi 39 --p 1 --q 3 --tk 0 `
  --network_architecture mps --mps_tensor_generation backflow `
  --mps_contraction fused4 --hidden_sizes 397 397 --mps_bond_dim 10 `
  --mps_backflow_embed_dim 128 --sampler_mode sector_uniform `
  --n_chains 4096 --vmc_chain_batch_size 4096 --n_steps 2000 `
  --optimizer natural_gradient --lr 0.005 --constant_lr `
  --forward_batch_size 1024 --skip_final_ritz `
  --history_output ../outputs/nphi39_example_history.csv `
  --params_output ../outputs/nphi39_example_params.pkl
```

The example selects fused4 but not contiguous row reduction. That reduction
requires fixed-width streamed rows and the two environment switches above.

## Magnetic-roton post-processing inputs

入口 `magnetoroton_cli` 读取训练时 `--params_output` 保存的 checkpoint，
从基态 sector 的 `|psi|^2` 采样，分别估计单模分子与密度范数。该入口已通过
小系统单元测试；尚无已验证的大系统产出。

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint`, `--output-dir` | required | Ground-state parameter checkpoint and result directory |
| `--q S T` | required, repeatable | Density momentum indices; one or more nonzero `(S,T)` pairs |
| `--ground-energy`, `--history` | unset | Ground-state energy directly, or mean of last 100 `history.csv` energies; history defaults beside checkpoint |
| `--source-tk` | first checkpoint sector | Source momentum sector |
| `--n-chains`, `--n-burn`, `--n-between`, `--n-samples` | `512`, `2000`, `100`, `8` | Source-state MCMC controls |
| `--density-draws`, `--hamiltonian-draws` | `2`, `32` | Sampled density and Hamiltonian paths per state/row |
| `--batch-size`, `--seed`, `--sample-cache` | `1024`, `20261022`, output-dir cache | Wavefunction evaluation batch, RNG seed, reusable source samples |

The target sector is `(source_tk + T) % Nphi`. Hamiltonian rows are built from
all allowed interaction transfers within that target sector, not from the
source-sector Hamiltonian alone. The CLI writes `results.json`, per-q sample
archives, and a source-sample cache. It currently requires a single-state
checkpoint (`k=1`).

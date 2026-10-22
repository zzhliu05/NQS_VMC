# `nqs_torus_laqx_nes`

New torus research package that reuses the current Landau-gauge Coulomb
Hamiltonian, basis construction, sampling, optimizer, and workflow from the
validated torus code path, while switching the default ansatz to a
LaQX-inspired Neural Excited States network adapted to the 1D momentum
occupation basis.

Run from the repository's `code` directory with a Python environment that
contains JAX and the project dependencies:

```powershell
Set-Location code
python -m nqs_torus_laqx_nes --help
```

Python API:

```python
from nqs_torus_laqx_nes.config import RunConfig
from nqs_torus_laqx_nes.workflow import run

result = run(RunConfig())
```

## Optional MPS performance paths

See [INPUT_PARAMETERS.md](INPUT_PARAMETERS.md) for all training and
magnetic-roton inputs, defaults, constraints, and an example command.

Use `--mps_contraction fused4` to contract four-site blocks before the
balanced MPS tree. For fixed-width, non-deduplicated Hamiltonian rows, set
both `NQS_NO_DEDUP=1` and `NQS_CONTIGUOUS_ROW_REDUCE=1` to sum contiguous
neighbor contributions by row instead of using a scatter reduction. These
paths are opt-in; the defaults are unchanged. The historical
`NQS_ROW_REDUCE=1` baseline setting alone does not select contiguous row
reduction in this package.

## Default ansatz

The default architecture is:

```bash
--network_architecture ace_nes_1d
--k 1
--nes_hidden 58
--nes_layers 4
--nes_mlp_hidden 58
--nes_mlp_layers 1
--nes_cutoff 3
--activation silu
```

This is a torus adaptation of the LaQX ACE-NES idea:

- embed each momentum occupation `n_k in {0,1}` sitewise;
- apply periodic 1D convolutional residual blocks on the momentum ring;
- refine site features with an MLP;
- output one complex orbital matrix per optimized state;
- evaluate state amplitudes with a determinant readout over the occupied set.

For JAX shape stability, the determinant is implemented as a full `Nphi x Nphi`
matrix with unoccupied rows and columns patched to identity, which is
equivalent to taking the occupied principal submatrix determinant.

## Baseline ansatzes

The older baselines remain available:

```bash
--network_architecture mlp
```

or

```bash
--network_architecture vit
--vit_patch_size 3
--vit_embed_dim 256
--vit_num_heads 8
--vit_num_layers 4
--vit_mlp_ratio 4
--vit_fourier_modes 4
```

The neural MPS ansatz is also available:

```bash
--network_architecture mps
--hidden_sizes 64 64
--mps_bond_dim 16
--mps_tensor_generation sitewise
```

For an occupation string `sigma=(sigma_0,...,sigma_{Nphi-1})`, a shared MLP
takes the momentum-site one-hot vector and the local occupation one-hot vector
and emits the real and imaginary entries of `A_j^{sigma_j}`.  The virtual
indices use periodic boundary conditions, matching the orbital ring.  The
amplitude is

```text
psi(sigma) = Tr[A_0^{sigma_0} A_1^{sigma_1} ... A_{Nphi-1}^{sigma_{Nphi-1}}].
```

Every site tensor is a full `D x D` matrix.  There are no separate first or
last boundary vectors.

Three tensor-generation modes are available. `sitewise` uses only the site
one-hot vector and local occupation. `tensor_bank` evaluates the same local
ansatz once for all `(j, sigma_j)` pairs. `backflow` instead feeds
`[one_hot(j), sigma_0, ..., sigma_(Nphi-1)]` to the shared tensor MLP, so every
`A_j^{sigma_j}` depends on the complete occupation configuration. The
periodic trace contraction is unchanged.

`global_backflow` removes the site index from the network input. It maps the
full occupation vector directly to all selected tensors with output shape
`[Nphi, k, D, D]`. This evaluates the hidden MLP once per configuration while
retaining full-configuration dependence in every site tensor.

`site_head_backflow` evaluates a wide global trunk once per configuration,
then applies learned `(site, local occupation)` FiLM scale and shift vectors
and a lightweight shared site head. It preserves nonlinear site conditioning
without evaluating the full hidden MLP once per site.

The per-site `backflow` mode uses
`[one_hot(j), one_hot(sigma_j), sigma]`, embeds this vector to
`mps_backflow_embed_dim`, and then applies the configured tensor MLP. During
discrete VMC its local energy is evaluated from sampled Hamiltonian neighbors
with global neighbor deduplication and fixed-size padding; it does not require
a full-sector wavefunction forward pass.

## Nphi=39 no-cache performance baseline

The historical performance baseline was defined by an external local script
not included in this repository:

```text
seeta_nocache_sweep.sh
```

It runs the remaining `tk=26..38` sectors of the `Nphi=39`, `nu=1/3`
Site Backflow MPS sweep. A sector with more than 490 rows already present in
`history.csv` is treated as complete and skipped. The baseline configuration
is:

```bash
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export NUMEXPR_NUM_THREADS=18
export NUMBA_NUM_THREADS=18
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export NQS_KEEP_OCCUPATIONS=0
export NQS_MATERIALIZE_SECTOR=0
export NQS_NO_DEDUP=1
export NQS_ROW_REDUCE=1
export NQS_ROW_CACHE_MAX=20480
export NQS_DISABLE_ROW_CACHE=1

python -u -m nqs_torus_laqx_nes \
  --Nphi 39 --p 1 --q 3 --tk "${tk}" --k 1 --landau_level 0 \
  --network_architecture mps --mps_tensor_generation backflow \
  --mps_contraction balanced --hidden_sizes 397 397 \
  --mps_bond_dim 10 --init_std 0.01 --activation silu --seed 392039 \
  --optimize_method vmc --sampler_mode sector_uniform \
  --n_chains 4096 --n_burn 40 --n_between 1 --n_samples 1 \
  --n_steps 500 --min_steps 500 \
  --optimizer natural_gradient --lr 5e-3 --constant_lr \
  --sr_diag_shift 1e-3 --max_sr_step_norm 10 \
  --vmc_row_batch_size 4096 --vmc_chain_batch_size 4096 \
  --vmc_sparse_rows_from_bitstrings --vmc_deduplicate_neighbors \
  --vmc_fixed_neighbor_row_nnz 672 \
  --vmc_neighbor_padding_bucket_size 8192 \
  --forward_batch_size 1024 --forward_parallel_chunks 1 \
  --prob_n_samples 0 --skip_final_ritz \
  --history_output "${OUT}/history.csv"
```

This baseline deliberately disables the sparse-row cache with
`NQS_DISABLE_ROW_CACHE=1`. Performance comparisons must preserve all settings
above unless the changed variable is explicitly named. In particular, do not
compare cached and no-cache measurements as if they were the same baseline.

To evaluate FP64 MPS contraction, keep `JAX_ENABLE_X64=true` and replace
`--mps_contraction balanced` with `--mps_contraction balanced_fp64`. This casts
the generated complex64 matrices to complex128 immediately before the balanced
contraction. The BF16 tensor-generating MLP is unchanged, so this tests only
contraction precision and speed; the default path remains complex64.

At initialization, the embedding weights connected to `one_hot(j)` and
`one_hot(sigma_j)` are random, while every weight connected to the full
occupation vector is exactly zero. The initial ansatz therefore has no
backflow dependence; optimization can grow those initially-zero weights.

For VMC, `--vmc_sparse_rows_from_bitstrings` opts into streamed Hamiltonian
rows whose off-diagonal targets are packed neighbor bitstrings rather than
sector positions. The CPU builder applies the scattering terms directly to
sampled bitstrings, merges equal targets, and never builds the sorted
neighbor-to-sector-position lookup. Only the sampled rows and neighbor
bitstrings are sent to the accelerator for local-energy evaluation. The CPU
sector bitstring list remains available to the `sector_uniform` sampler; this
option removes the neighbor lookup table, not the sampler's basis. It requires
`--vmc_row_batch_size`, `--vmc_chain_batch_size`, `--skip_final_ritz`,
`--prob_n_samples 0`, no probability plots/snapshots, and currently cannot be
combined with `--vmc_prebuild_sparse_rows` or `--vmc_timing_analysis`. The
existing sector-index row path remains the default and is unchanged.

The contraction is normalized after every site while its logarithmic scale is
tracked separately, preventing underflow and overflow without changing the
represented amplitude.  For `k > 1`, the generator emits one independent MPS
matrix channel per NQS output; the existing determinant multi-state readout is
then applied unchanged.

`hidden_sizes`, `activation`, `layer_norm`, `init_std`, and
`logphi_real_clip` control the MPS tensor-generating MLP.  For `k=1`,
`--optimizer natural_gradient` uses MPS KFAC: full Kronecker factors on the
hidden layers and a diagonal output-gradient factor on the `2 D^2` output
layer.  The output approximation avoids an impractical dense
`(2 D^2) x (2 D^2)` inverse at every optimization step.

The generator can also be augmented for architecture comparisons:

```bash
--mps_residual_blocks 1
--mps_attention_blocks 1
--mps_attention_heads 4
```

Residual blocks add identity skips between equal-width hidden layers.
Attention blocks are pre-normalized multi-head self-attention over the site
tokens and are currently available for `sitewise` and `backflow` generation.
The attention width is the first tensor-generator hidden width (the backflow
embedding width in `backflow`).  These options are disabled by default, so
existing runs are unchanged.  The legacy MPS KFAC implementation does not yet
include the new residual/attention Jacobian paths; use Adam, AdamW, Muon, or
signed-gradient while testing these variants.

## Physics and workflow

This project keeps the current torus implementation for:

- momentum-sector basis construction;
- Landau-gauge torus Coulomb matrix generation;
- sparse Hamiltonian rows;
- projected Ritz diagonalization;
- discrete VMC;
- continuous occupation-space VMC;
- momentum-scattering Metropolis proposals;
- LaQX-style Hamiltonian-weighted Metropolis proposals;
- row-streamed local-energy evaluation and timing diagnostics.

So the main change in this package is the ansatz family, not the Hamiltonian or
the training workflow.

## Example

Projected NES run:

```powershell
python -m nqs_torus_laqx_nes `
  --Nphi 18 `
  --p 1 `
  --q 3 `
  --tk 0 `
  --k 4 `
  --network_architecture ace_nes_1d `
  --optimize_method projected `
  --optimizer adamw `
  --lr 2e-3 `
  --build_exact
```

Discrete VMC with the same ansatz:

```powershell
python -m nqs_torus_laqx_nes `
  --Nphi 18 `
  --p 1 `
  --q 3 `
  --tk 0 `
  --k 4 `
  --network_architecture ace_nes_1d `
  --optimize_method vmc `
  --sampler_mode momentum_scattering `
  --n_chains 4096 `
  --n_burn 40 `
  --optimizer adamw `
  --lr 2e-3
```

Use LaQX's Hamiltonian-driven proposal rule with:

```powershell
--sampler_mode laqx_hamiltonian
```

For each current occupation state, this mode chooses an off-diagonal sparse
Hamiltonian neighbor with weight `max(abs(H_ij), 0.05)` and includes the exact
forward/reverse proposal normalization in the Metropolis-Hastings ratio. Full
sparse rows must be prebuilt for this sampler.

# Density spectral weight

An all-sector run can export normalized projected NQS Ritz vectors in the
occupation basis:

```bash
python -m nqs_torus_laqx_nes --all_sectors --k 2 [training options] \
  --state_archive_output outputs/nphi18_k2_states.npz
```

The archive is sufficient to evaluate density matrix elements without
rerunning the neural network.  Compute the `qx=0` momentum cut with:

```bash
python -m nqs_torus_laqx_nes.spectral \
  --state_archive outputs/nphi18_k2_states.npz \
  --output_prefix outputs/nphi18_k2_qx0 \
  --source_tk 3 --source_level 0 --qx_indices 0
```

The program implements the Landau-gauge guiding-center density convention

```text
rho_bar(s,t) = sum_j exp[-2 pi i s (j+t/2) / Nphi] c^dagger[j+t] c[j].
```

Here `s=qx_index` is the orbital phase index and `t=qy_index` shifts the
stored orbital-momentum sector by `t` modulo `Nphi`.  The detailed CSV reports
both `|<n|rho_bar(q)|0>|^2 / Ne` and the physical LLL-projected density weight,
which includes the squared form factor `exp(-q^2 lB^2/2)`.  The sum-rule CSV
compares the weight captured by the archived `k` Ritz states with the exact
static norm of `rho_bar(q)|0>` in the full target occupation sector.  A small
captured fraction means that more NQS excited states are needed; it is not a
normalization error.

For the threefold torus ground-state manifold, run the spectral command once
for each ground sector and average the resulting dynamic structure factors.
Do not coherently add amplitudes from independently trained ground sectors.

"""Optional Pallas/Triton kernels for forward-only MPS evaluation.

These kernels are experimental and are not used by the default training path.
Keeping them in a separate module makes rollback equivalent to leaving the
standard JAX backend selected.
"""

from __future__ import annotations

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def pallas_triton_available() -> bool:
    """Return whether this JAX build exposes the Pallas Triton backend."""

    return hasattr(plgpu, "CompilerParams")


def _structured_backflow_first_layer_kernel(
    states_ref,
    weights_ref,
    bias_ref,
    output_ref,
    *,
    num_sites: int,
    block_batch: int,
    block_hidden: int,
    block_k: int,
    apply_silu: bool,
):
    """Compute SiLU(W [site, local occupation, full state] + b).

    The full-state projection is evaluated once per physical state and reused
    for every site token.  The site and local-occupation one-hot projections
    are loaded directly from the first-layer weight matrix, so the expanded
    ``(batch, num_sites, 2 * num_sites + 2)`` input is never materialized.
    """

    batch_start = pl.program_id(0) * block_batch
    hidden_start = pl.program_id(1) * block_hidden
    batch_idx = batch_start + jnp.arange(block_batch)
    hidden_idx = hidden_start + jnp.arange(block_hidden)
    batch_mask = batch_idx < states_ref.shape[0]
    hidden_mask = hidden_idx < weights_ref.shape[1]

    accumulator = jnp.zeros(
        (block_batch, block_hidden), dtype=jnp.float32
    )
    for k_start in range(0, num_sites, block_k):
        site_idx = k_start + jnp.arange(block_k)
        site_mask = site_idx < num_sites
        state_tile = plgpu.load(
            states_ref.at[batch_idx[:, None], site_idx[None, :]],
            mask=batch_mask[:, None] & site_mask[None, :],
            other=0.0,
        ).astype(jnp.bfloat16)
        weight_tile = plgpu.load(
            weights_ref.at[
                (num_sites + 2 + site_idx)[:, None],
                hidden_idx[None, :],
            ],
            mask=site_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        ).astype(jnp.bfloat16)
        accumulator += state_tile @ weight_tile

    bias = plgpu.load(
        bias_ref.at[hidden_idx],
        mask=hidden_mask,
        other=0.0,
    )
    local_zero = plgpu.load(
        weights_ref.at[
            num_sites + jnp.zeros_like(hidden_idx), hidden_idx
        ],
        mask=hidden_mask,
        other=0.0,
    )
    local_one = plgpu.load(
        weights_ref.at[
            num_sites + 1 + jnp.zeros_like(hidden_idx), hidden_idx
        ],
        mask=hidden_mask,
        other=0.0,
    )

    def write_site(site, unused):
        del unused
        site_weight = plgpu.load(
            weights_ref.at[
                site + jnp.zeros_like(hidden_idx), hidden_idx
            ],
            mask=hidden_mask,
            other=0.0,
        )
        occupation = plgpu.load(
            states_ref.at[
                batch_idx, site + jnp.zeros_like(batch_idx)
            ],
            mask=batch_mask,
            other=0.0,
        )
        local_weight = jnp.where(
            occupation[:, None] > 0.5,
            local_one[None, :],
            local_zero[None, :],
        )
        value = (
            accumulator.astype(jnp.bfloat16)
            + bias[None, :]
            + site_weight[None, :]
            + local_weight
        )
        if apply_silu:
            value_f32 = value.astype(jnp.float32)
            value = (
                value_f32 * jax.nn.sigmoid(value_f32)
            ).astype(jnp.bfloat16)
        plgpu.store(
            output_ref.at[
                batch_idx[:, None],
                site + jnp.zeros((block_batch, block_hidden), jnp.int32),
                hidden_idx[None, :],
            ],
            value,
            mask=batch_mask[:, None] & hidden_mask[None, :],
        )
        return None

    lax.fori_loop(0, num_sites, write_site, None)


@partial(
    jax.jit,
    static_argnames=(
        "num_sites",
        "block_batch",
        "block_hidden",
        "block_k",
        "num_warps",
        "num_stages",
        "apply_silu",
        "interpret",
    ),
)
def structured_backflow_first_layer_triton(
    states,
    weights,
    bias,
    *,
    num_sites: int,
    block_batch: int = 16,
    block_hidden: int = 64,
    block_k: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    apply_silu: bool = True,
    interpret: bool = False,
):
    """Fused first layer for Site Backflow MPS neighbor inference."""

    states = jnp.asarray(states, dtype=jnp.bfloat16)
    weights = jnp.asarray(weights, dtype=jnp.bfloat16)
    bias = jnp.asarray(bias, dtype=jnp.bfloat16)
    if states.ndim != 2 or states.shape[1] != num_sites:
        raise ValueError("states must have shape (batch, num_sites)")
    if weights.shape[0] != 2 * num_sites + 2:
        raise ValueError("weights have the wrong Site Backflow input size")
    if bias.shape != (weights.shape[1],):
        raise ValueError("bias shape must match the hidden dimension")

    output_shape = jax.ShapeDtypeStruct(
        (states.shape[0], num_sites, weights.shape[1]), jnp.bfloat16
    )
    kernel = partial(
        _structured_backflow_first_layer_kernel,
        num_sites=num_sites,
        block_batch=block_batch,
        block_hidden=block_hidden,
        block_k=block_k,
        apply_silu=apply_silu,
    )
    return pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid=(
            pl.cdiv(states.shape[0], block_batch),
            pl.cdiv(weights.shape[1], block_hidden),
        ),
        compiler_params=plgpu.CompilerParams(
            num_warps=num_warps,
            num_stages=num_stages,
        ),
        interpret=interpret,
        name="structured_backflow_first_layer",
    )(states, weights, bias)


def structured_backflow_preactivation_jax(states, weights, bias):
    """Structured preactivation matching the expanded first-layer GEMM."""

    num_sites = states.shape[1]
    occupation = jnp.clip(jnp.rint(states), 0, 1).astype(jnp.int32)
    global_projection = (
        states.astype(weights.dtype) @ weights[num_sites + 2 :, :]
    )
    site_projection = weights[:num_sites, :]
    local_projection = weights[num_sites : num_sites + 2, :][occupation]
    return (
        global_projection[:, None, :]
        + site_projection[None, :, :]
        + local_projection
        + bias[None, None, :]
    )


def _structured_backflow_preactivation_triton_impl(states, weights, bias):
    return structured_backflow_first_layer_triton(
        states,
        weights,
        bias,
        num_sites=int(states.shape[1]),
        block_batch=8,
        block_hidden=32,
        block_k=16,
        num_warps=4,
        num_stages=1,
        apply_silu=False,
    )


@jax.custom_vjp
def structured_backflow_preactivation_triton(states, weights, bias):
    """Trainable Triton forward with an equivalent structured JAX VJP."""

    return _structured_backflow_preactivation_triton_impl(
        states, weights, bias
    )


def _structured_backflow_preactivation_triton_fwd(states, weights, bias):
    value = _structured_backflow_preactivation_triton_impl(
        states, weights, bias
    )
    return value, (states, weights, bias)


def _structured_backflow_preactivation_triton_bwd(residual, cotangent):
    states, weights, bias = residual
    _, pullback = jax.vjp(
        structured_backflow_preactivation_jax, states, weights, bias
    )
    return pullback(cotangent)


structured_backflow_preactivation_triton.defvjp(
    _structured_backflow_preactivation_triton_fwd,
    _structured_backflow_preactivation_triton_bwd,
)


@jax.jit
def structured_backflow_first_layer_jax(states, weights, bias):
    """Reference implementation used for correctness and timing checks."""

    return jax.nn.silu(
        structured_backflow_preactivation_jax(states, weights, bias)
    )

"""Fixed-sector Metropolis sampling for determinant NQS states."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .config import NetworkConfig
from .network import (
    deduplicated_wavefunction_matrix_from_positions,
    wavefunction_matrix,
    occupation_states_from_positions,
)

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration.
    njit = None
    prange = range
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class MomentumScatteringTable:
    pairs: np.ndarray
    masks: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class AllowedOutgoingPairsCPU:
    """CPU-only CSR table keyed by (sector basis state, total momentum)."""

    nphi: int
    bitstrings: np.ndarray
    sorted_bitstrings: np.ndarray
    sorted_positions: np.ndarray
    occupied_orbitals: np.ndarray
    occupied_pair_slots: np.ndarray
    offsets: np.ndarray
    outgoing_masks: np.ndarray
    max_row_nnz: int

    def bounds(self, position: int, total_momentum: int) -> tuple[int, int]:
        row = int(position) * self.nphi + int(total_momentum)
        return int(self.offsets[row]), int(self.offsets[row + 1])

    @property
    def nbytes(self) -> int:
        return int(
            self.bitstrings.nbytes
            + self.sorted_bitstrings.nbytes
            + self.sorted_positions.nbytes
            + self.occupied_orbitals.nbytes
            + self.occupied_pair_slots.nbytes
            + self.offsets.nbytes
            + self.outgoing_masks.nbytes
        )


@dataclass(frozen=True)
class LaQXHamiltonianProposalTable:
    """CPU proposal table mirroring LaQX's Hamiltonian-weighted MCMC."""

    cols: np.ndarray
    weights: np.ndarray
    cumulative_weights: np.ndarray
    weight_sums: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(
            self.cols.nbytes
            + self.weights.nbytes
            + self.cumulative_weights.nbytes
            + self.weight_sums.nbytes
        )


def build_laqx_hamiltonian_proposal_table(
    sparse_rows,
    weight_floor: float = 0.05,
) -> LaQXHamiltonianProposalTable:
    """Build LaQX-style proposal weights from full sparse Hamiltonian rows."""
    if sparse_rows is None:
        raise ValueError(
            "laqx_hamiltonian sampling requires prebuilt full sparse rows."
        )
    cols = np.asarray(sparse_rows.cols, dtype=np.int64)
    vals = np.asarray(sparse_rows.vals, dtype=np.complex128)
    mask = np.asarray(sparse_rows.mask, dtype=np.bool_)
    if cols.ndim != 2 or vals.shape != cols.shape or mask.shape != cols.shape:
        raise ValueError("sparse Hamiltonian rows must share a rank-2 shape.")
    weights = np.where(mask, np.maximum(np.abs(vals), weight_floor), 0.0)
    cumulative = np.cumsum(weights, axis=1)
    return LaQXHamiltonianProposalTable(
        cols=cols,
        weights=weights,
        cumulative_weights=cumulative,
        weight_sums=np.sum(weights, axis=1),
    )


@lru_cache(maxsize=None)
def build_momentum_scattering_table(
    nphi: int,
) -> MomentumScatteringTable:
    """Group unordered orbital pairs by total momentum modulo Nphi."""
    if nphi < 2:
        raise ValueError("momentum scattering requires Nphi >= 2.")
    grouped = [[] for _ in range(nphi)]
    for first in range(nphi):
        for second in range(first + 1, nphi):
            grouped[(first + second) % nphi].append((first, second))
    counts = np.asarray([len(group) for group in grouped], dtype=np.int64)
    max_count = int(np.max(counts))
    pairs = np.zeros((nphi, max_count, 2), dtype=np.int64)
    masks = np.zeros((nphi, max_count), dtype=np.int64)
    for total, group in enumerate(grouped):
        pairs[total, : len(group)] = np.asarray(group, dtype=np.int64)
        for slot, (first, second) in enumerate(group):
            masks[total, slot] = (
                np.left_shift(np.int64(1), first)
                | np.left_shift(np.int64(1), second)
            )
    return MomentumScatteringTable(
        pairs=pairs,
        masks=masks,
        counts=counts,
    )


@lru_cache(maxsize=None)
def build_pair_slots(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("momentum scattering requires at least two particles.")
    return np.asarray(
        [
            (first, second)
            for first in range(count)
            for second in range(first + 1, count)
        ],
        dtype=np.int64,
    )


def build_allowed_outgoing_pairs_cpu(
    sector_states: np.ndarray,
) -> AllowedOutgoingPairsCPU:
    """Precompute all Pauli-allowed outgoing pairs in host memory."""
    states = np.asarray(sector_states, dtype=np.int8)
    if states.ndim != 2:
        raise ValueError("sector_states must be a rank-2 occupation array.")
    sector_size, nphi = states.shape
    particle_counts = np.sum(states, axis=1).astype(np.int64)
    if sector_size == 0 or not np.all(particle_counts == particle_counts[0]):
        raise ValueError("sector states must have a fixed particle count.")
    n_particles = int(particle_counts[0])
    pair_slots = build_pair_slots(n_particles)
    occupied_orbitals = np.argsort(-states, axis=1)[
        :, :n_particles
    ].astype(np.int64)
    powers = np.left_shift(
        np.int64(1), np.arange(nphi, dtype=np.int64)
    )
    bitstrings = np.sum(
        states.astype(np.int64) * powers[None, :],
        axis=1,
        dtype=np.int64,
    )
    order = np.argsort(bitstrings)
    scattering = build_momentum_scattering_table(nphi)
    offsets = np.zeros(sector_size * nphi + 1, dtype=np.int64)
    outgoing_masks = []
    cursor = 0
    for position, bits in enumerate(bitstrings):
        for total_momentum in range(nphi):
            count = int(scattering.counts[total_momentum])
            masks = scattering.masks[total_momentum, :count]
            allowed = masks[(bits & masks) == 0]
            outgoing_masks.extend(int(mask) for mask in allowed)
            cursor += int(allowed.shape[0])
            offsets[position * nphi + total_momentum + 1] = cursor
    if NUMBA_AVAILABLE:
        row_nnz = _allowed_row_nnz_numba(
            occupied_orbitals, pair_slots, offsets, nphi
        )
    else:  # pragma: no cover - production uses Numba.
        row_nnz = np.zeros(sector_size, dtype=np.int64)
        for position in range(sector_size):
            for left, right in pair_slots:
                total_momentum = int(
                    occupied_orbitals[position, left]
                    + occupied_orbitals[position, right]
                ) % nphi
                row = position * nphi + total_momentum
                row_nnz[position] += offsets[row + 1] - offsets[row]
    return AllowedOutgoingPairsCPU(
        nphi=nphi,
        bitstrings=bitstrings,
        sorted_bitstrings=bitstrings[order],
        sorted_positions=order.astype(np.int64),
        occupied_orbitals=occupied_orbitals,
        occupied_pair_slots=pair_slots,
        offsets=offsets,
        outgoing_masks=np.asarray(outgoing_masks, dtype=np.int64),
        max_row_nnz=int(np.max(row_nnz, initial=0)),
    )


if NUMBA_AVAILABLE:

    @njit(cache=True, parallel=True)
    def _allowed_row_nnz_numba(
        occupied_orbitals, pair_slots, offsets, nphi
    ):
        sector_size = occupied_orbitals.shape[0]
        row_nnz = np.zeros(sector_size, dtype=np.int64)
        for position in prange(sector_size):
            count = 0
            for pair_index in range(pair_slots.shape[0]):
                left = pair_slots[pair_index, 0]
                right = pair_slots[pair_index, 1]
                total_momentum = (
                    occupied_orbitals[position, left]
                    + occupied_orbitals[position, right]
                ) % nphi
                row = position * nphi + total_momentum
                count += offsets[row + 1] - offsets[row]
            row_nnz[position] = count
        return row_nnz

    @njit(cache=True, parallel=True)
    def _momentum_scattering_proposals_numba(
        positions,
        rows,
        incoming_slots,
        outgoing_uniform,
        nphi,
        bitstrings,
        sorted_bitstrings,
        sorted_positions,
        occupied_orbitals,
        occupied_pair_slots,
        offsets,
        outgoing_masks,
    ):
        n_chains, k = positions.shape
        proposal = positions.copy()
        valid = np.zeros(n_chains, dtype=np.bool_)
        log_hastings = np.zeros(n_chains, dtype=np.float64)
        sorted_size = sorted_bitstrings.shape[0]
        outgoing_size = outgoing_masks.shape[0]

        for chain in prange(n_chains):
            row = rows[chain]
            current_position = positions[chain, row]
            pair_slot = occupied_pair_slots[incoming_slots[chain]]
            incoming_first = occupied_orbitals[
                current_position, pair_slot[0]
            ]
            incoming_second = occupied_orbitals[
                current_position, pair_slot[1]
            ]
            total_momentum = (
                incoming_first + incoming_second
            ) % nphi
            csr_row = current_position * nphi + total_momentum
            start = offsets[csr_row]
            stop = offsets[csr_row + 1]
            forward_count = stop - start
            if forward_count <= 0 or outgoing_size == 0:
                continue

            outgoing_slot = start + int(
                outgoing_uniform[chain] * forward_count
            )
            if outgoing_slot >= stop:
                outgoing_slot = stop - 1
            outgoing_mask = outgoing_masks[outgoing_slot]
            incoming_mask = (
                (np.int64(1) << incoming_first)
                | (np.int64(1) << incoming_second)
            )
            proposed_bits = (
                bitstrings[current_position] & ~incoming_mask
            ) | outgoing_mask

            left = 0
            right = sorted_size
            while left < right:
                middle = (left + right) // 2
                if sorted_bitstrings[middle] < proposed_bits:
                    left = middle + 1
                else:
                    right = middle
            if (
                left >= sorted_size
                or sorted_bitstrings[left] != proposed_bits
            ):
                continue
            candidate = sorted_positions[left]
            reverse_row = candidate * nphi + total_momentum
            reverse_count = (
                offsets[reverse_row + 1] - offsets[reverse_row]
            )
            if reverse_count <= 0:
                continue

            duplicate = False
            for column in range(k):
                if column != row and positions[chain, column] == candidate:
                    duplicate = True
                    break
            if duplicate:
                continue

            proposal[chain, row] = candidate
            valid[chain] = True
            log_hastings[chain] = (
                np.log(forward_count) - np.log(reverse_count)
            )
        return proposal, valid, log_hastings
else:  # pragma: no cover - import compatibility without Numba.
    _momentum_scattering_proposals_numba = None


def logprob_from_positions(
    params, sector_states, positions, net_cfg: NetworkConfig
):
    positions = jnp.atleast_2d(jnp.asarray(positions, dtype=jnp.int64))
    if (
        net_cfg.architecture == "mps"
        and net_cfg.mps_tensor_generation
        in {"backflow", "site_head_backflow"}
    ):
        phi_flat = deduplicated_wavefunction_matrix_from_positions(
            params, sector_states, positions, net_cfg
        )
    else:
        states = occupation_states_from_positions(
            sector_states, positions.reshape((-1,)), net_cfg
        )
        phi_flat = wavefunction_matrix(params, states, net_cfg)
    psi_mats = phi_flat.reshape(
        (positions.shape[0], net_cfg.k, net_cfg.k)
    )
    _, logabsdet = jnp.linalg.slogdet(psi_mats)
    return 2.0 * logabsdet


@partial(
    jax.jit,
    static_argnames=("net_cfg", "n_chains", "n_target_chains"),
)
def _sector_uniform_step_fn(
    params,
    sector_states,
    key,
    positions,
    logprob,
    chain_betas,
    net_cfg: NetworkConfig,
    n_chains: int,
    n_target_chains: int,
):
    row_key, cand_key, u_key = jax.random.split(key, 3)
    row = jax.random.randint(
        row_key,
        shape=(n_chains,),
        minval=0,
        maxval=net_cfg.k,
        dtype=jnp.int64,
    )
    candidate = jax.random.randint(
        cand_key,
        shape=(n_chains,),
        minval=0,
        maxval=sector_states.shape[0],
        dtype=jnp.int64,
    )

    chain_ids = jnp.arange(n_chains)
    proposal = positions.at[chain_ids, row].set(candidate)
    columns = jnp.arange(net_cfg.k)[None, :]
    duplicate_other_row = jnp.any(
        (positions == candidate[:, None]) & (columns != row[:, None]), axis=1
    )
    unchanged = candidate == positions[chain_ids, row]
    valid = (~duplicate_other_row) & (~unchanged)

    prop_logprob = logprob_from_positions(
        params, sector_states, proposal, net_cfg
    )
    logu = jnp.log(
        jax.random.uniform(u_key, shape=(n_chains,), dtype=jnp.float64)
    )
    accept = (logu < chain_betas * (prop_logprob - logprob)) & valid
    new_positions = jnp.where(accept[:, None], proposal, positions)
    new_logprob = jnp.where(accept, prop_logprob, logprob)
    accept_rate = jnp.sum(accept[:n_target_chains]) / jnp.maximum(
        1, n_target_chains
    )
    return new_positions, new_logprob, accept_rate


@partial(jax.jit, static_argnames=("net_cfg",))
def _logprob_from_packed_batches(params, packed_batches, net_cfg: NetworkConfig):
    """Evaluate log probabilities after a CPU-side sector-table gather."""
    packed_batches = jnp.asarray(packed_batches, dtype=jnp.uint64)
    if packed_batches.ndim == 1:
        packed_batches = packed_batches[:, None]
    m, k = packed_batches.shape
    shifts = jnp.arange(net_cfg.num_sites, dtype=jnp.uint64)
    occupations = (
        (packed_batches[..., None] >> shifts) & jnp.uint64(1)
    ).astype(jnp.float32)
    phi_flat = wavefunction_matrix(
        params, occupations.reshape((m * k, net_cfg.num_sites)), net_cfg
    )
    psi_mats = phi_flat.reshape((m, k, net_cfg.k))
    _, logabsdet = jnp.linalg.slogdet(psi_mats)
    return 2.0 * logabsdet


@partial(
    jax.jit,
    static_argnames=("net_cfg", "n_chains", "n_target_chains"),
)
def _accept_host_proposals_fn(
    params,
    sector_states,
    key,
    positions,
    logprob,
    proposal,
    valid,
    log_hastings,
    chain_betas,
    net_cfg: NetworkConfig,
    n_chains: int,
    n_target_chains: int,
):
    prop_logprob = logprob_from_positions(
        params, sector_states, proposal, net_cfg
    )
    logu = jnp.log(
        jax.random.uniform(key, shape=(n_chains,), dtype=jnp.float64)
    )
    accept = (
        logu < (chain_betas * (prop_logprob - logprob) + log_hastings)
    ) & valid
    new_positions = jnp.where(accept[:, None], proposal, positions)
    new_logprob = jnp.where(accept, prop_logprob, logprob)
    accept_rate = jnp.sum(accept[:n_target_chains]) / jnp.maximum(
        1, n_target_chains
    )
    return new_positions, new_logprob, accept_rate


def sample_initial_positions_cpu(
    rng: np.random.Generator,
    n_chains: int,
    sector_size: int,
    k: int,
) -> np.ndarray:
    """Sample distinct sector indices using O(n_chains * k) storage."""
    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    if k <= 0 or k > sector_size:
        raise ValueError("k must lie in [1, sector_size].")
    if k == 1:
        return rng.integers(
            0, sector_size, size=(n_chains, 1), dtype=np.int64
        )

    positions = np.empty((n_chains, k), dtype=np.int64)
    for chain in range(n_chains):
        positions[chain] = rng.choice(sector_size, size=k, replace=False)
    return positions


class MetropolisSamplerJAX:
    """Metropolis sampler inside one momentum sector."""

    def __init__(
        self,
        params,
        sector_states,
        n_chains: int,
        net_cfg: NetworkConfig,
        seed: int = 1234,
        proposal_mode: str = "sector_uniform",
        sparse_rows=None,
        tempering_betas: tuple[float, ...] | None = None,
        tempering_swap_interval: int = 1,
        sector_states_host=None,
        sector_nphi: int | None = None,
        sector_n_particles: int | None = None,
        sector_on_host: bool = False,
    ):
        self.params = params
        self.net_cfg = net_cfg
        # Occupation states are binary; keep the full basis table compact on
        # the accelerator and cast selected batches at the network boundary.
        # Reuse a caller-owned device array when available so training and the
        # sampler do not retain two complete copies of a large sector table.
        self.sector_states_host = None
        self.sector_on_host = bool(sector_on_host)
        if self.sector_on_host:
            if proposal_mode != "sector_uniform":
                raise ValueError(
                    "CPU-resident sector states currently require "
                    "sector_uniform proposals."
                )
            host_source = (
                sector_states_host
                if sector_states_host is not None
                else sector_states
            )
            host_packed = np.asarray(host_source, dtype=np.uint64)
            if host_packed.ndim != 1:
                raise ValueError(
                    "CPU-resident sector states must be packed uint64 values."
                )
            if sector_nphi is None or sector_n_particles is None:
                raise ValueError(
                    "CPU-resident sector states require sector_nphi and "
                    "sector_n_particles."
                )
            self.sector_states = None
            self.sector_states_host = host_packed
            sector_states_np = None
        elif isinstance(sector_states, jax.Array):
            packed_sector = sector_states.ndim == 1
            expected_dtype = jnp.uint64 if packed_sector else jnp.int8
            if sector_states.dtype != expected_dtype:
                raise ValueError(
                    "device sector_states must use uint64 packed storage or "
                    "int8 occupation storage."
                )
            self.sector_states = sector_states
            if sector_states_host is None:
                if packed_sector:
                    if sector_nphi is None or sector_n_particles is None:
                        raise ValueError(
                            "packed states without a host occupation table "
                            "require sector_nphi and sector_n_particles."
                        )
                    sector_states_np = None
                else:
                    sector_states_np = np.asarray(jax.device_get(sector_states))
            else:
                sector_states_np = np.asarray(sector_states_host)
                if sector_states_np.shape[0] != sector_states.shape[0]:
                    raise ValueError(
                        "sector_states_host length must match sector_states."
                    )
        else:
            sector_states_np = np.asarray(sector_states, dtype=np.int8)
            self.sector_states = jax.device_put(sector_states_np)
        self.sector_size = int(
            self.sector_states_host.shape[0]
            if self.sector_on_host
            else sector_states.shape[0]
        )
        if self.sector_size < net_cfg.k:
            raise ValueError(
                f"sector_size={self.sector_size} is smaller than k={net_cfg.k}."
            )
        self.n_target_chains = int(n_chains)
        betas = np.asarray(
            (1.0,) if tempering_betas is None else tempering_betas,
            dtype=np.float64,
        )
        if betas.ndim != 1 or betas.size == 0:
            raise ValueError("tempering_betas must be a non-empty rank-1 sequence.")
        if not np.isclose(betas[0], 1.0):
            raise ValueError("the first tempering beta must equal 1.")
        if np.any(betas <= 0.0) or np.any(betas > 1.0):
            raise ValueError("tempering betas must lie in (0, 1].")
        if np.any(np.diff(betas) >= 0.0):
            raise ValueError("tempering betas must be strictly decreasing.")
        if tempering_swap_interval <= 0:
            raise ValueError("tempering_swap_interval must be positive.")
        self.tempering_betas = betas
        self.n_replicas = int(betas.size)
        self.tempering_swap_interval = int(tempering_swap_interval)
        self.n_chains = self.n_target_chains * self.n_replicas
        self.chain_betas = jnp.asarray(
            np.repeat(betas, self.n_target_chains), dtype=jnp.float64
        )
        self.proposal_mode = proposal_mode
        if proposal_mode not in {
            "sector_uniform",
            "momentum_scattering",
            "laqx_hamiltonian",
        }:
            raise ValueError(
                f"Unsupported sampler proposal mode: {proposal_mode}"
            )
        if sector_states_np is None:
            self.nphi = int(sector_nphi)
            self.n_particles = int(sector_n_particles)
        else:
            self.nphi = int(sector_states_np.shape[1])
            particle_counts = np.sum(sector_states_np, axis=1).astype(np.int64)
            if not np.all(particle_counts == particle_counts[0]):
                raise ValueError("sector states must have a fixed particle count.")
            self.n_particles = int(particle_counts[0])
        self.allowed_outgoing_cpu = None
        self.laqx_proposal_cpu = None
        self.positions_host = None
        self.host_rng = np.random.default_rng(seed)
        self.cpu_proposal_backend = (
            "numba_parallel" if NUMBA_AVAILABLE else "numpy"
        )
        if proposal_mode == "momentum_scattering":
            if sector_states_np is None:
                raise ValueError(
                    "momentum_scattering requires a host occupation table."
                )
            if self.n_particles < 2:
                raise ValueError(
                    "momentum scattering requires at least two particles."
                )
            self.allowed_outgoing_cpu = build_allowed_outgoing_pairs_cpu(
                sector_states_np
            )
        elif proposal_mode == "laqx_hamiltonian":
            self.laqx_proposal_cpu = build_laqx_hamiltonian_proposal_table(
                sparse_rows
            )
            if self.laqx_proposal_cpu.cols.shape[0] != self.sector_size:
                raise ValueError(
                    "sparse Hamiltonian row count must match sector size."
                )
        self.key = jax.random.PRNGKey(seed)
        self.positions = None
        self.logprob = None
        self.n_accepted = 0
        self.n_proposed = 0
        self.n_swap_accepted = 0
        self.n_swap_proposed = 0
        self._local_step = 0
        self._swap_phase = 0
        self.reinitialize()

    def set_params(self, params) -> None:
        self.params = params
        self.refresh_logprob()

    def _split(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def refresh_logprob(self) -> None:
        if self.sector_on_host:
            packed = self.sector_states_host[self.positions_host]
            self.logprob = _logprob_from_packed_batches(
                self.params, jnp.asarray(packed, dtype=jnp.uint64), self.net_cfg
            )
        else:
            self.logprob = logprob_from_positions(
                self.params, self.sector_states, self.positions, self.net_cfg
            )

    def reinitialize(self) -> None:
        initial_positions = sample_initial_positions_cpu(
            self.host_rng,
            self.n_chains,
            self.sector_size,
            self.net_cfg.k,
        )
        self.positions = jnp.asarray(initial_positions, dtype=jnp.int64)
        if self.sector_on_host or self.proposal_mode in {
            "momentum_scattering",
            "laqx_hamiltonian",
        }:
            self.positions_host = initial_positions.copy()
        else:
            self.positions_host = None
        self.refresh_logprob()
        self.n_accepted = 0
        self.n_proposed = 0
        self.n_swap_accepted = 0
        self.n_swap_proposed = 0
        self._local_step = 0
        self._swap_phase = 0

    def step(self) -> float:
        if self.sector_on_host:
            row = self.host_rng.integers(
                0, self.net_cfg.k, size=self.n_chains, dtype=np.int64
            )
            candidate = self.host_rng.integers(
                0, self.sector_size, size=self.n_chains, dtype=np.int64
            )
            chain_ids = np.arange(self.n_chains)
            proposal = self.positions_host.copy()
            duplicate_other_row = np.any(
                (self.positions_host == candidate[:, None])
                & (np.arange(self.net_cfg.k)[None, :] != row[:, None]),
                axis=1,
            )
            unchanged = candidate == self.positions_host[chain_ids, row]
            valid = (~duplicate_other_row) & (~unchanged)
            proposal[chain_ids, row] = candidate
            proposal_packed = self.sector_states_host[proposal]
            prop_logprob = np.asarray(
                _logprob_from_packed_batches(
                    self.params,
                    jnp.asarray(proposal_packed, dtype=jnp.uint64),
                    self.net_cfg,
                )
            )
            # JAX device buffers expose read-only NumPy views.  The accepted
            # chains are updated in place below, so materialize writable host
            # storage rather than relying on np.asarray.
            logprob = np.array(self.logprob, dtype=np.float64, copy=True)
            logu = np.log(self.host_rng.random(self.n_chains))
            betas = np.repeat(self.tempering_betas, self.n_target_chains)
            accept = (logu < betas * (prop_logprob - logprob)) & valid
            self.positions_host[accept] = proposal[accept]
            logprob[accept] = prop_logprob[accept]
            self.positions = jnp.asarray(self.positions_host, dtype=jnp.int64)
            self.logprob = jnp.asarray(logprob, dtype=jnp.float64)
            rate = float(np.mean(accept[: self.n_target_chains]))
            self._local_step += 1
            accepted = int(rate * self.n_target_chains)
            self.n_accepted += accepted
            self.n_proposed += self.n_target_chains
            return rate
        self.key, subkey = jax.random.split(self.key, 2)
        if self.proposal_mode in {
            "momentum_scattering",
            "laqx_hamiltonian",
        }:
            if self.proposal_mode == "momentum_scattering":
                proposal, valid, log_hastings = (
                    self._momentum_scattering_proposals_cpu()
                )
            else:
                proposal, valid, log_hastings = (
                    self._laqx_hamiltonian_proposals_cpu()
                )
            self.positions, self.logprob, rate = _accept_host_proposals_fn(
                self.params,
                self.sector_states,
                subkey,
                self.positions,
                self.logprob,
                jnp.asarray(proposal, dtype=jnp.int64),
                jnp.asarray(valid, dtype=jnp.bool_),
                jnp.asarray(log_hastings, dtype=jnp.float64),
                self.chain_betas,
                self.net_cfg,
                self.n_chains,
                self.n_target_chains,
            )
            self.positions_host = np.asarray(self.positions).copy()
        else:
            self.positions, self.logprob, rate = _sector_uniform_step_fn(
                self.params,
                self.sector_states,
                subkey,
                self.positions,
                self.logprob,
                self.chain_betas,
                self.net_cfg,
                self.n_chains,
                self.n_target_chains,
            )
        self._local_step += 1
        if (
            self.n_replicas > 1
            and self._local_step % self.tempering_swap_interval == 0
        ):
            self._attempt_replica_swaps()
        accepted = int(float(rate) * self.n_target_chains)
        self.n_accepted += accepted
        self.n_proposed += self.n_target_chains
        return float(rate)

    def _attempt_replica_swaps(self) -> float:
        positions = np.asarray(self.positions).reshape(
            self.n_replicas,
            self.n_target_chains,
            self.net_cfg.k,
        ).copy()
        logprob = np.asarray(self.logprob).reshape(
            self.n_replicas, self.n_target_chains
        ).copy()
        accepted_total = 0
        proposed_total = 0
        for left in range(self._swap_phase, self.n_replicas - 1, 2):
            right = left + 1
            log_alpha = (
                (self.tempering_betas[left] - self.tempering_betas[right])
                * (logprob[right] - logprob[left])
            )
            accept = np.log(self.host_rng.random(self.n_target_chains)) < np.minimum(
                0.0, log_alpha
            )
            if np.any(accept):
                left_positions = positions[left, accept].copy()
                positions[left, accept] = positions[right, accept]
                positions[right, accept] = left_positions
                left_logprob = logprob[left, accept].copy()
                logprob[left, accept] = logprob[right, accept]
                logprob[right, accept] = left_logprob
            accepted_total += int(np.sum(accept))
            proposed_total += self.n_target_chains
        self._swap_phase = 1 - self._swap_phase
        self.n_swap_accepted += accepted_total
        self.n_swap_proposed += proposed_total
        flat_positions = positions.reshape(self.n_chains, self.net_cfg.k)
        self.positions = jnp.asarray(flat_positions, dtype=jnp.int64)
        self.logprob = jnp.asarray(logprob.reshape(self.n_chains), dtype=jnp.float64)
        if self.proposal_mode in {"momentum_scattering", "laqx_hamiltonian"}:
            self.positions_host = flat_positions.copy()
        return accepted_total / max(1, proposed_total)

    def _momentum_scattering_proposals_cpu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = self.allowed_outgoing_cpu
        positions = self.positions_host
        rows = self.host_rng.integers(
            0,
            self.net_cfg.k,
            size=self.n_chains,
            dtype=np.int64,
        )
        incoming_slots = self.host_rng.integers(
            0,
            table.occupied_pair_slots.shape[0],
            size=self.n_chains,
            dtype=np.int64,
        )
        outgoing_uniform = self.host_rng.random(self.n_chains)
        if NUMBA_AVAILABLE:
            return _momentum_scattering_proposals_numba(
                positions,
                rows,
                incoming_slots,
                outgoing_uniform,
                table.nphi,
                table.bitstrings,
                table.sorted_bitstrings,
                table.sorted_positions,
                table.occupied_orbitals,
                table.occupied_pair_slots,
                table.offsets,
                table.outgoing_masks,
            )
        return self._momentum_scattering_proposals_numpy(
            rows, incoming_slots, outgoing_uniform
        )

    def _laqx_hamiltonian_proposals_cpu(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw Hamiltonian-connected proposals with LaQX operator weights."""
        table = self.laqx_proposal_cpu
        positions = self.positions_host
        chain_ids = np.arange(self.n_chains)
        rows = self.host_rng.integers(
            0,
            self.net_cfg.k,
            size=self.n_chains,
            dtype=np.int64,
        )
        current = positions[chain_ids, rows]
        forward_sum = table.weight_sums[current]
        draws = self.host_rng.random(self.n_chains) * forward_sum
        slots = np.sum(
            table.cumulative_weights[current] < draws[:, None], axis=1
        )
        safe_slots = np.minimum(slots, table.cols.shape[1] - 1)
        candidate = table.cols[current, safe_slots]
        forward_weight = table.weights[current, safe_slots]

        reverse_sum = table.weight_sums[candidate]
        reverse_matches = table.cols[candidate] == current[:, None]
        reverse_weight = np.sum(
            table.weights[candidate] * reverse_matches, axis=1
        )
        columns = np.arange(self.net_cfg.k)[None, :]
        duplicate_other_row = np.any(
            (positions == candidate[:, None]) & (columns != rows[:, None]),
            axis=1,
        )
        valid = (
            (forward_sum > 0.0)
            & (forward_weight > 0.0)
            & (reverse_sum > 0.0)
            & (reverse_weight > 0.0)
            & (~duplicate_other_row)
        )
        proposal = positions.copy()
        proposal[chain_ids[valid], rows[valid]] = candidate[valid]
        log_hastings = np.zeros(self.n_chains, dtype=np.float64)
        log_hastings[valid] = (
            np.log(reverse_weight[valid])
            - np.log(reverse_sum[valid])
            - np.log(forward_weight[valid])
            + np.log(forward_sum[valid])
        )
        return proposal, valid, log_hastings

    def _momentum_scattering_proposals_numpy(
        self,
        rows: np.ndarray,
        incoming_slots: np.ndarray,
        outgoing_uniform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        table = self.allowed_outgoing_cpu
        positions = self.positions_host
        proposal = positions.copy()
        chain_ids = np.arange(self.n_chains)
        current_position = positions[chain_ids, rows]
        occupied = table.occupied_orbitals[current_position]
        pair_slots = table.occupied_pair_slots[incoming_slots]
        incoming_first = occupied[chain_ids, pair_slots[:, 0]]
        incoming_second = occupied[chain_ids, pair_slots[:, 1]]
        total_momentum = (
            incoming_first + incoming_second
        ) % table.nphi

        csr_rows = current_position * table.nphi + total_momentum
        starts = table.offsets[csr_rows]
        stops = table.offsets[csr_rows + 1]
        forward_count = stops - starts
        outgoing_slots = starts + np.floor(
            outgoing_uniform
            * np.maximum(forward_count, 1)
        ).astype(np.int64)
        if table.outgoing_masks.size:
            safe_slots = np.minimum(
                outgoing_slots, table.outgoing_masks.size - 1
            )
            outgoing_masks = table.outgoing_masks[safe_slots]
        else:
            outgoing_masks = np.zeros(self.n_chains, dtype=np.int64)

        incoming_masks = (
            np.left_shift(np.int64(1), incoming_first)
            | np.left_shift(np.int64(1), incoming_second)
        )
        proposed_bits = (
            table.bitstrings[current_position] & ~incoming_masks
        ) | outgoing_masks
        insertion = np.searchsorted(
            table.sorted_bitstrings, proposed_bits
        )
        safe_insertion = np.minimum(
            insertion, table.sorted_bitstrings.shape[0] - 1
        )
        candidate = table.sorted_positions[safe_insertion]
        found = (
            insertion < table.sorted_bitstrings.shape[0]
        ) & (
            table.sorted_bitstrings[safe_insertion] == proposed_bits
        )

        reverse_rows = candidate * table.nphi + total_momentum
        reverse_count = (
            table.offsets[reverse_rows + 1]
            - table.offsets[reverse_rows]
        )
        columns = np.arange(self.net_cfg.k)[None, :]
        duplicate_other_row = np.any(
            (positions == candidate[:, None])
            & (columns != rows[:, None]),
            axis=1,
        )
        valid = (
            (forward_count > 0)
            & found
            & (reverse_count > 0)
            & (~duplicate_other_row)
        )
        proposal[chain_ids[valid], rows[valid]] = candidate[valid]
        log_hastings = np.zeros(self.n_chains, dtype=np.float64)
        log_hastings[valid] = (
            np.log(forward_count[valid])
            - np.log(reverse_count[valid])
        )
        return proposal, valid, log_hastings

    def sample(
        self, n_burn: int, n_between: int, n_samples: int
    ) -> tuple[np.ndarray, float]:
        for _ in range(n_burn):
            self.step()
        positions = []
        accepts = []
        for _ in range(n_samples):
            for _ in range(n_between):
                accepts.append(self.step())
            positions.append(
                np.asarray(self.positions)[: self.n_target_chains]
            )
        positions_out = (
            np.concatenate(positions, axis=0)
            if positions
            else np.empty((0,), dtype=np.int64)
        )
        accept = float(np.mean(accepts)) if accepts else 0.0
        return positions_out, accept

    def acceptance_rate(self) -> float:
        return self.n_accepted / max(1, self.n_proposed)

    def swap_acceptance_rate(self) -> float:
        return self.n_swap_accepted / max(1, self.n_swap_proposed)

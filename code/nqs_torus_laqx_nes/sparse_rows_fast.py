"""Opt-in row-builder with row-local hashing and compact padded outputs."""

from __future__ import annotations

import numpy as np

from . import physics


if physics.NUMBA_AVAILABLE:
    from numba import njit, prange

    @njit(cache=True, inline="always")
    def _popcount64(x):
        x = np.uint64(x)
        x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
        x = (x & np.uint64(0x3333333333333333)) + (
            (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))

    @njit(cache=True, inline="always")
    def _find_sector(conf, sorted_bitstrings, sorted_positions):
        pos = np.searchsorted(sorted_bitstrings, conf)
        if pos < sorted_bitstrings.size and sorted_bitstrings[pos] == conf:
            return sorted_positions[pos]
        return -1

    @njit(cache=True, parallel=True)
    def _build_rows_hash(
        row_indices, sector_bitstrings, sorted_bitstrings, sorted_positions,
        jlist, coulomb_mat, out_capacity, hash_capacity,
    ):
        n_rows = row_indices.size
        diag = np.zeros(n_rows, dtype=np.complex128)
        cols = np.zeros((n_rows, out_capacity), dtype=np.int64)
        vals = np.zeros((n_rows, out_capacity), dtype=np.complex128)
        counts = np.zeros(n_rows, dtype=np.int64)
        overflow = np.zeros(n_rows, dtype=np.uint8)

        for row_flat in prange(n_rows):
            row_index = int(row_indices[row_flat])
            conf = np.uint64(sector_bitstrings[row_index])
            local_cols = np.empty(out_capacity, dtype=np.int64)
            local_vals = np.zeros(out_capacity, dtype=np.complex128)
            hash_slots = np.zeros(hash_capacity, dtype=np.int64)
            local_nnz = 0
            diag_coeff = 0.0j

            for term_idx in range(jlist.shape[0]):
                j1 = int(jlist[term_idx, 0])
                j2 = int(jlist[term_idx, 1])
                j3 = int(jlist[term_idx, 2])
                j4 = int(jlist[term_idx, 3])
                if ((conf >> np.uint64(j3)) & np.uint64(1)) == 0 or (
                    (conf >> np.uint64(j4)) & np.uint64(1)
                ) == 0:
                    continue

                sign = 1
                newconf = conf
                if _popcount64(newconf >> np.uint64(j4 + 1)) & 1:
                    sign = -sign
                newconf &= ~(np.uint64(1) << np.uint64(j4))
                if _popcount64(newconf >> np.uint64(j3 + 1)) & 1:
                    sign = -sign
                newconf &= ~(np.uint64(1) << np.uint64(j3))
                if ((newconf >> np.uint64(j1)) & np.uint64(1)) != 0 or (
                    (newconf >> np.uint64(j2)) & np.uint64(1)
                ) != 0:
                    continue
                if _popcount64(newconf >> np.uint64(j2 + 1)) & 1:
                    sign = -sign
                newconf |= np.uint64(1) << np.uint64(j2)
                if _popcount64(newconf >> np.uint64(j1 + 1)) & 1:
                    sign = -sign
                newconf |= np.uint64(1) << np.uint64(j1)

                col_idx = _find_sector(
                    np.int64(newconf), sorted_bitstrings, sorted_positions
                )
                if col_idx < 0:
                    continue

                vq = (
                    coulomb_mat[j1, j2, j3, j4]
                    - coulomb_mat[j2, j1, j3, j4]
                    - coulomb_mat[j1, j2, j4, j3]
                    + coulomb_mat[j2, j1, j4, j3]
                )
                hij = np.conjugate(vq * sign)
                if np.abs(hij) <= 1e-14:
                    continue
                if col_idx == row_index:
                    diag_coeff += hij
                    continue

                key = np.uint64(col_idx)
                key ^= key >> np.uint64(16)
                key *= np.uint64(0x7FEB352D)
                key ^= key >> np.uint64(15)
                slot = int(key & np.uint64(hash_capacity - 1))
                while True:
                    entry = hash_slots[slot]
                    if entry == 0:
                        if local_nnz >= out_capacity:
                            overflow[row_flat] = 1
                            break
                        local_cols[local_nnz] = col_idx
                        local_vals[local_nnz] = hij
                        hash_slots[slot] = local_nnz + 1
                        local_nnz += 1
                        break
                    index = entry - 1
                    if local_cols[index] == col_idx:
                        local_vals[index] += hij
                        break
                    slot = (slot + 1) & (hash_capacity - 1)
                if overflow[row_flat]:
                    break

            if overflow[row_flat]:
                continue
            write = 0
            for existing in range(local_nnz):
                value = local_vals[existing]
                if np.abs(value) > 1e-14:
                    cols[row_flat, write] = local_cols[existing]
                    vals[row_flat, write] = value
                    write += 1
            diag[row_flat] = diag_coeff
            counts[row_flat] = write
        return diag, cols, vals, counts, overflow

else:  # pragma: no cover
    _build_rows_hash = None


_ORIGINAL_METHOD = physics.TorusCoulombHamiltonian.build_sparse_hamiltonian_rows_subset


def _build_subset_fast(self, tk, basis, row_indices, pad_to_nnz=None):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    flat = row_indices.reshape(-1)
    if flat.size == 0 or not physics.NUMBA_AVAILABLE:
        return _ORIGINAL_METHOD(self, tk, basis, row_indices, pad_to_nnz)

    sector_bitstrings = self._sector_bitstrings(tk, basis)
    sorted_bitstrings, sorted_positions = self._sector_sorted_arrays(tk, basis)
    jlist = self.jlist
    capacity = int(pad_to_nnz) if pad_to_nnz is not None else int(jlist.shape[0])
    capacity = max(1, capacity)
    hash_capacity = 1
    while hash_capacity < 2 * capacity:
        hash_capacity <<= 1
    diag, cols, vals, counts, overflow = _build_rows_hash(
        flat, sector_bitstrings, sorted_bitstrings, sorted_positions,
        jlist, self._require_coulomb_matrix(), capacity, hash_capacity,
    )
    if np.any(overflow):
        return _ORIGINAL_METHOD(self, tk, basis, row_indices, pad_to_nnz)

    max_nnz = max(1, int(np.max(counts)))
    if pad_to_nnz is not None:
        max_nnz = max(max_nnz, int(pad_to_nnz))
    cols = cols[:, :max_nnz].copy()
    vals = vals[:, :max_nnz].copy()
    mask = (
        np.arange(max_nnz, dtype=np.int64)[None, :] < counts[:, None]
    ).astype(np.float64, copy=False)
    shape = row_indices.shape
    return physics.SparseHamiltonianRows(
        diag.reshape(shape), cols.reshape(shape + (max_nnz,)),
        vals.reshape(shape + (max_nnz,)), mask.reshape(shape + (max_nnz,)),
    )


def install():
    physics.TorusCoulombHamiltonian.build_sparse_hamiltonian_rows_subset = _build_subset_fast

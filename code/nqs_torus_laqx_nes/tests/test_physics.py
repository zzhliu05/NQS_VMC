from __future__ import annotations

import math
import unittest

import numpy as np

from nqs_torus_laqx_nes.config import SystemConfig
from nqs_torus_laqx_nes.physics import (
    NUMBA_AVAILABLE,
    TorusCoulombHamiltonian,
    build_basis,
    build_sector_basis,
    generate_coulomb_matrix_python,
    landau_level_form_factor_polynomial,
    sector_basis_bitstrings,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SystemConfig(
            nphi=6,
            p=1,
            q=3,
            coulomb_cut=4,
            disable_numba_coulomb=True,
        )
        self.basis = build_basis(self.cfg)

    def test_basis_count_particles_and_momentum(self):
        self.assertEqual(self.basis.size, math.comb(6, 2))
        self.assertEqual(
            self.basis.number_tk.tolist(), [2, 3, 2, 3, 2, 3]
        )
        for tk in range(self.cfg.nphi):
            for conf in sector_basis_bitstrings(self.basis, tk):
                occupied = [
                    orbital
                    for orbital in range(self.cfg.nphi)
                    if (int(conf) >> orbital) & 1
                ]
                self.assertEqual(len(occupied), self.cfg.ne)
                self.assertEqual(sum(occupied) % self.cfg.nphi, tk)

    def test_compact_sector_basis_matches_full_basis(self):
        for tk in range(self.cfg.nphi):
            compact = build_sector_basis(self.cfg, tk)
            np.testing.assert_array_equal(
                compact.bitstrings,
                sector_basis_bitstrings(self.basis, tk),
            )
            np.testing.assert_array_equal(
                sector_basis_bitstrings(compact, tk), compact.bitstrings
            )
            self.assertEqual(compact.size, int(self.basis.number_tk[tk]))
            self.assertEqual(int(compact.number_tk[tk]), compact.size)

    def test_compact_sector_basis_row_terms_match_full_basis(self):
        tk = 1
        compact = build_sector_basis(self.cfg, tk)
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        full_lookup = {
            int(conf): i
            for i, conf in enumerate(sector_basis_bitstrings(self.basis, tk))
        }
        compact_lookup = {
            int(conf): i
            for i, conf in enumerate(sector_basis_bitstrings(compact, tk))
        }
        for row in range(compact.size):
            full_terms = hamiltonian.row_terms(
                row, tk, self.basis, full_lookup
            )
            hamiltonian.row_cache.clear()
            compact_terms = hamiltonian.row_terms(
                row, tk, compact, compact_lookup
            )
            self.assertEqual(full_terms, compact_terms)
            hamiltonian.row_cache.clear()

    def test_dense_hamiltonian_is_hermitian(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        dense = hamiltonian.build_exact_hamiltonian(0, self.basis)
        np.testing.assert_allclose(dense, dense.conj().T, atol=1e-14)

    def test_landau_level_form_factor_polynomials(self):
        x = 0.37
        self.assertEqual(landau_level_form_factor_polynomial(0, x), 1.0)
        self.assertAlmostEqual(
            landau_level_form_factor_polynomial(1, x), 1.0 - x
        )
        self.assertAlmostEqual(
            landau_level_form_factor_polynomial(2, x),
            1.0 - 2.0 * x + 0.5 * x * x,
        )

    def test_first_landau_level_changes_coulomb_form_factor(self):
        lll = generate_coulomb_matrix_python(4, 1.0, 3, landau_level=0)
        first = generate_coulomb_matrix_python(
            4, 1.0, 3, landau_level=1
        )
        self.assertFalse(np.allclose(first, lll))
        np.testing.assert_allclose(
            first[1, 2, 1, 2], first[1, 2, 1, 2].conjugate()
        )

    def test_sparse_rows_reconstruct_dense_hamiltonian(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        dense = hamiltonian.build_exact_hamiltonian(1, self.basis)
        sparse = hamiltonian.build_sparse_hamiltonian_rows(1, self.basis)
        rebuilt = np.diag(sparse.diag.copy())
        for row in range(rebuilt.shape[0]):
            for col, value, valid in zip(
                sparse.cols[row], sparse.vals[row], sparse.mask[row]
            ):
                if valid:
                    rebuilt[row, int(col)] += value
        np.testing.assert_allclose(rebuilt, dense, atol=1e-14)

    def test_chunked_sparse_rows_match_single_chunk(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        single = hamiltonian.build_sparse_hamiltonian_rows(1, self.basis)
        chunked = hamiltonian.build_sparse_hamiltonian_rows(
            1, self.basis, row_chunk_size=1
        )
        np.testing.assert_allclose(chunked.diag, single.diag, atol=1e-14)
        np.testing.assert_array_equal(chunked.cols, single.cols)
        np.testing.assert_allclose(chunked.vals, single.vals, atol=1e-14)
        np.testing.assert_array_equal(chunked.mask, single.mask)

    def test_sparse_csr_reconstructs_dense_hamiltonian(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        dense = hamiltonian.build_exact_hamiltonian(1, self.basis)
        sparse_csr = hamiltonian.build_sparse_hamiltonian_csr(1, self.basis)
        np.testing.assert_allclose(sparse_csr.toarray(), dense, atol=1e-14)

    def test_sparse_row_subset_matches_full_rows(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        full = hamiltonian.build_sparse_hamiltonian_rows(1, self.basis)
        subset_idx = np.asarray([[0], [2], [1], [2]], dtype=np.int64)
        subset = hamiltonian.build_sparse_hamiltonian_rows_subset(
            1, self.basis, subset_idx
        )
        flat = subset_idx.reshape(-1)
        np.testing.assert_allclose(
            subset.diag.reshape(-1),
            full.diag[flat],
            atol=1e-14,
        )
        for out_row, src_row in enumerate(flat):
            count = int(np.sum(subset.mask.reshape(len(flat), -1)[out_row]))
            full_count = int(np.sum(full.mask[src_row]))
            self.assertEqual(count, full_count)
            np.testing.assert_array_equal(
                subset.cols.reshape(len(flat), -1)[out_row, :count],
                full.cols[src_row, :full_count],
            )
            np.testing.assert_allclose(
                subset.vals.reshape(len(flat), -1)[out_row, :count],
                full.vals[src_row, :full_count],
                atol=1e-14,
            )

        padded = hamiltonian.build_sparse_hamiltonian_rows_subset(
            1, self.basis, subset_idx, pad_to_nnz=full.cols.shape[1] + 3
        )
        self.assertEqual(padded.cols.shape[-1], full.cols.shape[1] + 3)
        np.testing.assert_allclose(padded.diag, subset.diag, atol=1e-14)

    def test_bitstring_rows_match_sector_index_rows_without_lookup(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        sector_bits = sector_basis_bitstrings(self.basis, 1)
        row_idx = np.asarray([0, 2, 1, 2], dtype=np.int64)
        indexed = hamiltonian.build_sparse_hamiltonian_rows_subset(
            1, self.basis, row_idx
        )
        direct = hamiltonian.build_sparse_hamiltonian_bitstring_rows(
            sector_bits[row_idx]
        )
        np.testing.assert_allclose(direct.diag, indexed.diag, atol=1e-14)
        for row in range(row_idx.size):
            indexed_count = int(np.sum(indexed.mask[row]))
            direct_count = int(np.sum(direct.mask[row]))
            self.assertEqual(indexed_count, direct_count)
            indexed_terms = {
                int(sector_bits[int(col)]): value
                for col, value, keep in zip(
                    indexed.cols[row], indexed.vals[row], indexed.mask[row]
                )
                if keep
            }
            direct_terms = {
                int(bits): value
                for bits, value, keep in zip(
                    direct.neighbor_bitstrings[row],
                    direct.vals[row],
                    direct.mask[row],
                )
                if keep
            }
            self.assertEqual(indexed_terms.keys(), direct_terms.keys())
            for bits in indexed_terms:
                self.assertAlmostEqual(
                    indexed_terms[bits], direct_terms[bits], delta=1e-14
                )

    def test_full_sparse_rows_match_subset_over_all_indices(self):
        hamiltonian = TorusCoulombHamiltonian(self.cfg)
        hamiltonian.generate_coulomb_matrix()
        full = hamiltonian.build_sparse_hamiltonian_rows(1, self.basis)
        size = int(self.basis.number_tk[1])
        subset = hamiltonian.build_sparse_hamiltonian_rows_subset(
            1, self.basis, np.arange(size, dtype=np.int64)
        )
        np.testing.assert_allclose(subset.diag, full.diag, atol=1e-14)
        np.testing.assert_array_equal(subset.cols, full.cols)
        np.testing.assert_allclose(subset.vals, full.vals, atol=1e-14)
        np.testing.assert_array_equal(subset.mask, full.mask)

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
    def test_python_and_numba_coulomb_paths_match(self):
        cfg = SystemConfig(
            nphi=4,
            p=1,
            q=2,
            coulomb_cut=3,
            disable_numba_coulomb=False,
        )
        hamiltonian = TorusCoulombHamiltonian(cfg)
        numba_matrix, _, _ = hamiltonian.generate_coulomb_matrix()
        python_matrix = generate_coulomb_matrix_python(
            cfg.nphi,
            cfg.ratio_rect,
            cfg.coulomb_cut,
            cfg.landau_level,
        )
        np.testing.assert_allclose(
            numba_matrix, python_matrix, rtol=1e-12, atol=1e-12
        )

    @unittest.skipUnless(NUMBA_AVAILABLE, "numba is not installed")
    def test_python_and_numba_first_landau_level_paths_match(self):
        cfg = SystemConfig(
            nphi=4,
            p=1,
            q=2,
            landau_level=1,
            coulomb_cut=3,
            disable_numba_coulomb=False,
        )
        hamiltonian = TorusCoulombHamiltonian(cfg)
        numba_matrix, _, _ = hamiltonian.generate_coulomb_matrix()
        python_matrix = generate_coulomb_matrix_python(
            cfg.nphi,
            cfg.ratio_rect,
            cfg.coulomb_cut,
            cfg.landau_level,
        )
        np.testing.assert_allclose(
            numba_matrix, python_matrix, rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()


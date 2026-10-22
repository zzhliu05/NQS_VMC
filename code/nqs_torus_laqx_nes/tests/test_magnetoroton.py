from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from nqs_torus_laqx_nes.config import SystemConfig
from nqs_torus_laqx_nes.magnetoroton import (
    _evaluate_density_action,
    density_transitions,
    estimate_sma_samples,
    summarize_sma,
)
from nqs_torus_laqx_nes.physics import (
    TorusCoulombHamiltonian,
    build_basis,
    sector_basis_bitstrings,
)
from nqs_torus_laqx_nes.spectral import apply_guiding_center_density


class MagnetorotonTests(unittest.TestCase):
    def setUp(self):
        self.nphi = 6
        self.basis = build_basis(SystemConfig(nphi=6, p=1, q=3))
        self.source = sector_basis_bitstrings(self.basis, 1).astype(np.uint64)
        self.target = sector_basis_bitstrings(self.basis, 2).astype(np.uint64)
        rng = np.random.default_rng(18)
        self.psi = rng.normal(size=self.source.size) + 1j * rng.normal(
            size=self.source.size
        )
        self.psi /= np.linalg.norm(self.psi)
        raw = rng.normal(size=(self.target.size, self.target.size)) + 1j * rng.normal(
            size=(self.target.size, self.target.size)
        )
        self.h = (raw + raw.conj().T) / 2
        self.source_lookup = {int(state): value for state, value in zip(self.source, self.psi)}
        self.target_lookup = {int(state): index for index, state in enumerate(self.target)}

    def amplitudes(self, states):
        return np.asarray([self.source_lookup[int(state)] for state in states])

    def rows(self, states):
        indices = [self.target_lookup[int(state)] for state in states]
        diagonal = np.diag(self.h)[indices]
        neighbors = np.zeros((len(indices), max(1, self.target.size - 1)), dtype=np.uint64)
        values = np.zeros(neighbors.shape, dtype=np.complex128)
        mask = np.zeros(neighbors.shape, dtype=bool)
        for row, index in enumerate(indices):
            column = 0
            for other, state in enumerate(self.target):
                if other == index:
                    continue
                neighbors[row, column] = state
                values[row, column] = self.h[index, other]
                mask[row, column] = True
                column += 1
        return SimpleNamespace(
            diag=diagonal,
            neighbor_bitstrings=neighbors,
            vals=values,
            mask=mask,
        )

    def test_density_matches_existing_operator(self):
        for qx in (-2, 0, 1):
            actual = _evaluate_density_action(
                self.target, self.amplitudes, self.nphi, qx, 1, 3
            )
            expected = apply_guiding_center_density(
                self.source, self.target, self.psi, self.nphi, qx, 1
            )
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertEqual(density_transitions(int(self.source[0]), 6, 0, 0),
                         {int(self.source[0]): 2 + 0j})

    def test_exact_path_sum_matches_dense_sma(self):
        qx = 1
        e0 = -0.4
        phi = apply_guiding_center_density(
            self.source, self.target, self.psi, self.nphi, qx, 1
        )
        expected_den = np.vdot(phi, phi)
        expected_num = np.vdot(phi, (self.h - e0 * np.eye(len(phi))) @ phi)
        samples = estimate_sma_samples(
            self.source,
            self.amplitudes,
            self.rows,
            nphi=self.nphi,
            qx_index=qx,
            qy_index=1,
            ground_energy=e0,
            rng=np.random.default_rng(3),
            density_draws=None,
            hamiltonian_draws=None,
            evaluation_chunk_size=2,
        )
        weights = np.abs(self.psi) ** 2
        np.testing.assert_allclose(np.dot(weights, samples.denominator), expected_den)
        np.testing.assert_allclose(np.dot(weights, samples.numerator), expected_num)

    def test_stochastic_path_sampling_and_summary(self):
        qx = 1
        e0 = -0.4
        rng = np.random.default_rng(31)
        sampled = rng.choice(self.source, size=6000, p=np.abs(self.psi) ** 2)
        estimates = estimate_sma_samples(
            sampled,
            self.amplitudes,
            self.rows,
            nphi=self.nphi,
            qx_index=qx,
            qy_index=1,
            ground_energy=e0,
            rng=rng,
            density_draws=1,
            hamiltonian_draws=1,
            evaluation_chunk_size=64,
        )
        phi = apply_guiding_center_density(
            self.source, self.target, self.psi, self.nphi, qx, 1
        )
        expected = np.vdot(phi, (self.h - e0 * np.eye(len(phi))) @ phi) / np.vdot(phi, phi)
        summary = summarize_sma(estimates, block_size=500)
        self.assertAlmostEqual(summary["energy"], expected.real, delta=0.12)
        self.assertGreater(summary["guiding_center_norm"], 0)
        self.assertEqual(summary["blocks"], 12)

    def test_full_coulomb_rows_match_dense_target_sector(self):
        hamiltonian = TorusCoulombHamiltonian(SystemConfig(nphi=6, p=1, q=3))
        hamiltonian.generate_coulomb_matrix()
        rows = hamiltonian.build_sparse_hamiltonian_bitstring_rows(self.target)
        dense = np.diag(np.asarray(rows.diag, dtype=np.complex128))
        for row in range(self.target.size):
            for state, value, valid in zip(
                rows.neighbor_bitstrings[row], rows.vals[row], rows.mask[row]
            ):
                if valid:
                    dense[row, self.target_lookup[int(state)]] += value
        phi = apply_guiding_center_density(
            self.source, self.target, self.psi, self.nphi, 1, 1
        )
        expected_num = np.vdot(phi, dense @ phi)
        estimates = estimate_sma_samples(
            self.source,
            self.amplitudes,
            hamiltonian.build_sparse_hamiltonian_bitstring_rows,
            nphi=self.nphi,
            qx_index=1,
            qy_index=1,
            ground_energy=0.0,
            rng=np.random.default_rng(9),
            density_draws=None,
            hamiltonian_draws=None,
        )
        np.testing.assert_allclose(
            np.dot(np.abs(self.psi) ** 2, estimates.numerator), expected_num
        )


if __name__ == "__main__":
    unittest.main()

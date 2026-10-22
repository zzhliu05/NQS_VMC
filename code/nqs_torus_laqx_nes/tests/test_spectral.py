from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nqs_torus_laqx_nes.config import RunConfig, SystemConfig
from nqs_torus_laqx_nes.physics import build_basis, sector_basis_bitstrings
from nqs_torus_laqx_nes.spectral import (
    SectorStates,
    StateArchive,
    apply_guiding_center_density,
    calculate_spectral_weights,
    load_state_archive,
    save_run_state_archive,
)


class SpectralTests(unittest.TestCase):
    def setUp(self):
        self.system = SystemConfig(nphi=6, p=1, q=3)
        self.basis = build_basis(self.system)

    def random_vector(self, size: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=size) + 1j * rng.normal(size=size)
        return vector / np.linalg.norm(vector)

    def test_zero_momentum_density_is_particle_number(self):
        bitstrings = sector_basis_bitstrings(self.basis, 1)
        vector = self.random_vector(bitstrings.size, 1)
        result = apply_guiding_center_density(
            bitstrings, bitstrings, vector, 6, 0, 0
        )
        np.testing.assert_allclose(result, self.system.ne * vector, atol=1e-14)

    def test_density_adjoint_relation(self):
        source_tk = 1
        qx_index = 2
        qy_index = -2
        target_tk = (source_tk + qy_index) % self.system.nphi
        source_basis = sector_basis_bitstrings(self.basis, source_tk)
        target_basis = sector_basis_bitstrings(self.basis, target_tk)
        source = self.random_vector(source_basis.size, 2)
        target = self.random_vector(target_basis.size, 3)
        forward = apply_guiding_center_density(
            source_basis,
            target_basis,
            source,
            self.system.nphi,
            qx_index,
            qy_index,
        )
        backward = apply_guiding_center_density(
            target_basis,
            source_basis,
            target,
            self.system.nphi,
            -qx_index,
            -qy_index,
        )
        np.testing.assert_allclose(
            np.vdot(target, forward), np.vdot(backward, source), atol=1e-14
        )

    def test_complete_target_basis_saturates_static_sum_rule(self):
        source_tk = 0
        qy_index = 1
        source_bits = sector_basis_bitstrings(self.basis, source_tk)
        target_tk = (source_tk + qy_index) % self.system.nphi
        target_bits = sector_basis_bitstrings(self.basis, target_tk)
        source = self.random_vector(source_bits.size, 4)
        archive = StateArchive(
            nphi=6,
            ne=2,
            ratio_rect=1.0,
            sectors={
                source_tk: SectorStates(
                    source_tk,
                    source_bits,
                    np.asarray([-1.0]),
                    source[:, None],
                ),
                target_tk: SectorStates(
                    target_tk,
                    target_bits,
                    np.arange(target_bits.size, dtype=float),
                    np.eye(target_bits.size, dtype=np.complex128),
                ),
            },
            metadata={},
        )
        _, summaries = calculate_spectral_weights(
            archive, [0], [qy_index], source_tk=source_tk
        )
        self.assertAlmostEqual(summaries[0].captured_fraction, 1.0, places=13)

    def test_archive_round_trip(self):
        bits = sector_basis_bitstrings(self.basis, 0)
        vector = self.random_vector(bits.size, 5)[:, None]
        result = SimpleNamespace(
            tk=0,
            bitstrings=bits,
            nqs_eigenvalues=np.asarray([-1.25]),
            spectral_eigenvalues=np.asarray([-1.25]),
            nqs_wavefunctions=vector,
        )
        config = RunConfig(system=self.system)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "states.npz"
            save_run_state_archive([result], config, str(filename))
            archive = load_state_archive(str(filename))
        self.assertEqual(archive.nphi, 6)
        np.testing.assert_array_equal(archive.sectors[0].bitstrings, bits)
        np.testing.assert_allclose(archive.sectors[0].wavefunctions, vector)


if __name__ == "__main__":
    unittest.main()

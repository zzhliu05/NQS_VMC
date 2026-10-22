from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from nqs_torus_laqx_nes.cli import build_parser, config_from_namespace
from nqs_torus_laqx_nes.config import NetworkConfig, RunConfig, SystemConfig
from nqs_torus_laqx_nes.network import init_params, wavefunction_matrix
from nqs_torus_laqx_nes.physics import (
    SparseHamiltonianRows,
    TorusCoulombHamiltonian,
    bitstrings_to_occupations,
    build_basis,
    sector_basis_bitstrings,
)
from nqs_torus_laqx_nes.sampling import (
    MetropolisSamplerJAX,
    build_laqx_hamiltonian_proposal_table,
)
from nqs_torus_laqx_nes.workflow import run


class NesSmokeTests(unittest.TestCase):
    def test_cli_defaults_to_ace_nes_1d(self):
        args = build_parser().parse_args([])
        config = config_from_namespace(args)
        self.assertEqual(config.network.architecture, "ace_nes_1d")
        self.assertEqual(config.network.nes_hidden, 58)
        self.assertEqual(config.network.nes_layers, 4)
        self.assertEqual(config.network.nes_mlp_hidden, 58)
        self.assertEqual(config.network.nes_mlp_layers, 1)
        self.assertEqual(config.network.nes_cutoff, 3)

    def test_ace_nes_wavefunction_shape_and_dtype(self):
        cfg = NetworkConfig(
            architecture="ace_nes_1d",
            k=2,
            nes_hidden=16,
            nes_layers=2,
            nes_mlp_hidden=16,
            nes_mlp_layers=1,
            nes_cutoff=3,
        )
        params = init_params(123, 6, cfg)
        states = jnp.asarray(
            [[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]],
            dtype=jnp.float32,
        )
        phi = wavefunction_matrix(params, states, cfg)
        self.assertEqual(phi.shape, (2, 2))
        self.assertTrue(jnp.issubdtype(phi.dtype, jnp.complexfloating))
        self.assertTrue(bool(jnp.all(jnp.isfinite(phi.real))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(phi.imag))))

    def test_small_projected_run(self):
        config = RunConfig()
        config = RunConfig(
            system=SystemConfig(
                nphi=6,
                p=1,
                q=3,
                tk=0,
                coulomb_cut=6,
                disable_numba_coulomb=True,
            ),
            network=NetworkConfig(
                architecture="ace_nes_1d",
                k=2,
                nes_hidden=16,
                nes_layers=1,
                nes_mlp_hidden=16,
                nes_mlp_layers=1,
                nes_cutoff=3,
                forward_batch_size=64,
            ),
            training=RunConfig().training.__class__(
                optimize_method="projected",
                n_steps=1,
                min_steps=1,
            ),
            optimizer=RunConfig().optimizer.__class__(
                name="adamw",
                lr=1e-3,
                constant_lr=True,
            ),
            output=RunConfig().output.__class__(build_exact=False),
        )
        result = run(config)
        self.assertEqual(len(result.sector_results), 1)
        sector = result.sector_results[0]
        self.assertEqual(sector.tk, 0)
        self.assertEqual(sector.nqs_eigenvalues.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(sector.nqs_eigenvalues)))

    def test_laqx_hamiltonian_sampler_uses_weighted_neighbors(self):
        rows = SparseHamiltonianRows(
            diag=np.zeros(3, dtype=np.complex128),
            cols=np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
            vals=np.asarray(
                [[0.01, 0.2], [0.01, 0.4], [0.2, 0.4]],
                dtype=np.complex128,
            ),
            mask=np.ones((3, 2), dtype=np.float64),
        )
        table = build_laqx_hamiltonian_proposal_table(rows)
        np.testing.assert_allclose(
            table.weights,
            np.asarray([[0.05, 0.2], [0.05, 0.4], [0.2, 0.4]]),
        )

        states = np.eye(3, dtype=np.float64)
        cfg = NetworkConfig(
            k=1,
            architecture="mlp",
            hidden_sizes=(4,),
            layer_norm=False,
        )
        params = init_params(7, 3, cfg)
        sampler = MetropolisSamplerJAX(
            params,
            states,
            32,
            cfg,
            seed=9,
            proposal_mode="laqx_hamiltonian",
            sparse_rows=rows,
        )
        proposal, valid, log_hastings = (
            sampler._laqx_hamiltonian_proposals_cpu()
        )
        self.assertTrue(np.all(valid))
        self.assertTrue(np.all(np.isfinite(log_hastings)))
        self.assertTrue(
            np.all(proposal[:, 0] != sampler.positions_host[:, 0])
        )


if __name__ == "__main__":
    unittest.main()

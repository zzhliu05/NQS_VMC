from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

from nqs_torus_laqx_nes.config import (
    ContinuousConfig,
    NetworkConfig,
    OptimizerConfig,
    OutputConfig,
    RunConfig,
    SamplerConfig,
    SystemConfig,
    TrainingConfig,
)
from nqs_torus_laqx_nes.workflow import run


class WorkflowTests(unittest.TestCase):
    def make_config(
        self,
        directory: str,
        method: str,
        proposal_mode: str = "sector_uniform",
    ) -> RunConfig:
        return RunConfig(
            system=SystemConfig(
                nphi=6,
                p=1,
                q=3,
                tk=0,
                coulomb_cut=3,
                disable_numba_coulomb=True,
            ),
            network=NetworkConfig(hidden_sizes=(8,)),
            sampler=SamplerConfig(
                proposal_mode=proposal_mode,
                n_chains=8,
                n_burn=1,
                n_between=1,
                n_samples=1,
                prob_n_samples=0,
            ),
            optimizer=OptimizerConfig(
                name="signed_gradient", lr=1e-3
            ),
            training=TrainingConfig(
                optimize_method=method,
                n_steps=1,
                min_steps=1,
            ),
            output=OutputConfig(
                energy_curve_output=str(
                    Path(directory) / f"{method}_energy.png"
                )
            ),
        )

    def test_minimal_projected_and_vmc_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in ("projected", "vmc", "continuous_vmc"):
                with self.subTest(method=method):
                    result = run(self.make_config(directory, method))
                    self.assertEqual(len(result.sector_results), 1)
                    sector = result.sector_results[0]
                    self.assertEqual(len(sector.training.history), 1)
                    self.assertIn("energy", sector.training.history[0])
                    if method == "continuous_vmc":
                        self.assertIn(
                            "proposal_width", sector.training.history[0]
                        )
                        self.assertIn(
                            "peak_device_memory_bytes",
                            sector.training.history[0],
                        )
                    self.assertTrue(
                        Path(result.energy_curve_output).is_file()
                    )

    def test_minimal_vit_runs_all_training_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            for method in ("projected", "vmc", "continuous_vmc"):
                with self.subTest(method=method):
                    base = self.make_config(directory, method)
                    config = RunConfig(
                        system=base.system,
                        network=NetworkConfig(
                            architecture="vit",
                            k=1,
                            vit_patch_size=3,
                            vit_embed_dim=8,
                            vit_num_heads=2,
                            vit_num_layers=1,
                            vit_mlp_ratio=2,
                            vit_fourier_modes=2,
                            forward_batch_size=8,
                        ),
                        sampler=base.sampler,
                        continuous=base.continuous,
                        optimizer=base.optimizer,
                        training=base.training,
                        output=OutputConfig(
                            energy_curve_output=str(
                                Path(directory)
                                / f"vit_{method}_energy.png"
                            )
                        ),
                    )
                    result = run(config)
                    history = result.sector_results[0].training.history
                    self.assertEqual(len(history), 1)
                    self.assertTrue(
                        Path(result.energy_curve_output).is_file()
                    )

    def test_minimal_momentum_scattering_vmc_run(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run(
                self.make_config(
                    directory,
                    "vmc",
                    proposal_mode="momentum_scattering",
                )
            )
            history = result.sector_results[0].training.history
            self.assertEqual(len(history), 1)
            self.assertIn("accept", history[0])

    def test_minimal_row_streaming_vmc_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.make_config(directory, "vmc")
            config = RunConfig(
                system=base.system,
                network=base.network,
                sampler=base.sampler,
                continuous=base.continuous,
                optimizer=base.optimizer,
                training=TrainingConfig(
                    optimize_method="vmc",
                    n_steps=1,
                    min_steps=1,
                    vmc_row_batch_size=2,
                ),
                output=base.output,
            )
            result = run(config)
            history = result.sector_results[0].training.history
            self.assertEqual(len(history), 1)
            self.assertIn("accept", history[0])

    def test_minimal_compact_overlap_continuous_vmc_run(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.make_config(directory, "continuous_vmc")
            config = RunConfig(
                system=base.system,
                network=base.network,
                sampler=base.sampler,
                continuous=ContinuousConfig(
                    basis="compact_overlap",
                    overlap_epsilon=0.5,
                    proposal_width=0.01,
                ),
                optimizer=base.optimizer,
                training=base.training,
                output=base.output,
            )
            result = run(config)
            history = result.sector_results[0].training.history
            self.assertEqual(len(history), 1)
            self.assertGreater(history[0]["active_union_count"], 0)
            self.assertEqual(history[0]["invalid_node_count"], 0)

    def test_probability_diagnostic_uses_configured_mcmc_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(
                directory,
                "vmc",
                proposal_mode="momentum_scattering",
            )
            config = RunConfig(
                system=config.system,
                network=config.network,
                sampler=SamplerConfig(
                    proposal_mode="momentum_scattering",
                    n_chains=8,
                    n_burn=3,
                    n_between=2,
                    n_samples=4,
                    prob_n_burn=3,
                    prob_n_between=2,
                    prob_n_samples=4,
                ),
                optimizer=config.optimizer,
                training=TrainingConfig(
                    optimize_method="vmc",
                    n_steps=100,
                    min_steps=100,
                ),
                output=config.output,
            )
            with patch(
                "nqs_torus_laqx_nes.training.probability_metrics",
                return_value=(0.0,) * 7 + (32,),
            ) as metrics:
                run(config)
            args = metrics.call_args.args
            self.assertEqual(args[4:7], (3, 2, 4))


if __name__ == "__main__":
    unittest.main()


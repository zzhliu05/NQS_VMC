from __future__ import annotations

import unittest

from nqs_torus_laqx_nes.cli import build_parser, config_from_namespace


class CliTests(unittest.TestCase):
    def test_defaults_match_monolithic_script(self):
        args = build_parser().parse_args([])
        config = config_from_namespace(args)
        self.assertEqual(config.system.nphi, 24)
        self.assertEqual(config.system.landau_level, 0)
        self.assertEqual(config.training.optimize_method, "vmc")
        self.assertEqual(config.optimizer.name, "signed_gradient")
        self.assertEqual(config.sampler.proposal_mode, "sector_uniform")
        self.assertEqual(config.sampler.prob_n_burn, 2048)
        self.assertFalse(config.sampler.parallel_tempering)
        self.assertEqual(config.sampler.tempering_replicas, 4)
        self.assertEqual(config.network.architecture, "ace_nes_1d")
        self.assertEqual(config.network.hidden_sizes, (64, 64))
        self.assertEqual(config.network.vit_patch_size, 3)
        self.assertEqual(config.network.vit_embed_dim, 256)
        self.assertEqual(config.network.vit_num_heads, 8)
        self.assertEqual(config.network.vit_num_layers, 4)
        self.assertEqual(config.network.vit_mlp_ratio, 4)
        self.assertEqual(config.network.vit_fourier_modes, 4)
        self.assertEqual(config.continuous.harmonic, 1)
        self.assertEqual(config.continuous.basis, "fourier")
        self.assertEqual(config.continuous.overlap_epsilon, 0.1)
        self.assertEqual(config.continuous.proposal_width, 0.05)
        self.assertFalse(config.optimizer.constant_lr)
        self.assertFalse(config.optimizer.multi_step_lr)
        self.assertFalse(config.optimizer.cyclic_lr)
        self.assertIsNone(config.optimizer.cyclic_lr_min)
        self.assertEqual(config.optimizer.cyclic_lr_period, 1000)
        self.assertEqual(config.optimizer.cyclic_lr_mode, "triangular")
        self.assertFalse(config.optimizer.report_fisher_condition)
        self.assertIsNone(config.training.vmc_row_batch_size)
        self.assertFalse(config.training.vmc_prebuild_sparse_rows)
        self.assertFalse(config.training.vmc_deduplicate_neighbors)
        self.assertFalse(config.training.vmc_sparse_rows_from_bitstrings)
        self.assertFalse(config.training.vmc_timing_analysis)
        self.assertEqual(config.training.vmc_timing_interval, 10)
        self.assertIsNone(config.output.history_output)
        self.assertIsNone(config.output.prob_snapshot_dir)
        self.assertEqual(config.output.prob_snapshot_steps, ())

    def test_all_original_options_are_exposed(self):
        parser = build_parser()
        destinations = {action.dest for action in parser._actions}
        expected = {
            "Nphi",
            "p",
            "q",
            "tk",
            "all_sectors",
            "ratio_rect",
            "landau_level",
            "coulomb_cut",
            "disable_numba_coulomb",
            "k",
            "network_architecture",
            "hidden_sizes",
            "activation",
            "no_layer_norm",
            "logphi_real_clip",
            "init_std",
            "vit_patch_size",
            "vit_embed_dim",
            "vit_num_heads",
            "vit_num_layers",
            "vit_mlp_ratio",
            "vit_fourier_modes",
            "sampler_mode",
            "n_chains",
            "n_burn",
            "n_burn_warmup",
            "n_between",
            "n_samples",
            "parallel_tempering",
            "tempering_replicas",
            "tempering_beta_min",
            "tempering_swap_interval",
            "n_steps",
            "min_steps",
            "converge_tol",
            "converge_window",
            "loss_levels",
            "optimize_method",
            "overlap_shift",
            "lr",
            "sr_diag_shift",
            "cg_tol",
            "cg_maxiter",
            "max_sr_step_norm",
            "projection_rcond",
            "seed",
            "skip_train",
            "vmc_row_batch_size",
            "vmc_prebuild_sparse_rows",
            "vmc_deduplicate_neighbors",
            "vmc_sparse_rows_from_bitstrings",
            "vmc_timing_analysis",
            "vmc_timing_interval",
            "build_exact",
            "plot_output",
            "energy_curve_output",
            "prob_plot_output",
            "history_output",
            "prob_snapshot_dir",
            "prob_snapshot_steps",
            "prob_n_burn",
            "prob_n_between",
            "prob_n_samples",
            "constant_lr",
            "decay_rate",
            "multi_step_lr",
            "cyclic_lr",
            "cyclic_lr_min",
            "cyclic_lr_period",
            "cyclic_lr_mode",
            "optimizer",
            "adam_b1",
            "adam_b2",
            "adam_eps",
            "adam_weight_decay",
            "muon_beta",
            "muon_ns_steps",
            "muon_weight_decay",
            "muon_adaptive",
            "report_fisher_condition",
            "forward_batch_size",
            "perturb_steps",
            "perturb_scale",
            "continuous_harmonic",
            "continuous_basis",
            "continuous_overlap_epsilon",
            "continuous_proposal_width",
            "continuous_target_acceptance",
            "continuous_adapt_rate",
            "continuous_min_width",
            "continuous_max_width",
            "continuous_node_epsilon",
        }
        self.assertTrue(expected.issubset(destinations))

    def test_first_landau_level_configuration(self):
        args = build_parser().parse_args(["--landau_level", "1"])
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.system.landau_level, 1)

    def test_negative_landau_level_is_rejected(self):
        args = build_parser().parse_args(["--landau_level", "-1"])
        config = config_from_namespace(args)
        with self.assertRaisesRegex(ValueError, "landau_level"):
            config.validate()

    def test_continuous_vmc_rejects_k_greater_than_one(self):
        args = build_parser().parse_args(
            ["--optimize_method", "continuous_vmc", "--k", "2"]
        )
        config = config_from_namespace(args)
        with self.assertRaisesRegex(ValueError, "network.k == 1"):
            config.validate()

    def test_parallel_tempering_configuration(self):
        args = build_parser().parse_args(
            [
                "--parallel_tempering",
                "--tempering_replicas",
                "4",
                "--tempering_beta_min",
                "0.125",
                "--tempering_swap_interval",
                "2",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.sampler.tempering_betas[0], 1.0)
        self.assertAlmostEqual(config.sampler.tempering_betas[-1], 0.125)
        self.assertEqual(config.sampler.tempering_swap_interval, 2)

    def test_compact_overlap_basis_configuration(self):
        args = build_parser().parse_args(
            [
                "--continuous_basis",
                "compact_overlap",
                "--continuous_overlap_epsilon",
                "0.2",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.continuous.basis, "compact_overlap")
        self.assertEqual(config.continuous.overlap_epsilon, 0.2)

    def test_vmc_row_batch_size_configuration(self):
        args = build_parser().parse_args(
            [
                "--vmc_row_batch_size",
                "128",
                "--vmc_prebuild_sparse_rows",
                "--vmc_deduplicate_neighbors",
                "--vmc_timing_analysis",
                "--vmc_timing_interval",
                "25",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.training.vmc_row_batch_size, 128)
        self.assertTrue(config.training.vmc_prebuild_sparse_rows)
        self.assertTrue(config.training.vmc_deduplicate_neighbors)
        self.assertTrue(config.training.vmc_timing_analysis)
        self.assertEqual(config.training.vmc_timing_interval, 25)

    def test_bitstring_sparse_rows_configuration(self):
        args = build_parser().parse_args(
            [
                "--vmc_row_batch_size",
                "16",
                "--vmc_chain_batch_size",
                "16",
                "--vmc_sparse_rows_from_bitstrings",
                "--skip_final_ritz",
                "--prob_n_samples",
                "0",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertTrue(config.training.vmc_sparse_rows_from_bitstrings)

    def test_vit_configuration_and_validation(self):
        args = build_parser().parse_args(
            [
                "--network_architecture",
                "vit",
                "--vit_patch_size",
                "2",
                "--vit_embed_dim",
                "16",
                "--vit_num_heads",
                "4",
                "--vit_num_layers",
                "2",
                "--vit_mlp_ratio",
                "3",
                "--vit_fourier_modes",
                "2",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.network.architecture, "vit")
        self.assertEqual(config.network.vit_embed_dim, 16)
        self.assertEqual(config.network.vit_num_heads, 4)

        invalid = config.network.__class__(
            architecture="vit",
            vit_embed_dim=10,
            vit_num_heads=4,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            invalid.validate()

    def test_mps_configuration_and_validation(self):
        args = build_parser().parse_args(
            [
                "--network_architecture",
                "mps",
                "--hidden_sizes",
                "12",
                "10",
                "--mps_bond_dim",
                "6",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.network.architecture, "mps")
        self.assertEqual(config.network.hidden_sizes, (12, 10))
        self.assertEqual(config.network.mps_bond_dim, 6)
        self.assertEqual(config.network.mps_contraction, "sequential")

        balanced_args = build_parser().parse_args(
            ["--network_architecture", "mps", "--mps_contraction", "balanced"]
        )
        balanced_config = config_from_namespace(balanced_args)
        balanced_config.validate()
        self.assertEqual(balanced_config.network.mps_contraction, "balanced")

        fp64_args = build_parser().parse_args(
            ["--network_architecture", "mps", "--mps_contraction", "balanced_fp64"]
        )
        fp64_config = config_from_namespace(fp64_args)
        fp64_config.validate()
        self.assertEqual(fp64_config.network.mps_contraction, "balanced_fp64")

        invalid = config.network.__class__(
            architecture="mps", mps_bond_dim=0
        )
        with self.assertRaisesRegex(ValueError, "mps_bond_dim"):
            invalid.validate()

        invalid_contraction = config.network.__class__(
            architecture="mps", mps_contraction="unsupported"
        )
        with self.assertRaisesRegex(ValueError, "mps_contraction"):
            invalid_contraction.validate()

        backflow_args = build_parser().parse_args(
            [
                "--network_architecture",
                "mps",
                "--mps_tensor_generation",
                "backflow",
                "--mps_backflow_embed_dim",
                "24",
            ]
        )
        backflow_config = config_from_namespace(backflow_args)
        backflow_config.validate()
        self.assertEqual(
            backflow_config.network.mps_tensor_generation, "backflow"
        )
        self.assertEqual(backflow_config.network.mps_backflow_embed_dim, 24)

        invalid_embed = backflow_config.network.__class__(
            architecture="mps",
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=0,
        )
        with self.assertRaisesRegex(ValueError, "mps_backflow_embed_dim"):
            invalid_embed.validate()

        global_args = build_parser().parse_args(
            [
                "--network_architecture",
                "mps",
                "--mps_tensor_generation",
                "global_backflow",
            ]
        )
        global_config = config_from_namespace(global_args)
        global_config.validate()
        self.assertEqual(
            global_config.network.mps_tensor_generation,
            "global_backflow",
        )

        site_head_args = build_parser().parse_args(
            [
                "--network_architecture",
                "mps",
                "--mps_tensor_generation",
                "site_head_backflow",
                "--mps_backflow_embed_dim",
                "128",
            ]
        )
        site_head_config = config_from_namespace(site_head_args)
        site_head_config.validate()
        self.assertEqual(
            site_head_config.network.mps_tensor_generation,
            "site_head_backflow",
        )
        self.assertEqual(
            site_head_config.network.mps_backflow_embed_dim,
            128,
        )

        direct_args = build_parser().parse_args(
            [
                "--network_architecture",
                "mps",
                "--mps_tensor_generation",
                "direct",
            ]
        )
        direct_config = config_from_namespace(direct_args)
        direct_config.validate()
        self.assertEqual(
            direct_config.network.mps_tensor_generation, "direct"
        )

    def test_diagnostic_output_configuration(self):
        args = build_parser().parse_args(
            [
                "--history_output",
                "history.csv",
                "--prob_snapshot_dir",
                "snapshots",
                "--prob_snapshot_steps",
                "100",
                "1000",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertEqual(config.output.history_output, "history.csv")
        self.assertEqual(config.output.prob_snapshot_dir, "snapshots")
        self.assertEqual(config.output.prob_snapshot_steps, (100, 1000))

    def test_multi_step_lr_configuration(self):
        args = build_parser().parse_args(["--multi_step_lr"])
        config = config_from_namespace(args)
        config.validate()
        self.assertTrue(config.optimizer.multi_step_lr)

    def test_constant_lr_and_fisher_condition_configuration(self):
        args = build_parser().parse_args(
            ["--constant_lr", "--report_fisher_condition"]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertTrue(config.optimizer.constant_lr)
        self.assertTrue(config.optimizer.report_fisher_condition)

    def test_cyclic_lr_configuration(self):
        args = build_parser().parse_args(
            [
                "--cyclic_lr",
                "--cyclic_lr_min",
                "1e-3",
                "--cyclic_lr_period",
                "200",
                "--cyclic_lr_mode",
                "triangular2",
            ]
        )
        config = config_from_namespace(args)
        config.validate()
        self.assertTrue(config.optimizer.cyclic_lr)
        self.assertEqual(config.optimizer.cyclic_lr_min, 1e-3)
        self.assertEqual(config.optimizer.cyclic_lr_period, 200)
        self.assertEqual(config.optimizer.cyclic_lr_mode, "triangular2")

    def test_constant_lr_conflicts_with_multi_step_lr(self):
        args = build_parser().parse_args(
            ["--constant_lr", "--multi_step_lr"]
        )
        config = config_from_namespace(args)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config.validate()

    def test_cyclic_lr_conflicts_with_constant_lr(self):
        args = build_parser().parse_args(
            ["--constant_lr", "--cyclic_lr"]
        )
        config = config_from_namespace(args)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config.validate()


if __name__ == "__main__":
    unittest.main()


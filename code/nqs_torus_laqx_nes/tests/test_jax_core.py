from __future__ import annotations

import unittest
from unittest import mock

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from nqs_torus_laqx_nes.config import (
    NetworkConfig,
    OptimizerConfig,
    SystemConfig,
    TrainingConfig,
)
from nqs_torus_laqx_nes.network import (
    batched_wavefunction_matrix_from_positions,
    contract_open_mps,
    contract_periodic_mps,
    contract_periodic_mps_balanced,
    contract_periodic_mps_selected,
    deduplicated_wavefunction_matrix_from_positions,
    determinant_log_wavefunction,
    init_params,
    mps_backflow_embedding_stats,
    mps_backflow_matrix_variation,
    mps_selected_tensor_inputs,
    occupation_states_from_positions,
    patch_occupation_states,
    periodic_fourier_features,
    psi_matrices_from_positions_direct,
    projected_diagonalization,
    wavefunction_compile_cache_size,
    wavefunction_matrix,
)
import nqs_torus_laqx_nes.optimization as optimization_module
from nqs_torus_laqx_nes.optimization import (
    apply_optimizer_update,
    init_optimizer_state,
    learning_rate_at_step,
    learning_rate_from_count,
    local_energies,
    local_energies_sliced,
    profile_vmc_autodiff_sliced,
    profile_vmc_loss_sliced,
    projected_loss,
)
from nqs_torus_laqx_nes.physics import (
    TorusCoulombHamiltonian,
    bitstrings_to_occupations,
    build_basis,
    sector_basis_bitstrings,
)
from nqs_torus_laqx_nes.sampling import (
    NUMBA_AVAILABLE,
    MetropolisSamplerJAX,
    _momentum_scattering_proposals_numba,
    build_allowed_outgoing_pairs_cpu,
    build_momentum_scattering_table,
    sample_initial_positions_cpu,
)


def all_finite(tree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


class JaxCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system_cfg = SystemConfig(
            nphi=6,
            p=1,
            q=3,
            coulomb_cut=6,
            disable_numba_coulomb=True,
        )
        cls.net_cfg = NetworkConfig(hidden_sizes=(8, 8))
        cls.basis = build_basis(cls.system_cfg)
        cls.hamiltonian = TorusCoulombHamiltonian(cls.system_cfg)
        cls.hamiltonian.generate_coulomb_matrix()
        cls.h_exact = cls.hamiltonian.build_exact_hamiltonian(
            0, cls.basis
        )
        cls.sparse = cls.hamiltonian.build_sparse_hamiltonian_rows(
            0, cls.basis
        ).to_jax()
        bitstrings = sector_basis_bitstrings(cls.basis, 0)
        cls.states_np = bitstrings_to_occupations(
            bitstrings, cls.system_cfg.num_sites
        )
        cls.states = jnp.asarray(cls.states_np, dtype=jnp.float64)

    def test_network_is_deterministic_and_complex(self):
        params_a = init_params(1234, 6, self.net_cfg)
        params_b = init_params(1234, 6, self.net_cfg)
        phi_a = wavefunction_matrix(params_a, self.states, self.net_cfg)
        phi_b = wavefunction_matrix(params_b, self.states, self.net_cfg)
        self.assertEqual(phi_a.shape, (2, 1))
        self.assertTrue(jnp.issubdtype(phi_a.dtype, jnp.complexfloating))
        np.testing.assert_array_equal(np.asarray(phi_a), np.asarray(phi_b))

    def test_vit_network_shapes_padding_and_gradients(self):
        for num_sites in (18, 24, 27, 30, 19):
            with self.subTest(num_sites=num_sites):
                cfg = NetworkConfig(
                    architecture="vit",
                    k=2,
                    vit_patch_size=3,
                    vit_embed_dim=16,
                    vit_num_heads=4,
                    vit_num_layers=1,
                    vit_mlp_ratio=2,
                    vit_fourier_modes=2,
                )
                params_a = init_params(123, num_sites, cfg)
                params_b = init_params(123, num_sites, cfg)
                states = jnp.asarray(
                    np.random.default_rng(num_sites).integers(
                        0, 2, size=(3, num_sites)
                    ),
                    dtype=jnp.float32,
                )
                phi_a = wavefunction_matrix(params_a, states, cfg)
                phi_b = wavefunction_matrix(params_b, states, cfg)
                self.assertEqual(phi_a.shape, (3, 2))
                self.assertTrue(
                    jnp.issubdtype(phi_a.dtype, jnp.complexfloating)
                )
                np.testing.assert_array_equal(
                    np.asarray(phi_a), np.asarray(phi_b)
                )
                grad = jax.grad(
                    lambda current_params: jnp.real(
                        jnp.sum(
                            wavefunction_matrix(
                                current_params, states, cfg
                            )
                        )
                    )
                )(params_a)
                self.assertTrue(all_finite(grad))

        patches = patch_occupation_states(
            jnp.asarray([[1, 0, 1, 1, 0]], dtype=jnp.float32), 3
        )
        np.testing.assert_array_equal(
            np.asarray(patches),
            np.asarray([[[1, 0, 1], [1, 0, 0]]]),
        )

    def test_mps_network_shapes_determinism_and_gradients(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=2,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
        )
        params_a = init_params(127, 6, cfg)
        params_b = init_params(127, 6, cfg)
        phi_a = wavefunction_matrix(params_a, self.states, cfg)
        phi_b = wavefunction_matrix(params_b, self.states, cfg)
        self.assertEqual(phi_a.shape, (2, 2))
        self.assertTrue(jnp.issubdtype(phi_a.dtype, jnp.complexfloating))
        np.testing.assert_array_equal(np.asarray(phi_a), np.asarray(phi_b))
        self.assertTrue(bool(jnp.all(jnp.isfinite(phi_a))))

        grad = jax.grad(
            lambda current_params: jnp.real(
                jnp.sum(wavefunction_matrix(current_params, self.states, cfg))
            )
        )(params_a)
        self.assertTrue(all_finite(grad))

        state_batches = self.states[
            jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int64)
        ]
        logpsi = determinant_log_wavefunction(params_a, state_batches, cfg)
        self.assertEqual(logpsi.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logpsi))))

    def test_periodic_mps_uses_trace_contraction(self):
        matrices = jnp.asarray(
            [
                [
                    [[[1.0, 2.0], [101.0, 102.0]]],
                    [[[1.0, 3.0], [2.0, 4.0]]],
                    [[[5.0, 201.0], [6.0, 202.0]]],
                ]
            ],
            dtype=jnp.complex64,
        )
        amplitude = contract_periodic_mps(matrices, 30.0)
        np.testing.assert_allclose(
            np.asarray(amplitude),
            np.asarray([[205018.0 + 0.0j]]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.asarray(amplitude),
            np.asarray(contract_open_mps(matrices, 30.0)),
        )

    def test_balanced_periodic_mps_matches_sequential_value_and_gradient(self):
        key_real, key_imag = jax.random.split(jax.random.PRNGKey(181))
        matrices = (
            jax.random.normal(key_real, (3, 18, 2, 4, 4))
            + 1j * jax.random.normal(key_imag, (3, 18, 2, 4, 4))
        ).astype(jnp.complex64)
        sequential = contract_periodic_mps(matrices, 30.0)
        balanced = contract_periodic_mps_balanced(matrices, 30.0)
        np.testing.assert_allclose(
            np.asarray(balanced),
            np.asarray(sequential),
            rtol=2e-5,
            atol=2e-5,
        )

        def scalar(current, balanced_mode):
            contraction = (
                contract_periodic_mps_balanced
                if balanced_mode
                else contract_periodic_mps
            )
            values = contraction(current, 30.0)
            return jnp.real(jnp.sum(values)) + jnp.imag(jnp.sum(values))

        grad_sequential = jax.grad(lambda x: scalar(x, False))(matrices)
        grad_balanced = jax.grad(lambda x: scalar(x, True))(matrices)
        np.testing.assert_allclose(
            np.asarray(grad_balanced),
            np.asarray(grad_sequential),
            rtol=3e-4,
            atol=3e-4,
        )

    def test_fused4_contraction_is_selectable_and_matches_balanced(self):
        cfg = NetworkConfig(mps_contraction="fused4")
        cfg.validate()
        key_real, key_imag = jax.random.split(jax.random.PRNGKey(189))
        matrices = (
            jax.random.normal(key_real, (2, 7, 1, 3, 3))
            + 1j * jax.random.normal(key_imag, (2, 7, 1, 3, 3))
        ).astype(jnp.complex64)
        fused = contract_periodic_mps_selected(matrices, 30.0, "fused4")
        balanced = contract_periodic_mps_selected(matrices, 30.0, "balanced")
        np.testing.assert_allclose(
            np.asarray(fused), np.asarray(balanced), rtol=2e-5, atol=2e-5
        )

    def test_contiguous_row_reduction_matches_scatter(self):
        net_cfg = NetworkConfig(num_sites=6, k=1)
        bits = np.asarray([1, 2, 0, 3, 4, 5], dtype=np.uint64)
        coeff = jnp.asarray([1, 2, 0, 3, 4, 5], dtype=jnp.complex128)
        row_ids = jnp.asarray([0, 0, 0, 1, 1, 1], dtype=jnp.int64)

        def forward(_params, states, _cfg):
            return jnp.sum(states, axis=1, keepdims=True).astype(jnp.complex128)

        with mock.patch.object(
            optimization_module, "batched_wavefunction_matrix", forward
        ), mock.patch.dict("os.environ", {"NQS_NO_DEDUP": "1"}):
            with mock.patch.dict(
                "os.environ", {"NQS_CONTIGUOUS_ROW_REDUCE": "1"}
            ):
                reduced = optimization_module._accumulate_offdiagonal_from_bitstrings(
                    None, bits, coeff, row_ids, 2, net_cfg
                )
            with mock.patch.dict(
                "os.environ", {"NQS_CONTIGUOUS_ROW_REDUCE": ""}
            ):
                scattered = optimization_module._accumulate_offdiagonal_from_bitstrings(
                    None, bits, coeff, row_ids, 2, net_cfg
                )
        np.testing.assert_allclose(np.asarray(reduced), np.asarray(scattered))

    def test_balanced_fp64_contraction_value_and_gradient(self):
        key_real, key_imag = jax.random.split(jax.random.PRNGKey(187))
        matrices = (
            jax.random.normal(key_real, (2, 7, 1, 3, 3))
            + 1j * jax.random.normal(key_imag, (2, 7, 1, 3, 3))
        ).astype(jnp.complex64)
        fp64 = contract_periodic_mps_selected(
            matrices, 30.0, "balanced_fp64"
        )
        reference = contract_periodic_mps_balanced(
            matrices.astype(jnp.complex128), 30.0
        )
        self.assertEqual(fp64.dtype, jnp.complex128)
        np.testing.assert_allclose(np.asarray(fp64), np.asarray(reference))

        def scalar(current):
            values = contract_periodic_mps_selected(
                current, 30.0, "balanced_fp64"
            )
            return jnp.real(jnp.sum(values))

        gradient = jax.grad(scalar)(matrices)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())

    def test_balanced_fp64_is_used_by_site_backflow(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=4,
            mps_contraction="balanced_fp64",
        )
        params = init_params(189, 6, cfg)
        values = wavefunction_matrix(params, self.states, cfg)
        self.assertEqual(values.dtype, jnp.complex128)
        self.assertTrue(np.isfinite(np.asarray(values)).all())

    def test_balanced_contraction_is_used_by_mps_network(self):
        sequential_cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=4,
            mps_contraction="sequential",
        )
        balanced_cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=4,
            mps_contraction="balanced",
        )
        params = init_params(183, 6, sequential_cfg)
        sequential = wavefunction_matrix(params, self.states, sequential_cfg)
        balanced = wavefunction_matrix(params, self.states, balanced_cfg)
        np.testing.assert_allclose(
            np.asarray(balanced),
            np.asarray(sequential),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_mps_tensor_bank_matches_sitewise_value_and_gradient(self):
        sitewise_cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="sitewise",
        )
        bank_cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="tensor_bank",
        )
        params = init_params(191, 6, sitewise_cfg)
        sitewise = wavefunction_matrix(params, self.states, sitewise_cfg)
        bank = wavefunction_matrix(params, self.states, bank_cfg)
        np.testing.assert_allclose(
            np.asarray(bank), np.asarray(sitewise), rtol=2e-6, atol=2e-6
        )

        def scalar(current_params, cfg):
            values = wavefunction_matrix(current_params, self.states, cfg)
            return jnp.real(jnp.sum(values)) + jnp.imag(jnp.sum(values))

        grad_sitewise = jax.grad(
            lambda current: scalar(current, sitewise_cfg)
        )(params)
        grad_bank = jax.grad(lambda current: scalar(current, bank_cfg))(params)
        leaves_sitewise = jax.tree_util.tree_leaves(grad_sitewise)
        leaves_bank = jax.tree_util.tree_leaves(grad_bank)
        for bank_leaf, sitewise_leaf in zip(leaves_bank, leaves_sitewise):
            np.testing.assert_allclose(
                np.asarray(bank_leaf),
                np.asarray(sitewise_leaf),
                rtol=2e-2,
                atol=5e-3,
            )

    def test_tensor_bank_repeated_positions_use_equivalent_full_sector(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="tensor_bank",
        )
        params = init_params(192, 6, cfg)
        positions = jnp.asarray(
            [[0], [1], [0], [0], [1], [1], [0], [1]], dtype=jnp.int64
        )

        optimized = psi_matrices_from_positions_direct(
            params, self.states, positions, cfg
        )
        flat_states = self.states[positions.reshape((-1,))]
        reference = wavefunction_matrix(params, flat_states, cfg).reshape(
            optimized.shape
        )
        np.testing.assert_allclose(
            np.asarray(optimized), np.asarray(reference), rtol=1e-6, atol=1e-6
        )

        def scalar(current, full_sector):
            if full_sector:
                values = psi_matrices_from_positions_direct(
                    current, self.states, positions, cfg
                )
            else:
                values = wavefunction_matrix(
                    current, flat_states, cfg
                ).reshape(optimized.shape)
            return jnp.real(jnp.sum(values)) + jnp.imag(jnp.sum(values))

        grad_optimized = jax.grad(lambda current: scalar(current, True))(
            params
        )
        grad_reference = jax.grad(lambda current: scalar(current, False))(
            params
        )
        for optimized_leaf, reference_leaf in zip(
            jax.tree_util.tree_leaves(grad_optimized),
            jax.tree_util.tree_leaves(grad_reference),
        ):
            np.testing.assert_allclose(
                np.asarray(optimized_leaf),
                np.asarray(reference_leaf),
                rtol=2e-5,
                atol=2e-5,
            )

    def test_mps_backflow_uses_full_occupation_and_has_finite_gradient(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=7,
        )
        states = jnp.asarray(
            [[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0]],
            dtype=jnp.float32,
        )
        inputs = mps_selected_tensor_inputs(states, "backflow")
        self.assertEqual(inputs.shape, (2, 6, 14))
        np.testing.assert_array_equal(
            np.asarray(inputs[0, 0, :6]),
            np.asarray(inputs[1, 0, :6]),
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(inputs[0, 0, 8:]),
                np.asarray(inputs[1, 0, 8:]),
            )
        )
        np.testing.assert_array_equal(
            np.asarray(inputs[0, 0, 6:8]), np.asarray([0.0, 1.0])
        )

        params = init_params(193, 6, cfg)
        tensor_mlp = params["mps"]["tensor_mlp"]
        self.assertEqual(tensor_mlp[0]["w"].shape, (14, 7))
        self.assertEqual(tensor_mlp[1]["w"].shape, (7, 8))
        embedding_w = np.asarray(tensor_mlp[0]["w"])
        self.assertTrue(np.any(embedding_w[:8] != 0))
        np.testing.assert_array_equal(
            embedding_w[8:], np.zeros_like(embedding_w[8:])
        )
        first_linear = np.asarray(inputs) @ embedding_w
        np.testing.assert_allclose(
            first_linear[0, 0], first_linear[1, 0], rtol=0, atol=0
        )
        values = wavefunction_matrix(params, states, cfg)
        self.assertEqual(values.shape, (2, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
        grad = jax.grad(
            lambda current: jnp.real(
                jnp.sum(wavefunction_matrix(current, states, cfg))
            )
        )(params)
        self.assertTrue(all_finite(grad))
        occupation_grad = np.asarray(
            grad["mps"]["tensor_mlp"][0]["w"]
        )[8:]
        self.assertTrue(np.any(np.abs(occupation_grad) > 0))

    def test_mps_backflow_embedding_stats_separate_weight_groups(self):
        num_sites = 6
        cfg = NetworkConfig(
            architecture="mps",
            hidden_sizes=(8,),
            k=1,
            mps_bond_dim=2,
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=5,
        )
        params = init_params(7, num_sites, cfg)
        initial = mps_backflow_embedding_stats(params, num_sites)
        self.assertEqual(
            initial["mps_embedding_backflow_power_fraction"], 0.0
        )
        self.assertEqual(initial["mps_embedding_backflow_norm"], 0.0)

        weights = params["mps"]["tensor_mlp"][0]["w"]
        params["mps"]["tensor_mlp"][0]["w"] = weights.at[
            num_sites + 2 :, :
        ].set(1.0)
        trained = mps_backflow_embedding_stats(params, num_sites)
        self.assertGreater(
            trained["mps_embedding_backflow_power_fraction"], 0.0
        )
        self.assertGreater(
            trained["mps_embedding_backflow_to_local_rms"], 0.0
        )

    def test_mps_backflow_matrix_variation_is_zero_at_initialization(self):
        cfg = NetworkConfig(
            architecture="mps",
            hidden_sizes=(8,),
            k=1,
            mps_bond_dim=2,
            mps_tensor_generation="backflow",
            mps_backflow_embed_dim=5,
        )
        params = init_params(11, 6, cfg)
        states = np.asarray(
            [
                [1, 0, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 0],
                [0, 1, 1, 0, 0, 1],
                [0, 1, 0, 1, 0, 1],
            ],
            dtype=np.float32,
        )
        variation = mps_backflow_matrix_variation(params, states, cfg)
        np.testing.assert_allclose(
            variation["relative_rms"], 0.0, rtol=0.0, atol=1e-7
        )

    def test_mps_backflow_natural_gradient_uses_matching_kfac(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(4,),
            mps_bond_dim=2,
            mps_backflow_embed_dim=3,
            init_std=0.01,
            mps_tensor_generation="backflow",
        )
        params = init_params(197, 6, cfg)
        train_cfg = TrainingConfig(optimize_method="projected")

        def loss_fn(current_params):
            return projected_loss(
                current_params,
                self.states.astype(jnp.float32),
                jnp.asarray(self.h_exact, dtype=jnp.complex64),
                1,
                cfg,
                train_cfg,
            )[0]

        grad_tree = jax.grad(loss_fn)(params)
        opt_cfg = OptimizerConfig(
            name="natural_gradient", lr=5e-3, constant_lr=True
        )
        state = init_optimizer_state(params, opt_cfg, 1)
        updated, _, stats = apply_optimizer_update(
            params,
            self.states,
            grad_tree,
            1,
            cfg,
            opt_cfg,
            state,
            1,
        )
        self.assertTrue(all_finite(updated))
        self.assertEqual(stats["optimizer_impl"], "kfac")
        self.assertEqual(
            stats["kfac_variant"],
            "mps_backflow_tensor_mlp_diag_output",
        )

    def test_mps_backflow_padded_neighbor_forward_matches_full_sector(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(5,),
            mps_bond_dim=2,
            mps_backflow_embed_dim=4,
            init_std=0.01,
            mps_tensor_generation="backflow",
            forward_batch_size=128,
        )
        params = init_params(223, 6, cfg)
        positions = jnp.asarray([[0], [1], [0]], dtype=jnp.int64)
        args = (
            params,
            self.states,
            self.sparse.diag[np.asarray(positions)],
            self.sparse.cols[np.asarray(positions)],
            self.sparse.vals[np.asarray(positions)],
            self.sparse.mask[np.asarray(positions)],
            positions,
            cfg,
        )
        full_sector = local_energies_sliced(
            *args, deduplicate_neighbors=False
        )
        padded_neighbors = local_energies_sliced(
            *args,
            deduplicate_neighbors=True,
            neighbor_padding_size=int(np.prod(args[3].shape)),
        )
        bucketed_neighbors = local_energies_sliced(
            *args,
            deduplicate_neighbors=True,
            neighbor_padding_size=int(np.prod(args[3].shape)),
            neighbor_padding_bucket_size=4,
        )
        np.testing.assert_allclose(
            np.asarray(padded_neighbors),
            np.asarray(full_sector),
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.asarray(bucketed_neighbors),
            np.asarray(full_sector),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_mps_backflow_repeated_positions_use_equivalent_deduplicated_forward(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(5,),
            mps_bond_dim=2,
            mps_backflow_embed_dim=4,
            init_std=0.01,
            mps_tensor_generation="backflow",
        )
        params = init_params(227, 6, cfg)
        positions = jnp.asarray([[0], [1], [0], [2], [1], [0]])
        flat_positions = positions.reshape((-1,))
        reference = wavefunction_matrix(
            params, self.states[flat_positions], cfg
        )
        optimized = deduplicated_wavefunction_matrix_from_positions(
            params, self.states, positions, cfg
        )
        np.testing.assert_allclose(
            np.asarray(optimized), np.asarray(reference), rtol=2e-6, atol=2e-6
        )

        def summed_real(current_params, optimized_path):
            if optimized_path:
                values = deduplicated_wavefunction_matrix_from_positions(
                    current_params, self.states, positions, cfg
                )
            else:
                values = wavefunction_matrix(
                    current_params, self.states[flat_positions], cfg
                )
            return jnp.real(jnp.sum(values))

        params_f32 = jax.tree_util.tree_map(
            lambda value: value.astype(jnp.float32)
            if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
            else value,
            params,
        )
        optimized_grad = jax.grad(summed_real)(params_f32, True)
        optimized_leaves = jax.tree_util.tree_leaves(optimized_grad)
        self.assertTrue(all_finite(optimized_grad))
        self.assertGreater(
            sum(float(jnp.linalg.norm(leaf)) for leaf in optimized_leaves),
            0.0,
        )

    def test_position_batched_forward_matches_full_gather_and_gradient(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(5,),
            mps_bond_dim=2,
            mps_backflow_embed_dim=4,
            init_std=0.01,
            mps_tensor_generation="backflow",
            forward_batch_size=2,
        )
        params = init_params(229, 6, cfg)
        states = self.states.astype(jnp.int8)
        positions = jnp.asarray([0, 1, 0, 1, 1], dtype=jnp.int64)
        reference = wavefunction_matrix(
            params, states[positions].astype(jnp.float32), cfg
        )
        chunked = batched_wavefunction_matrix_from_positions(
            params, states, positions, cfg
        )
        np.testing.assert_allclose(
            np.asarray(chunked), np.asarray(reference), rtol=2e-6, atol=2e-6
        )

        def summed_real(current_params, chunked_path):
            if chunked_path:
                values = batched_wavefunction_matrix_from_positions(
                    current_params, states, positions, cfg
                )
            else:
                values = wavefunction_matrix(
                    current_params,
                    states[positions].astype(jnp.float32),
                    cfg,
                )
            return jnp.real(jnp.sum(values))

        params_f32 = jax.tree_util.tree_map(
            lambda value: value.astype(jnp.float32)
            if hasattr(value, "dtype")
            and jnp.issubdtype(value.dtype, jnp.floating)
            else value,
            params,
        )
        chunked_grad = jax.grad(summed_real)(params_f32, True)
        reference_grad = jax.grad(summed_real)(params_f32, False)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(chunked_grad),
            jax.tree_util.tree_leaves(reference_grad),
        ):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5
            )

    def test_packed_sector_gather_matches_int8_occupations(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(5,),
            mps_bond_dim=2,
            mps_backflow_embed_dim=4,
            init_std=0.01,
            mps_tensor_generation="backflow",
            forward_batch_size=2,
            num_sites=6,
        )
        bitstrings = sector_basis_bitstrings(self.basis, 0).astype(np.uint64)
        packed = jnp.asarray(bitstrings, dtype=jnp.uint64)
        occupations = jnp.asarray(self.states_np, dtype=jnp.int8)
        positions = jnp.asarray([0, 1, 0, 1, 1], dtype=jnp.int64)
        unpacked = occupation_states_from_positions(
            packed, positions, cfg
        )
        np.testing.assert_array_equal(
            np.asarray(unpacked),
            np.asarray(occupations[positions], dtype=np.float32),
        )

        params = init_params(233, 6, cfg)
        packed_values = batched_wavefunction_matrix_from_positions(
            params, packed, positions, cfg
        )
        reference = batched_wavefunction_matrix_from_positions(
            params, occupations, positions, cfg
        )
        np.testing.assert_allclose(
            np.asarray(packed_values),
            np.asarray(reference),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_sitewise_mps_direct_positions_unpack_packed_sector(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(5,),
            mps_bond_dim=2,
            init_std=0.01,
            mps_tensor_generation="sitewise",
            num_sites=6,
        )
        packed = jnp.asarray(
            sector_basis_bitstrings(self.basis, 0), dtype=jnp.uint64
        )
        occupations = jnp.asarray(self.states_np, dtype=jnp.int8)
        positions = jnp.asarray(
            [[0], [1], [0], [1]], dtype=jnp.int64
        )
        params = init_params(235, 6, cfg)

        packed_values = psi_matrices_from_positions_direct(
            params, packed, positions, cfg
        )
        reference = psi_matrices_from_positions_direct(
            params, occupations, positions, cfg
        )
        np.testing.assert_allclose(
            np.asarray(packed_values),
            np.asarray(reference),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_mps_global_backflow_outputs_all_site_tensors_once(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8,),
            mps_bond_dim=3,
            init_std=0.01,
            mps_tensor_generation="global_backflow",
        )
        params = init_params(199, 6, cfg)
        tensor_mlp = params["mps"]["tensor_mlp"]
        self.assertEqual(tensor_mlp[0]["w"].shape[0], 6)
        self.assertEqual(tensor_mlp[-1]["w"].shape[1], 2 * 6 * 3 * 3)
        values = wavefunction_matrix(params, self.states, cfg)
        self.assertEqual(values.shape, (2, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
        grad = jax.grad(
            lambda current: jnp.real(
                jnp.sum(wavefunction_matrix(current, self.states, cfg))
            )
        )(params)
        self.assertTrue(all_finite(grad))

    def test_mps_site_head_backflow_parameter_budget_and_gradient(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(894, 894),
            mps_bond_dim=10,
            mps_backflow_embed_dim=128,
            init_std=0.01,
            mps_tensor_generation="site_head_backflow",
        )
        cfg.validate()
        params = init_params(203, 27, cfg)
        parameter_count = sum(
            int(value.size) for value in jax.tree_util.tree_leaves(params)
        )
        self.assertEqual(parameter_count, 999_690)
        site_head = params["mps"]
        self.assertEqual(site_head["gamma"].shape, (27, 2, 128))
        self.assertEqual(site_head["beta"].shape, (27, 2, 128))

        states = jnp.zeros((2, 27), dtype=jnp.float32)
        states = states.at[0, :9].set(1.0)
        states = states.at[1, 9:18].set(1.0)
        values = wavefunction_matrix(params, states, cfg)
        self.assertEqual(values.shape, (2, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(values))))
        grad = jax.grad(
            lambda current: jnp.real(
                jnp.sum(wavefunction_matrix(current, states, cfg))
            )
        )(params)
        self.assertTrue(all_finite(grad))
        self.assertGreater(
            sum(
                float(jnp.linalg.norm(value.astype(jnp.float32)))
                for value in jax.tree_util.tree_leaves(grad)
            ),
            0.0,
        )

    def test_mps_site_head_backflow_uses_blockwise_kfac(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(8, 8),
            mps_bond_dim=2,
            mps_backflow_embed_dim=4,
            init_std=0.01,
            mps_tensor_generation="site_head_backflow",
        )
        params = init_params(205, 6, cfg)
        train_cfg = TrainingConfig(optimize_method="projected")

        def loss_fn(current_params):
            return projected_loss(
                current_params,
                self.states.astype(jnp.float32),
                jnp.asarray(self.h_exact, dtype=jnp.complex64),
                1,
                cfg,
                train_cfg,
            )[0]

        grad_tree = jax.grad(loss_fn)(params)
        opt_cfg = OptimizerConfig(
            name="natural_gradient", lr=5e-3, constant_lr=True
        )
        state = init_optimizer_state(params, opt_cfg, 1)
        updated, _, stats = apply_optimizer_update(
            params,
            self.states,
            grad_tree,
            1,
            cfg,
            opt_cfg,
            state,
            1,
        )
        self.assertTrue(all_finite(updated))
        self.assertEqual(stats["optimizer_impl"], "kfac")
        self.assertEqual(
            stats["kfac_variant"],
            "mps_site_head_blockwise_diag_film",
        )

    def test_direct_mps_parameters_forward_and_matrix_free_sr(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(4,),
            mps_bond_dim=2,
            init_std=0.01,
            mps_tensor_generation="direct",
        )
        params = init_params(207, 6, cfg)
        tensors = params["mps"]["tensors"]
        self.assertEqual(tensors.shape, (6, 2, 1, 2, 2, 2))
        self.assertEqual(tensors.size, 96)
        values = wavefunction_matrix(params, self.states, cfg)
        self.assertEqual(values.shape, (2, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(values))))

        train_cfg = TrainingConfig(optimize_method="projected")

        def loss_fn(current_params):
            return projected_loss(
                current_params,
                self.states.astype(jnp.float32),
                jnp.asarray(self.h_exact, dtype=jnp.complex64),
                1,
                cfg,
                train_cfg,
            )[0]

        grad_tree = jax.grad(loss_fn)(params)
        opt_cfg = OptimizerConfig(
            name="natural_gradient",
            lr=5e-3,
            constant_lr=True,
            cg_maxiter=20,
        )
        state = init_optimizer_state(params, opt_cfg, 1)
        updated, _, stats = apply_optimizer_update(
            params,
            self.states,
            grad_tree,
            1,
            cfg,
            opt_cfg,
            state,
            1,
        )
        self.assertTrue(all_finite(updated))
        self.assertEqual(stats["optimizer_impl"], "matrix_free_sr")

    def test_mps_global_backflow_natural_gradient_uses_matching_kfac(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(4,),
            mps_bond_dim=2,
            init_std=0.01,
            mps_tensor_generation="global_backflow",
        )
        params = init_params(211, 6, cfg)
        train_cfg = TrainingConfig(optimize_method="projected")

        def loss_fn(current_params):
            return projected_loss(
                current_params,
                self.states.astype(jnp.float32),
                jnp.asarray(self.h_exact, dtype=jnp.complex64),
                1,
                cfg,
                train_cfg,
            )[0]

        grad_tree = jax.grad(loss_fn)(params)
        opt_cfg = OptimizerConfig(
            name="natural_gradient", lr=5e-3, constant_lr=True
        )
        state = init_optimizer_state(params, opt_cfg, 1)
        updated, _, stats = apply_optimizer_update(
            params,
            self.states,
            grad_tree,
            1,
            cfg,
            opt_cfg,
            state,
            1,
        )
        self.assertTrue(all_finite(updated))
        self.assertEqual(stats["optimizer_impl"], "kfac")
        self.assertEqual(
            stats["kfac_variant"],
            "mps_global_backflow_mlp_diag_output",
        )

    def test_mps_kfac_output_score_matches_autodiff(self):
        matrices = jnp.asarray(
            [
                [
                    [[1.1 + 0.1j, 0.2 - 0.1j], [0.3, 0.9 + 0.2j]],
                    [[0.8, 0.1 + 0.2j], [-0.2j, 1.2]],
                    [[1.0 - 0.1j, 0.4 + 0.1j], [0.2, 0.7]],
                ]
            ],
            dtype=jnp.complex64,
        )
        matrix_size = 4
        outputs = jnp.concatenate(
            (
                jnp.real(matrices).reshape((1, 3, matrix_size)),
                jnp.imag(matrices).reshape((1, 3, matrix_size)),
            ),
            axis=-1,
        )

        def logpsi(current_outputs):
            current = (
                current_outputs[..., :matrix_size]
                + 1j * current_outputs[..., matrix_size:]
            ).reshape((1, 3, 2, 2))
            return jnp.log(
                contract_periodic_mps(current[:, :, None], 30.0)[0, 0]
            )

        real_score, imag_score = optimization_module._mps_logpsi_output_score(
            matrices, 30.0
        )
        grad_real = jax.grad(lambda value: jnp.real(logpsi(value)))(outputs)
        grad_imag = jax.grad(lambda value: jnp.imag(logpsi(value)))(outputs)
        np.testing.assert_allclose(
            np.asarray(real_score.reshape((1, 3, -1))),
            np.asarray(grad_real),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            np.asarray(imag_score.reshape((1, 3, -1))),
            np.asarray(grad_imag),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_mps_natural_gradient_uses_kfac(self):
        cfg = NetworkConfig(
            architecture="mps",
            k=1,
            hidden_sizes=(4,),
            mps_bond_dim=2,
            init_std=0.01,
        )
        params = init_params(131, 6, cfg)
        train_cfg = TrainingConfig(optimize_method="projected")

        def loss_fn(current_params):
            return projected_loss(
                current_params,
                self.states.astype(jnp.float32),
                jnp.asarray(self.h_exact, dtype=jnp.complex64),
                1,
                cfg,
                train_cfg,
            )[0]

        grad_tree = jax.grad(loss_fn)(params)
        opt_cfg = OptimizerConfig(
            name="natural_gradient",
            lr=5e-3,
            constant_lr=True,
        )
        state = init_optimizer_state(params, opt_cfg, 1)
        updated, _, stats = apply_optimizer_update(
            params,
            self.states,
            grad_tree,
            1,
            cfg,
            opt_cfg,
            state,
            1,
        )
        self.assertTrue(all_finite(updated))
        self.assertEqual(stats["optimizer_impl"], "kfac")
        self.assertEqual(
            stats["kfac_variant"], "mps_shared_tensor_mlp_diag_output"
        )

    def test_periodic_fourier_features_are_periodic(self):
        features = periodic_fourier_features(
            jnp.asarray([0.25, 6.25]), period=6, num_modes=4
        )
        np.testing.assert_allclose(
            np.asarray(features[0]),
            np.asarray(features[1]),
            rtol=0,
            atol=1e-6,
        )

    def test_vit_determinant_and_local_energy_paths(self):
        cfg = NetworkConfig(
            architecture="vit",
            k=2,
            vit_patch_size=3,
            vit_embed_dim=16,
            vit_num_heads=4,
            vit_num_layers=1,
            vit_mlp_ratio=2,
            vit_fourier_modes=2,
        )
        params = init_params(321, 6, cfg)
        state_batches = self.states[
            jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int64)
        ]
        logpsi = determinant_log_wavefunction(
            params, state_batches, cfg
        )
        self.assertEqual(logpsi.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(logpsi))))

        positions = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int64)
        direct = local_energies(
            params,
            self.states,
            self.sparse,
            positions,
            cfg,
        )
        positions_np = np.asarray(positions)
        sliced = local_energies_sliced(
            params,
            self.states,
            self.sparse.diag[positions_np],
            self.sparse.cols[positions_np],
            self.sparse.vals[positions_np],
            self.sparse.mask[positions_np],
            positions,
            cfg,
        )
        np.testing.assert_allclose(
            np.asarray(direct),
            np.asarray(sliced),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_sampler_keeps_unique_determinant_rows(self):
        net_cfg = NetworkConfig(k=2, hidden_sizes=(8,))
        bitstrings = sector_basis_bitstrings(self.basis, 1)
        states = bitstrings_to_occupations(bitstrings, 6)
        params = init_params(7, 6, net_cfg)
        sampler = MetropolisSamplerJAX(
            params, states, 16, net_cfg, seed=9
        )
        positions, _ = sampler.sample(2, 1, 3)
        self.assertTrue(np.all(positions >= 0))
        self.assertTrue(np.all(positions < states.shape[0]))
        self.assertTrue(
            np.all(
                np.apply_along_axis(
                    lambda row: len(set(row.tolist())) == net_cfg.k,
                    1,
                    positions,
                )
            )
        )

    def test_sampler_reuses_device_sector_table(self):
        net_cfg = NetworkConfig(k=2, hidden_sizes=(8,))
        bitstrings = sector_basis_bitstrings(self.basis, 1)
        states_host = bitstrings_to_occupations(bitstrings, 6).astype(
            np.int8
        )
        states_device = jax.device_put(states_host)
        params = init_params(17, 6, net_cfg)
        sampler = MetropolisSamplerJAX(
            params,
            states_device,
            16,
            net_cfg,
            seed=19,
            sector_states_host=states_host,
        )
        self.assertIs(sampler.sector_states, states_device)
        self.assertEqual(sampler.sector_states.dtype, jnp.int8)
        positions, _ = sampler.sample(1, 1, 1)
        self.assertEqual(positions.shape, (16, net_cfg.k))

    def test_sampler_reuses_packed_device_sector_table(self):
        net_cfg = NetworkConfig(
            k=2, hidden_sizes=(8,), num_sites=6
        )
        bitstrings = sector_basis_bitstrings(self.basis, 1)
        states_host = bitstrings_to_occupations(bitstrings, 6).astype(
            np.int8
        )
        packed_device = jax.device_put(bitstrings.astype(np.uint64))
        params = init_params(23, 6, net_cfg)
        sampler = MetropolisSamplerJAX(
            params,
            packed_device,
            16,
            net_cfg,
            seed=29,
            sector_states_host=states_host,
        )
        self.assertIs(sampler.sector_states, packed_device)
        self.assertEqual(sampler.sector_states.dtype, jnp.uint64)
        positions, _ = sampler.sample(1, 1, 1)
        self.assertEqual(positions.shape, (16, net_cfg.k))

    def test_initial_positions_scale_with_chains_and_k(self):
        positions = sample_initial_positions_cpu(
            np.random.default_rng(91),
            n_chains=4096,
            sector_size=173593,
            k=1,
        )
        self.assertEqual(positions.shape, (4096, 1))
        self.assertEqual(positions.nbytes, 4096 * np.dtype(np.int64).itemsize)
        self.assertTrue(np.all(positions >= 0))
        self.assertTrue(np.all(positions < 173593))

    def test_initial_positions_are_distinct_and_reproducible(self):
        first = sample_initial_positions_cpu(
            np.random.default_rng(93), 128, 1000, 4
        )
        second = sample_initial_positions_cpu(
            np.random.default_rng(93), 128, 1000, 4
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(
            np.all([len(set(row.tolist())) == 4 for row in first])
        )

    def test_parallel_tempering_returns_only_target_replica(self):
        net_cfg = NetworkConfig(k=1, hidden_sizes=(8,))
        bitstrings = sector_basis_bitstrings(self.basis, 1)
        states = bitstrings_to_occupations(bitstrings, 6)
        params = init_params(71, 6, net_cfg)
        sampler = MetropolisSamplerJAX(
            params,
            states,
            12,
            net_cfg,
            seed=73,
            tempering_betas=(1.0, 0.5, 0.25),
            tempering_swap_interval=1,
        )
        self.assertEqual(sampler.n_target_chains, 12)
        self.assertEqual(sampler.n_chains, 36)
        positions, accept = sampler.sample(2, 1, 3)
        self.assertEqual(positions.shape, (36, 1))
        self.assertTrue(np.all(positions >= 0))
        self.assertTrue(np.all(positions < states.shape[0]))
        self.assertGreater(sampler.n_swap_proposed, 0)
        self.assertGreaterEqual(sampler.swap_acceptance_rate(), 0.0)
        self.assertLessEqual(sampler.swap_acceptance_rate(), 1.0)
        self.assertGreaterEqual(accept, 0.0)
        self.assertLessEqual(accept, 1.0)

    def test_momentum_scattering_table_groups_pair_momenta(self):
        table = build_momentum_scattering_table(6)
        self.assertEqual(table.pairs.shape[0], 6)
        self.assertEqual(int(np.sum(table.counts)), 15)
        for total in range(6):
            pairs = table.pairs[total, : table.counts[total]]
            self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
            self.assertTrue(
                np.all((pairs[:, 0] + pairs[:, 1]) % 6 == total)
            )

    def test_allowed_outgoing_pairs_are_precomputed_on_cpu(self):
        system_cfg = SystemConfig(nphi=8, p=1, q=2)
        basis = build_basis(system_cfg)
        states = bitstrings_to_occupations(
            sector_basis_bitstrings(basis, 0), 8
        )
        table = build_allowed_outgoing_pairs_cpu(states)
        self.assertIsInstance(table.offsets, np.ndarray)
        self.assertIsInstance(table.outgoing_masks, np.ndarray)
        self.assertEqual(
            table.offsets.shape[0], states.shape[0] * 8 + 1
        )
        self.assertGreater(table.outgoing_masks.shape[0], 0)
        row_counts = []
        for position, state in enumerate(states):
            bits = int(table.bitstrings[position])
            occupied = np.flatnonzero(state)
            row_count = 0
            for left in range(len(occupied)):
                for right in range(left + 1, len(occupied)):
                    total = int(occupied[left] + occupied[right]) % 8
                    start, stop = table.bounds(position, total)
                    row_count += stop - start
            row_counts.append(row_count)
            for total_momentum in range(8):
                start, stop = table.bounds(position, total_momentum)
                for mask in table.outgoing_masks[start:stop]:
                    self.assertEqual(bits & int(mask), 0)
                    orbitals = [
                        orbital
                        for orbital in range(8)
                        if int(mask) & (1 << orbital)
                    ]
                    self.assertEqual(len(orbitals), 2)
                    self.assertEqual(
                        sum(orbitals) % 8, total_momentum
                    )
        self.assertEqual(table.max_row_nnz, max(row_counts))

    def test_momentum_scattering_sampler_preserves_constraints(self):
        system_cfg = SystemConfig(
            nphi=8,
            p=1,
            q=2,
            coulomb_cut=4,
            disable_numba_coulomb=True,
        )
        basis = build_basis(system_cfg)
        bitstrings = sector_basis_bitstrings(basis, 0)
        states = bitstrings_to_occupations(bitstrings, 8)
        net_cfg = NetworkConfig(
            k=1,
            hidden_sizes=(8,),
            init_std=0.0,
        )
        params = init_params(11, 8, net_cfg)
        sampler = MetropolisSamplerJAX(
            params,
            states,
            32,
            net_cfg,
            seed=13,
            proposal_mode="momentum_scattering",
        )
        self.assertIsInstance(
            sampler.allowed_outgoing_cpu.offsets, np.ndarray
        )
        self.assertIsInstance(sampler.positions_host, np.ndarray)
        self.assertEqual(
            sampler.cpu_proposal_backend,
            "numba_parallel" if NUMBA_AVAILABLE else "numpy",
        )
        observed_change = False
        for _ in range(12):
            old_positions = np.asarray(sampler.positions)
            sampler.step()
            new_positions = np.asarray(sampler.positions)
            self.assertTrue(
                np.all(
                    [
                        len(set(row.tolist())) == net_cfg.k
                        for row in new_positions
                    ]
                )
            )
            for old_row, new_row in zip(old_positions, new_positions):
                for old_position, new_position in zip(old_row, new_row):
                    old_state = states[old_position]
                    new_state = states[new_position]
                    self.assertEqual(
                        int(np.sum(old_state)), int(np.sum(new_state))
                    )
                    self.assertEqual(
                        int(np.dot(np.arange(8), old_state)) % 8,
                        int(np.dot(np.arange(8), new_state)) % 8,
                    )
                    if old_position != new_position:
                        observed_change = True
                        removed = np.flatnonzero(
                            (old_state == 1) & (new_state == 0)
                        )
                        added = np.flatnonzero(
                            (old_state == 0) & (new_state == 1)
                        )
                        self.assertEqual(len(removed), 2)
                        self.assertEqual(len(added), 2)
                        self.assertEqual(
                            int(np.sum(removed)) % 8,
                            int(np.sum(added)) % 8,
                        )
        self.assertTrue(observed_change)

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
    def test_numba_and_numpy_proposals_match(self):
        system_cfg = SystemConfig(nphi=8, p=1, q=2)
        basis = build_basis(system_cfg)
        states = bitstrings_to_occupations(
            sector_basis_bitstrings(basis, 0), 8
        )
        net_cfg = NetworkConfig(k=2, hidden_sizes=(8,))
        params = init_params(17, 8, net_cfg)
        sampler = MetropolisSamplerJAX(
            params,
            states,
            32,
            net_cfg,
            seed=19,
            proposal_mode="momentum_scattering",
        )
        rng = np.random.default_rng(23)
        rows = rng.integers(
            0, net_cfg.k, size=sampler.n_chains, dtype=np.int64
        )
        incoming_slots = rng.integers(
            0,
            sampler.allowed_outgoing_cpu.occupied_pair_slots.shape[0],
            size=sampler.n_chains,
            dtype=np.int64,
        )
        outgoing_uniform = rng.random(sampler.n_chains)
        numpy_result = sampler._momentum_scattering_proposals_numpy(
            rows, incoming_slots, outgoing_uniform
        )
        table = sampler.allowed_outgoing_cpu
        numba_result = _momentum_scattering_proposals_numba(
            sampler.positions_host,
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
        np.testing.assert_array_equal(numba_result[0], numpy_result[0])
        np.testing.assert_array_equal(numba_result[1], numpy_result[1])
        np.testing.assert_allclose(
            numba_result[2], numpy_result[2], rtol=0, atol=0
        )

    def test_sliced_and_unsliced_local_energies_match(self):
        params = init_params(1234, 6, self.net_cfg)
        positions = jnp.asarray([[0], [1]], dtype=jnp.int64)
        direct = local_energies(
            params,
            self.states,
            self.sparse,
            positions,
            self.net_cfg,
        )
        sliced = local_energies_sliced(
            params,
            self.states,
            self.sparse.diag[np.asarray(positions)],
            self.sparse.cols[np.asarray(positions)],
            self.sparse.vals[np.asarray(positions)],
            self.sparse.mask[np.asarray(positions)],
            positions,
            self.net_cfg,
        )
        np.testing.assert_allclose(
            np.asarray(direct), np.asarray(sliced), rtol=1e-12, atol=1e-12
        )

    def test_sliced_neighbor_deduplication_matches_full_sector(self):
        params = init_params(1234, 6, self.net_cfg)
        positions = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int64)
        args = (
            params,
            self.states,
            self.sparse.diag[np.asarray(positions)],
            self.sparse.cols[np.asarray(positions)],
            self.sparse.vals[np.asarray(positions)],
            self.sparse.mask[np.asarray(positions)],
            positions,
            self.net_cfg,
        )
        full_sector = local_energies_sliced(*args, deduplicate_neighbors=False)
        deduplicated = local_energies_sliced(*args, deduplicate_neighbors=True)
        padded = local_energies_sliced(
            *args,
            deduplicate_neighbors=True,
            neighbor_padding_size=int(np.prod(args[3].shape)) + 7,
        )
        np.testing.assert_allclose(
            np.asarray(full_sector),
            np.asarray(deduplicated),
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            np.asarray(deduplicated),
            np.asarray(padded),
            rtol=2e-12,
            atol=2e-12,
        )

    def test_bitstring_neighbor_rows_match_indexed_local_energy(self):
        params = init_params(1234, 6, self.net_cfg)
        sector_bits = sector_basis_bitstrings(self.basis, 0)
        row_positions = np.asarray([[0], [1], [0]], dtype=np.int64)
        indexed = self.hamiltonian.build_sparse_hamiltonian_rows_subset(
            0, self.basis, row_positions
        )
        direct_rows = self.hamiltonian.build_sparse_hamiltonian_bitstring_rows(
            sector_bits[row_positions]
        )
        indexed_eloc = local_energies_sliced(
            params,
            self.states,
            jnp.asarray(indexed.diag),
            jnp.asarray(indexed.cols),
            jnp.asarray(indexed.vals),
            jnp.asarray(indexed.mask),
            jnp.asarray(row_positions),
            self.net_cfg,
            deduplicate_neighbors=True,
        )

        sampled_bits = jnp.asarray(
            sector_bits[row_positions].reshape(-1), dtype=jnp.uint64
        )
        local_positions = jnp.arange(sampled_bits.size, dtype=jnp.int64)[:, None]
        direct_eloc = local_energies_sliced(
            params,
            sampled_bits,
            jnp.asarray(direct_rows.diag),
            jnp.asarray(direct_rows.neighbor_bitstrings),
            jnp.asarray(direct_rows.vals),
            jnp.asarray(direct_rows.mask),
            local_positions,
            self.net_cfg,
            deduplicate_neighbors=True,
            neighbor_bitstrings_pos=jnp.asarray(
                direct_rows.neighbor_bitstrings
            ),
        )
        np.testing.assert_allclose(
            np.asarray(direct_eloc),
            np.asarray(indexed_eloc),
            rtol=2e-12,
            atol=2e-12,
        )

    def test_row_streaming_matches_full_local_energies(self):
        params = init_params(1234, 6, self.net_cfg)
        positions = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int64)
        direct = local_energies(
            params,
            self.states,
            self.sparse,
            positions,
            self.net_cfg,
        )
        streamed = local_energies(
            params,
            self.states,
            self.sparse,
            positions,
            self.net_cfg,
            row_batch_size=2,
        )
        np.testing.assert_allclose(
            np.asarray(direct), np.asarray(streamed), rtol=1e-12, atol=1e-12
        )

    def test_local_energy_uses_full_sector_wavefunction_by_default(self):
        params = init_params(1234, 6, self.net_cfg)
        positions = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int64)
        sector_size = int(self.states.shape[0])
        original = optimization_module.wavefunction_matrix
        call_sizes = []

        def wrapped(current_params, states, cfg):
            call_sizes.append(int(states.shape[0]))
            return original(current_params, states, cfg)

        with mock.patch.object(
            optimization_module,
            "wavefunction_matrix",
            side_effect=wrapped,
        ):
            local_energies(
                params,
                self.states,
                self.sparse,
                positions,
                self.net_cfg,
            )

        self.assertEqual(call_sizes, [sector_size])

    def test_neighbor_deduplication_is_opt_in_and_numerically_equal(self):
        params = init_params(1234, 6, self.net_cfg)
        positions = jnp.asarray([[0], [1], [0], [1]], dtype=jnp.int64)
        direct = local_energies(
            params,
            self.states,
            self.sparse,
            positions,
            self.net_cfg,
            deduplicate_neighbors=False,
        )
        deduplicated = local_energies(
            params,
            self.states,
            self.sparse,
            positions,
            self.net_cfg,
            deduplicate_neighbors=True,
        )
        np.testing.assert_allclose(
            np.asarray(direct),
            np.asarray(deduplicated),
            rtol=2e-12,
            atol=2e-12,
        )

    def test_vmc_profilers_report_expected_keys(self):
        params = init_params(1234, 6, self.net_cfg)
        positions = jnp.asarray([[0], [1]], dtype=jnp.int64)
        positions_np = np.asarray(positions)
        sliced = (
            self.sparse.diag[positions_np],
            self.sparse.cols[positions_np],
            self.sparse.vals[positions_np],
            self.sparse.mask[positions_np],
        )
        loss_timing = profile_vmc_loss_sliced(
            params,
            self.states,
            sliced[0],
            sliced[1],
            sliced[2],
            sliced[3],
            positions,
            self.net_cfg,
        )
        autodiff_timing = profile_vmc_autodiff_sliced(
            params,
            self.states,
            sliced[0],
            sliced[1],
            sliced[2],
            sliced[3],
            positions,
            self.net_cfg,
        )
        for key in (
            "logpsi",
            "logpsi_state_gather",
            "logpsi_network_forward",
            "logpsi_slogdet",
            "sampled_forward",
            "sampled_state_gather",
            "sampled_network_forward",
            "neighbor_dedup_cpu",
            "neighbor_cols_d2h",
            "neighbor_unique_cpu",
            "neighbor_indices_h2d",
            "neighbor_state_gather",
            "neighbor_forward",
            "neighbor_forward_first",
            "neighbor_forward_warm",
            "neighbor_forward_compile_gap",
            "neighbor_inverse_gather",
            "offdiag_segment_scatter",
            "offdiag_scatter",
            "solve_trace",
            "loss_reduction",
            "profiled_total",
            "profiled_warm_total",
            "neighbor_total",
            "neighbor_unique",
            "neighbor_evaluated",
            "neighbor_deduplicated",
        ):
            self.assertIn(key, loss_timing)
        self.assertEqual(
            loss_timing["neighbor_evaluated"],
            loss_timing["neighbor_total"],
        )
        self.assertEqual(loss_timing["neighbor_deduplicated"], 0)
        for key in (
            "loss_forward_eval",
            "loss_value_and_grad_profiled",
            "loss_backward_extra",
        ):
            self.assertIn(key, autodiff_timing)
            self.assertGreaterEqual(float(autodiff_timing[key]), 0.0)
        self.assertGreaterEqual(wavefunction_compile_cache_size(), 0)

    def test_all_optimizer_updates_are_finite(self):
        params = init_params(3, 6, self.net_cfg)
        train_cfg = TrainingConfig(optimize_method="projected")

        def loss_fn(current_params):
            return projected_loss(
                current_params,
                self.states.astype(jnp.float32),
                jnp.asarray(self.h_exact, dtype=jnp.complex64),
                1,
                self.net_cfg,
                train_cfg,
            )[0]

        grad_tree = jax.grad(loss_fn)(params)
        for name in (
            "signed_gradient",
            "natural_gradient",
            "adam",
            "adamw",
            "muon",
        ):
            with self.subTest(optimizer=name):
                opt_cfg = OptimizerConfig(
                    name=name,
                    lr=1e-3,
                    cg_maxiter=20,
                )
                state = init_optimizer_state(params, opt_cfg, 1)
                updated, _, stats = apply_optimizer_update(
                    params,
                    self.states,
                    grad_tree,
                    1,
                    self.net_cfg,
                    opt_cfg,
                    state,
                    1,
                )
                self.assertTrue(all_finite(updated))
                self.assertTrue(np.isfinite(stats["grad_norm"]))
                self.assertTrue(np.isfinite(stats["lr"]))

    def test_multi_step_lr_schedule(self):
        opt_cfg = OptimizerConfig(lr=8e-3, multi_step_lr=True)
        by_step = [
            learning_rate_at_step(opt_cfg, step, 8)
            for step in range(1, 9)
        ]
        self.assertEqual(
            by_step,
            [
                8e-3,
                8e-3,
                4e-3,
                4e-3,
                2e-3,
                2e-3,
                1e-3,
                1e-3,
            ],
        )
        self.assertEqual(learning_rate_at_step(opt_cfg, 9, 8), 1e-3)
        self.assertEqual(learning_rate_from_count(opt_cfg, 0, 8), 8e-3)
        self.assertEqual(learning_rate_from_count(opt_cfg, 7, 8), 1e-3)

    def test_default_lr_schedule_preserves_smooth_decay(self):
        opt_cfg = OptimizerConfig(lr=1e-2, decay_rate=1000)
        self.assertAlmostEqual(
            learning_rate_at_step(opt_cfg, 1, 10000),
            1e-2 / 1.001,
        )
        self.assertAlmostEqual(
            learning_rate_from_count(opt_cfg, 0, 10000),
            1e-2,
        )

    def test_constant_lr_schedule(self):
        opt_cfg = OptimizerConfig(lr=2e-3, constant_lr=True)
        for step in (1, 10, 1000):
            self.assertEqual(
                learning_rate_at_step(opt_cfg, step, 10000), 2e-3
            )
        for count in (0, 10, 9999):
            self.assertEqual(
                learning_rate_from_count(opt_cfg, count, 10000), 2e-3
            )

    def test_cyclic_lr_schedule(self):
        opt_cfg = OptimizerConfig(
            lr=8e-3,
            cyclic_lr=True,
            cyclic_lr_min=2e-3,
            cyclic_lr_period=4,
            cyclic_lr_mode="triangular",
        )
        by_step = [
            learning_rate_at_step(opt_cfg, step, 10000)
            for step in range(1, 9)
        ]
        self.assertEqual(
            by_step,
            [
                8e-3,
                5e-3,
                2e-3,
                5e-3,
                8e-3,
                5e-3,
                2e-3,
                5e-3,
            ],
        )
        self.assertEqual(learning_rate_from_count(opt_cfg, 0, 10000), 8e-3)
        self.assertEqual(learning_rate_from_count(opt_cfg, 2, 10000), 2e-3)

    def test_cyclic_lr_triangular2_schedule(self):
        opt_cfg = OptimizerConfig(
            lr=8e-3,
            cyclic_lr=True,
            cyclic_lr_min=2e-3,
            cyclic_lr_period=4,
            cyclic_lr_mode="triangular2",
        )
        by_count = [
            learning_rate_from_count(opt_cfg, count, 10000)
            for count in range(8)
        ]
        self.assertEqual(
            by_count,
            [
                8e-3,
                5e-3,
                2e-3,
                5e-3,
                5e-3,
                3.5e-3,
                2e-3,
                3.5e-3,
            ],
        )

    def test_cyclic_lr_default_min_uses_one_eighth_peak(self):
        opt_cfg = OptimizerConfig(
            lr=8e-3,
            cyclic_lr=True,
            cyclic_lr_period=4,
        )
        self.assertEqual(learning_rate_from_count(opt_cfg, 0, 10000), 8e-3)
        self.assertEqual(learning_rate_from_count(opt_cfg, 2, 10000), 1e-3)

    def test_original_small_system_regression(self):
        system_cfg = SystemConfig(
            nphi=6,
            p=1,
            q=3,
            coulomb_cut=12,
            disable_numba_coulomb=True,
        )
        net_cfg = NetworkConfig()
        basis = build_basis(system_cfg)
        hamiltonian = TorusCoulombHamiltonian(system_cfg)
        hamiltonian.generate_coulomb_matrix()
        h_exact = hamiltonian.build_exact_hamiltonian(0, basis)
        states = bitstrings_to_occupations(
            sector_basis_bitstrings(basis, 0), 6
        )
        params = init_params(1234, 6, net_cfg)
        nqs, overlap = projected_diagonalization(
            params, states, h_exact, net_cfg, 1e-8
        )
        exact = np.linalg.eigvalsh(h_exact)
        self.assertEqual(states.shape[0], 2)
        if jax.__version__ == "0.4.30":
            self.assertAlmostEqual(
                float(nqs[0]), -0.10387761458442817, delta=1e-10
            )
            self.assertAlmostEqual(
                float(overlap[0]), 0.99999994039535522, delta=1e-10
            )
        else:
            self.assertTrue(np.all(np.isfinite(nqs)))
            np.testing.assert_allclose(
                overlap, np.ones_like(overlap), rtol=1e-6, atol=1e-6
            )
        np.testing.assert_allclose(
            exact,
            [-0.15239929572985686, -0.10373679374853262],
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import itertools
import unittest
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from nqs_torus_laqx_nes.config import ContinuousConfig, NetworkConfig, SystemConfig
from nqs_torus_laqx_nes.continuous import (
    ContinuousMetropolisSamplerJAX,
    build_continuous_evaluation_plan,
    compact_overlap_functions,
    compact_overlap_product_basis,
    compact_overlap_support_mask,
    continuous_psi_hpsi,
    continuous_vmc_surrogate_loss,
    continuous_wavefunction,
    fourier_product_basis,
    fourier_support_mask,
)
from nqs_torus_laqx_nes.network import (
    batched_wavefunction_matrix,
    init_params,
    projected_diagonalization,
    projected_diagonalization_sparse,
    wavefunction_matrix,
)
from nqs_torus_laqx_nes.physics import (
    TorusCoulombHamiltonian,
    bitstrings_to_occupations,
    build_basis,
    sector_basis_bitstrings,
)


class ContinuousOccupationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system_cfg = SystemConfig(
            nphi=4,
            p=1,
            q=2,
            tk=0,
            coulomb_cut=3,
            disable_numba_coulomb=True,
        )
        cls.net_cfg = NetworkConfig(
            hidden_sizes=(8,), forward_batch_size=2
        )
        cls.continuous_cfg = ContinuousConfig()
        cls.basis = build_basis(cls.system_cfg)
        cls.hamiltonian = TorusCoulombHamiltonian(cls.system_cfg)
        cls.hamiltonian.generate_coulomb_matrix()
        cls.sparse_cpu = (
            cls.hamiltonian.build_sparse_hamiltonian_rows(0, cls.basis)
        )
        cls.sparse = cls.sparse_cpu.to_jax()
        cls.plan_cpu = build_continuous_evaluation_plan(cls.sparse_cpu)
        cls.plan = cls.plan_cpu.to_jax()
        cls.h_exact = cls.hamiltonian.build_exact_hamiltonian(
            0, cls.basis
        )
        bitstrings = sector_basis_bitstrings(cls.basis, 0)
        cls.states_np = bitstrings_to_occupations(bitstrings, 4)
        cls.states = jnp.asarray(cls.states_np, dtype=jnp.float64)
        cls.params = init_params(1234, 4, cls.net_cfg)

    def test_fourier_basis_is_periodic_and_orthonormal(self):
        coordinates = jnp.asarray(
            [[0.13, 0.27, 0.41, 0.59]], dtype=jnp.float64
        )
        shifted = coordinates.at[:, 2].add(1.0)
        values = fourier_product_basis(
            coordinates, self.states, harmonic=1
        )
        shifted_values = fourier_product_basis(
            shifted, self.states, harmonic=1
        )
        np.testing.assert_allclose(
            shifted_values, values, rtol=0, atol=1e-14
        )

        grid = np.asarray(
            list(itertools.product((0.0, 1.0 / 3.0, 2.0 / 3.0), repeat=4))
        )
        chi = np.asarray(
            fourier_product_basis(grid, self.states, harmonic=1)
        )
        overlap = chi.conj().T @ chi / grid.shape[0]
        np.testing.assert_allclose(
            overlap,
            np.eye(self.states.shape[0]),
            rtol=0,
            atol=1e-14,
        )

    def test_compact_overlap_functions_are_continuous_and_orthonormal(self):
        epsilon = 0.2
        a = 0.5 - epsilon / 2.0
        b = 0.5 + epsilon / 2.0
        points = jnp.asarray([0.0, a, b, 1.0])
        f_values, g_values = compact_overlap_functions(
            points, epsilon
        )
        f_values = np.asarray(f_values)
        g_values = np.asarray(g_values)
        self.assertGreater(f_values[0], 0.0)
        self.assertEqual(f_values[-1], 0.0)
        self.assertEqual(g_values[0], 0.0)
        self.assertGreater(g_values[-1], 0.0)
        self.assertAlmostEqual(f_values[0], f_values[1], delta=1e-14)
        self.assertAlmostEqual(g_values[2], g_values[3], delta=1e-14)

        def values_at(x):
            f_value, g_value = compact_overlap_functions(
                jnp.asarray(x), epsilon
            )
            return float(f_value), float(g_value)

        f_norm = quad(
            lambda x: values_at(x)[0] ** 2,
            0.0,
            1.0,
            points=[a, b],
            epsabs=1e-12,
        )[0]
        g_norm = quad(
            lambda x: values_at(x)[1] ** 2,
            0.0,
            1.0,
            points=[a, b],
            epsabs=1e-12,
        )[0]
        overlap = quad(
            lambda x: values_at(x)[0] * values_at(x)[1],
            0.0,
            1.0,
            points=[a, 0.5, b],
            epsabs=1e-12,
        )[0]
        self.assertAlmostEqual(f_norm, 1.0, delta=1e-11)
        self.assertAlmostEqual(g_norm, 1.0, delta=1e-11)
        self.assertAlmostEqual(overlap, 0.0, delta=1e-11)

    def test_compact_overlap_support_has_controlled_overlap(self):
        epsilon = 0.2
        occupations = jnp.asarray([[0.0], [1.0]])
        coordinates = jnp.asarray(
            [[0.2], [0.45], [0.55], [0.8]], dtype=jnp.float64
        )
        support = np.asarray(
            compact_overlap_support_mask(
                coordinates, occupations, epsilon
            )
        )
        np.testing.assert_array_equal(
            support,
            [
                [True, False],
                [True, True],
                [True, True],
                [False, True],
            ],
        )

    def test_compact_overlap_evaluator_matches_dense_reference(self):
        compact_cfg = ContinuousConfig(
            basis="compact_overlap", overlap_epsilon=0.4
        )
        coordinates = jnp.asarray(
            [
                [0.31, 0.48, 0.63, 0.72],
                [0.20, 0.80, 0.25, 0.75],
            ],
            dtype=jnp.float64,
        )
        psi, hpsi, support = continuous_psi_hpsi(
            self.params,
            self.states,
            self.sparse,
            self.plan,
            coordinates,
            self.net_cfg,
            compact_cfg,
        )
        coefficients = np.asarray(
            wavefunction_matrix(self.params, self.states, self.net_cfg)
        )[:, 0].astype(np.complex128)
        chi = np.asarray(
            compact_overlap_product_basis(
                coordinates,
                self.states,
                compact_cfg.overlap_epsilon,
            )
        )
        np.testing.assert_allclose(
            psi, chi @ coefficients, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            hpsi,
            chi @ (self.h_exact @ coefficients),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertTrue(
            np.all(np.any(np.asarray(support), axis=1))
        )

    def test_continuous_integrals_match_discrete_norm_and_energy(self):
        grid = np.asarray(
            list(itertools.product((0.0, 1.0 / 3.0, 2.0 / 3.0), repeat=4))
        )
        chi = np.asarray(fourier_product_basis(grid, self.states))
        coefficients = np.asarray(
            wavefunction_matrix(self.params, self.states, self.net_cfg)
        )[:, 0].astype(np.complex128)
        psi = chi @ coefficients
        hpsi = chi @ (self.h_exact @ coefficients)
        continuous_norm = np.mean(np.abs(psi) ** 2)
        discrete_norm = np.vdot(coefficients, coefficients).real
        continuous_energy = np.mean(np.conj(psi) * hpsi) / continuous_norm
        discrete_energy = (
            np.vdot(coefficients, self.h_exact @ coefficients)
            / discrete_norm
        )
        self.assertAlmostEqual(continuous_norm, discrete_norm, delta=1e-13)
        self.assertAlmostEqual(
            continuous_energy.real, discrete_energy.real, delta=1e-13
        )
        self.assertAlmostEqual(continuous_energy.imag, 0.0, delta=1e-13)

    def test_sparse_projected_diagonalization_matches_dense(self):
        dense_evals, dense_overlap = projected_diagonalization(
            self.params,
            self.states_np,
            self.h_exact,
            self.net_cfg,
            overlap_shift=1e-8,
        )
        sparse_evals, sparse_overlap = projected_diagonalization_sparse(
            self.params,
            self.states_np,
            self.sparse,
            self.net_cfg,
            overlap_shift=1e-8,
        )
        np.testing.assert_allclose(
            sparse_evals, dense_evals, rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            sparse_overlap, dense_overlap, rtol=0, atol=1e-13
        )

    def test_global_neighbor_dedup_matches_dense_reference(self):
        coordinates = jnp.asarray(
            [
                [0.11, 0.23, 0.37, 0.41],
                [0.79, 0.61, 0.43, 0.17],
            ],
            dtype=jnp.float64,
        )
        expected_unique = np.unique(
            np.concatenate(
                (
                    np.arange(self.states.shape[0]),
                    np.asarray(self.sparse_cpu.cols)[
                        np.asarray(self.sparse_cpu.mask) > 0
                    ],
                )
            )
        )
        np.testing.assert_array_equal(
            self.plan_cpu.unique_state_indices, expected_unique
        )

        with patch(
            "nqs_torus_laqx_nes.continuous.batched_wavefunction_matrix",
            wraps=batched_wavefunction_matrix,
        ) as forward:
            psi, hpsi, _ = continuous_psi_hpsi(
                self.params,
                self.states,
                self.sparse,
                self.plan,
                coordinates,
                self.net_cfg,
                self.continuous_cfg,
            )
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(
            int(forward.call_args.args[1].shape[0]),
            self.plan_cpu.unique_count,
        )

        coefficients = np.asarray(
            wavefunction_matrix(self.params, self.states, self.net_cfg)
        )[:, 0].astype(np.complex128)
        chi = np.asarray(fourier_product_basis(coordinates, self.states))
        np.testing.assert_allclose(
            np.asarray(psi), chi @ coefficients, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(hpsi),
            chi @ (self.h_exact @ coefficients),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_forward_batch_size_does_not_change_values_or_gradient(self):
        coordinates = jnp.asarray(
            np.random.default_rng(3).random((5, 4)), dtype=jnp.float64
        )
        small_batch = NetworkConfig(
            hidden_sizes=(8,), forward_batch_size=1
        )
        large_batch = NetworkConfig(
            hidden_sizes=(8,), forward_batch_size=128
        )
        psi_small, hpsi_small, _ = continuous_psi_hpsi(
            self.params,
            self.states,
            self.sparse,
            self.plan,
            coordinates,
            small_batch,
            self.continuous_cfg,
        )
        psi_large, hpsi_large, _ = continuous_psi_hpsi(
            self.params,
            self.states,
            self.sparse,
            self.plan,
            coordinates,
            large_batch,
            self.continuous_cfg,
        )
        np.testing.assert_allclose(
            psi_small, psi_large, rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            hpsi_small, hpsi_large, rtol=0, atol=1e-13
        )

        def loss_for(current_params, net_cfg):
            return continuous_vmc_surrogate_loss(
                current_params,
                self.states,
                self.sparse,
                self.plan,
                coordinates,
                net_cfg,
                self.continuous_cfg,
            )[0]

        grad_small = jax.grad(
            lambda current_params: loss_for(
                current_params, small_batch
            )
        )(self.params)
        grad_large = jax.grad(
            lambda current_params: loss_for(
                current_params, large_batch
            )
        )(self.params)
        for left, right in zip(
            jax.tree_util.tree_leaves(grad_small),
            jax.tree_util.tree_leaves(grad_large),
        ):
            np.testing.assert_allclose(
                np.asarray(left, dtype=np.float32),
                np.asarray(right, dtype=np.float32),
                rtol=0,
                atol=1e-6,
            )

    def test_support_union_and_reconstruction(self):
        coordinates = jnp.asarray(
            [[0.2, 0.3, 0.4, 0.5], [0.7, 0.6, 0.1, 0.9]]
        )
        support = np.asarray(
            fourier_support_mask(coordinates, self.states)
        )
        self.assertTrue(np.all(support))
        self.assertEqual(
            np.flatnonzero(np.any(support, axis=0)).tolist(),
            list(range(self.states.shape[0])),
        )
        psi, returned_support = continuous_wavefunction(
            self.params,
            self.states,
            coordinates,
            self.plan.active_rows,
            self.net_cfg,
            self.continuous_cfg,
        )
        coefficients = np.asarray(
            wavefunction_matrix(self.params, self.states, self.net_cfg)
        )[:, 0]
        chi = np.asarray(fourier_product_basis(coordinates, self.states))
        np.testing.assert_allclose(psi, chi @ coefficients, atol=1e-12)
        np.testing.assert_array_equal(returned_support, support)

    def test_periodic_sampler_adapts_once_then_freezes(self):
        sampler = ContinuousMetropolisSamplerJAX(
            self.params,
            self.states_np,
            self.plan.active_rows,
            n_chains=16,
            net_cfg=self.net_cfg,
            continuous_cfg=self.continuous_cfg,
            seed=17,
        )
        initial_width = sampler.proposal_width
        coordinates, _ = sampler.sample(
            n_burn=4,
            n_between=2,
            n_samples=2,
            adapt_width=True,
            freeze_after_burn=True,
        )
        self.assertTrue(np.all(coordinates >= 0.0))
        self.assertTrue(np.all(coordinates < 1.0))
        self.assertTrue(sampler.width_frozen)
        self.assertNotEqual(sampler.proposal_width, initial_width)
        frozen_width = sampler.proposal_width
        sampler.sample(
            n_burn=3,
            n_between=1,
            n_samples=1,
            adapt_width=True,
            freeze_after_burn=True,
        )
        self.assertEqual(sampler.proposal_width, frozen_width)

    def test_compact_overlap_sampler_initializes_inside_support(self):
        compact_cfg = ContinuousConfig(
            basis="compact_overlap",
            overlap_epsilon=0.2,
            proposal_width=0.01,
        )
        sampler = ContinuousMetropolisSamplerJAX(
            self.params,
            self.states_np,
            self.plan.active_rows,
            n_chains=32,
            net_cfg=self.net_cfg,
            continuous_cfg=compact_cfg,
            seed=29,
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(sampler.logprob))))
        coordinates, acceptance = sampler.sample(
            n_burn=2,
            n_between=1,
            n_samples=1,
        )
        self.assertTrue(np.all(coordinates >= 0.0))
        self.assertTrue(np.all(coordinates < 1.0))
        self.assertTrue(np.isfinite(acceptance))


if __name__ == "__main__":
    unittest.main()


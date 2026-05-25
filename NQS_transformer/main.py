from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ViT import ViTWaveFunction, count_model_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_vit_model(N, token_size, d_model, num_heads, mlp_dim, num_layers, k):
    model = ViTAdapter(
        num_sites=N,
        token_size=token_size,
        d_model=d_model,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        k=k,
    ).to(device)
    count_model_parameters(model)
    return model


def build_samplers(model, N, n_chains, h_values, seed):
    return {
        field: MetropolisSampler(
            model,
            N=N,
            n_chains=n_chains,
            gamma=field,
            seed=seed + idx,
        )
        for idx, field in enumerate(h_values)
    }


def maybe_print_exact_energies(N, J, h_values, k):
    exact_E0 = {}
    if N <= 14:
        for field in h_values:
            exact_E0[field] = exact_ground_state_energy_tfim(N, J, field, k)
            print(f"[Exact] h = {field:.1f}, E0 = {exact_E0[field]}")
    return exact_E0


def plot_projection_vs_exact_energies(projection_info, exact_E0, output_path):
    common_fields = [field for field in projection_info if field in exact_E0]
    if not common_fields:
        print("[Plot] skipped because projection energies or exact diagonalization energies are unavailable.")
        return

    common_fields = sorted(common_fields)
    projected_levels = [
        np.asarray(projection_info[field]["generalized_evals"], dtype=np.float64)
        for field in common_fields
    ]
    exact_levels = [
        np.asarray(exact_E0[field], dtype=np.float64)
        for field in common_fields
    ]
    n_levels = min(
        min(levels.size for levels in projected_levels),
        min(levels.size for levels in exact_levels),
    )

    if n_levels == 0:
        print("[Plot] skipped because there are no comparable energy levels.")
        return

    h_axis = np.asarray(common_fields, dtype=np.float64)
    projected_arr = np.stack([levels[:n_levels] for levels in projected_levels], axis=0)
    exact_arr = np.stack([levels[:n_levels] for levels in exact_levels], axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cmap = plt.get_cmap("tab10")

    for level_idx in range(n_levels):
        color = cmap(level_idx % 10)
        ax.plot(
            h_axis,
            projected_arr[:, level_idx],
            marker="o",
            linewidth=1.8,
            color=color,
            label=f"Projected level {level_idx}",
        )
        ax.plot(
            h_axis,
            exact_arr[:, level_idx],
            marker="s",
            linewidth=1.6,
            linestyle="--",
            color=color,
            label=f"Exact level {level_idx}",
        )

        diff = np.abs(projected_arr[:, level_idx] - exact_arr[:, level_idx])
        print(
            f"[Plot] level = {level_idx} | compared over {len(common_fields)} h values | "
            f"max |E_proj - E_exact| = {diff.max():.6e}"
        )

    ax.set_xlabel("h")
    ax.set_ylabel("Energy")
    ax.set_title("Projected-subspace energies vs exact diagonalization")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved comparison figure to {output_path}")


def apply_joint_update(model, delta_sum):
    sync_if_cuda()
    update_t0 = time.perf_counter()
    with torch.no_grad():
        theta = get_param_vector(model)
        theta = theta + delta_sum
        set_param_vector(model, theta)
    sync_if_cuda()
    return time.perf_counter() - update_t0


def refresh_all_samplers(samplers, h_values):
    for field in h_values:
        samplers[field]._refresh_cache()


def print_iteration_log(it, field, E_mean, E_var, info, sampler, sample_time, sr_time):
    timing = info["timing"]
    msg = (
        f"Iter {it:4d} | "
        f"h = {field:.1f} | "
        f"E = {E_mean.real.item(): .10f} | "
        f"Var = {E_var.item(): .6e} | "
        f"|g| = {info['grad_norm']:.3e} | "
        f"|delta| = {info['delta_norm']:.3e} | "
        f"CG = {info['cg_iters']} | "
        f"Acc = {sampler.acceptance_rate():.3f} | "
        f"Tsample = {sample_time:.3f}s | "
        f"TSR = {sr_time:.3f}s | "
        f"TD = {timing['d_total_s']:.3f}s | "
        f"TAD = {timing['d_autograd_s']:.3f}s | "
        f"TCG = {timing['cg_total_s']:.3f}s | "
        f"TDlt = {timing['delta_build_s']:.3f}s | "
        f"TUpd = {timing['param_update_s']:.3f}s"
    )
    print(msg)
    detail_msg = (
        f"    SR detail | "
        f"Tstats = {timing['sr_stats_total_s']:.3f}s | "
        f"TpostD = {timing['sr_stats_post_d_s']:.3f}s | "
        f"TD_fwd = {timing['d_forward_s']:.3f}s | "
        f"TD_inv = {timing['d_inv_s']:.3f}s | "
        f"TD_pack = {timing['d_pack_s']:.3f}s | "
        f"Nsamples = {timing['d_samples']}"
    )
    print(detail_msg)


def initialize_combined_sr_stats(stats):
    return {
        "D_centered": stats["D_centered"].clone(),
        "E_centered": stats["E_centered"].clone(),
        "E_mean_list": [stats["E_mean"]],
        "timing": dict(stats["timing"]),
    }


def accumulate_combined_sr_stats(combined_stats, stats):
    if combined_stats is None:
        return initialize_combined_sr_stats(stats)

    combined_stats["D_centered"] = combined_stats["D_centered"] + stats["D_centered"]
    combined_stats["E_centered"] = combined_stats["E_centered"] + stats["E_centered"]
    combined_stats["E_mean_list"].append(stats["E_mean"])

    for key, value in stats["timing"].items():
        if isinstance(value, (int, float)):
            combined_stats["timing"][key] = combined_stats["timing"].get(key, 0.0) + value

    return combined_stats


def sr_step_cg_from_combined_stats(
    model,
    combined_stats,
    lr=0.005,
    diag_shift=1e-3,
    cg_tol=1e-8,
    cg_maxiter=200,
):
    sync_if_cuda()
    sr_step_t0 = time.perf_counter()

    D_centered = combined_stats["D_centered"]
    E_centered = combined_stats["E_centered"]
    E_mean = combined_stats["E_mean_list"]
    M = len(E_mean)
    D_centered_cat = torch.cat(
        [torch.real(D_centered), torch.imag(D_centered)],
        dim=0,
    ).to(torch.float64)/M**(1/2)
    E_cat = torch.cat(
        [torch.real(E_centered), torch.imag(E_centered)],
        dim=0,
    ).to(torch.float64)/M**(1/2)

    A_min_sr = minSRLinearOperator(D_centered_cat, diag_shift=diag_shift)
    b_min_sr = -lr * E_cat
    x_min_sr, n_cg_sr, cg_timing = conjugate_gradient(
        A_min_sr,
        b_min_sr,
        tol=cg_tol,
        maxiter=cg_maxiter,
    )
    delta = D_centered_cat.T @ x_min_sr
    sync_if_cuda()
    update_t0 = time.perf_counter()
    with torch.no_grad():
        theta = get_param_vector(model)
        theta = theta + delta
        set_param_vector(model, theta)
    sync_if_cuda()
    param_update_s = time.perf_counter() - update_t0
    sync_if_cuda()
    sr_step_total_s = time.perf_counter() - sr_step_t0

    timing = dict(combined_stats["timing"])
    timing.update(cg_timing)
    timing["delta_build_s"] = 0.0
    timing["param_update_s"] = param_update_s
    timing["sr_step_total_s"] = sr_step_total_s

    info = {
        "E_mean": torch.mean(torch.stack(combined_stats["E_mean_list"])),
        #"grad_norm": torch.linalg.norm(force).item(),
        "delta_norm": torch.linalg.norm(delta).item(),
        "cg_iters": n_cg_sr,
        #"force_norm": torch.linalg.norm(F).item(),
        #"cov_norm": torch.linalg.norm(C).item(),
        "timing": timing,
        "delta": delta,
    }
    return info


class ViTAdapter(nn.Module):
    """Thin adapter so the NQS_VMC training flow can call a ViT model."""

    def __init__(
        self,
        num_sites: int,
        token_size: int,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        k: int,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.k = k
        self.vit = ViTWaveFunction(
            num_sites=num_sites,
            token_size=token_size,
            d_model=d_model,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_outputs=k,
            bias=bias,
            dropout=dropout,
        )

    def forward(self, states: torch.Tensor, gamma: torch.Tensor | float) -> torch.Tensor:
        return self.vit.log_wavefunction_matrix(states, gamma)

    def log_wavefunction_matrix(self, states: torch.Tensor, gamma: torch.Tensor | float) -> torch.Tensor:
        return self.vit.log_wavefunction_matrix(states, gamma)

    def wavefunction_matrix(self, states: torch.Tensor, gamma: torch.Tensor | float) -> torch.Tensor:
        return self.vit.wavefunction_matrix(states, gamma)


def compute_wavefunction_matrix(model, states, gamma):
    return model.wavefunction_matrix(states, gamma)


def compute_wavefunction_matrices(model, states, gamma):
    if states.dim() != 3:
        raise ValueError("states must have shape (n_chains, k, N).")

    n_chains, k, n_sites = states.shape
    if k != model.k:
        raise ValueError(
            f"Expected states.shape[1] == model.k == {model.k}, got {k}."
        )
    flat_states = states.reshape(n_chains * k, n_sites)
    phi_flat = model.wavefunction_matrix(flat_states, gamma)
    return phi_flat.reshape(n_chains, k, model.k)


def compute_logdet_and_logprob(psi_matrices):
    if psi_matrices.dim() != 3:
        raise ValueError("psi_matrices must have shape (n_chains, k, k).")
    if psi_matrices.size(-1) != psi_matrices.size(-2):
        raise ValueError("Each wavefunction matrix must be square.")

    sign, logabsdet = torch.linalg.slogdet(psi_matrices)
    logdet = torch.log(sign) + logabsdet.to(torch.complex128)
    logprob = 2.0 * logabsdet
    return logdet, logprob


def row_replacement_det_ratio(new_rows, psi_inv, row_indices):
    selected_cols = psi_inv[
        torch.arange(psi_inv.size(0), device=psi_inv.device),
        :,
        row_indices,
    ]
    return torch.einsum("bi,bi->b", new_rows, selected_cols)


def get_param_vector(model):
    return parameters_to_vector([p for p in model.parameters()])


def set_param_vector(model, vec):
    vector_to_parameters(vec, [p for p in model.parameters()])


def num_params(model):
    return sum(p.numel() for p in model.parameters())


class MetropolisSampler:
    def __init__(self, model, N, n_chains, gamma, seed=1234):
        self.model = model
        self.N = N
        self.n_chains = n_chains
        self.k = model.k
        self.gamma = gamma
        self.device = next(model.parameters()).device

        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=gen_device)
        self.generator.manual_seed(seed)

        self.states = self._random_states(n_chains)
        self.psi_matrices = None
        self.psi_inv = None
        self.logdet = None
        self.logprob = None
        self.chain_ids = torch.arange(n_chains, device=self.device)
        self.n_accepted = 0
        self.n_proposed = 0
        self._refresh_cache()

    def _random_states(self, n_chains):
        states = torch.empty(
            (n_chains, self.k, self.N),
            dtype=torch.float64,
            device=self.device,
        )

        for chain_id in range(n_chains):
            chain_rows = []
            while len(chain_rows) < self.k:
                row = torch.randint(
                    0,
                    2,
                    (self.N,),
                    generator=self.generator,
                    device=self.device,
                )
                row = 2.0 * row.to(torch.float64) - 1.0
                chain_rows.append(row)

            states[chain_id] = torch.stack(chain_rows, dim=0)

        return states

    @torch.no_grad()
    def _refresh_cache(self, max_tries=64):
        for _ in range(max_tries):
            psi_matrices = compute_wavefunction_matrices(self.model, self.states, self.gamma)
            sign, logabsdet = torch.linalg.slogdet(psi_matrices)
            valid = (sign != 0) & torch.isfinite(logabsdet)
            if valid.all():
                self.psi_matrices = psi_matrices
                self.psi_inv = torch.linalg.inv(psi_matrices)
                self.logdet, self.logprob = compute_logdet_and_logprob(psi_matrices)
                return

            invalid_ids = self.chain_ids[~valid]
            self.states[invalid_ids] = self._random_states(invalid_ids.numel())

        raise RuntimeError(
            "Failed to find non-singular wavefunction matrices after switching gamma. "
            "Try reducing k, changing initialization, or inspecting the model outputs."
        )

    @torch.no_grad()
    def set_gamma(self, gamma):
        self.gamma = gamma
        self._refresh_cache()

    @torch.no_grad()
    def sweep(self, record_psi=False):
        n_chains, k, N = self.states.shape

        for _ in range(k * N):
            config_indices = torch.randint(
                0, k, (n_chains,), generator=self.generator, device=self.device
            )
            flip_sites = torch.randint(
                0, N, (n_chains,), generator=self.generator, device=self.device
            )

            proposed_rows = self.states[self.chain_ids, config_indices].clone()
            proposed_rows[self.chain_ids, flip_sites] *= -1.0

            new_row_values = self.model.wavefunction_matrix(proposed_rows, self.gamma)
            det_ratio = row_replacement_det_ratio(new_row_values, self.psi_inv, config_indices)
            valid_ratio = torch.isfinite(det_ratio.real) & torch.isfinite(det_ratio.imag)
            valid_ratio = valid_ratio & (torch.abs(det_ratio) > 1e-12)

            safe_det_ratio = torch.where(
                valid_ratio,
                det_ratio,
                torch.ones_like(det_ratio),
            )
            logprob_new = self.logprob + 2.0 * torch.log(torch.abs(safe_det_ratio))
            logdet_new = self.logdet + torch.log(safe_det_ratio)

            log_acc = logprob_new - self.logprob
            rand_log = torch.log(
                torch.rand(n_chains, generator=self.generator, device=self.device)
            )
            accept = valid_ratio & (rand_log < torch.minimum(log_acc, torch.zeros_like(log_acc)))

            if accept.any():
                accept_ids = self.chain_ids[accept]
                accept_rows = config_indices[accept]
                accept_new_states = proposed_rows[accept]

                self.states[accept_ids, accept_rows] = accept_new_states
                self.psi_matrices[accept_ids, accept_rows] = new_row_values[accept]
                self.logdet[accept_ids] = logdet_new[accept]
                self.logprob[accept_ids] = logprob_new[accept]
                self.psi_inv[accept_ids] = torch.linalg.inv(self.psi_matrices[accept_ids])

            self.n_accepted += accept.sum().item()
            self.n_proposed += n_chains

    @torch.no_grad()
    def sample(self, n_burn, n_between, n_samples, record_psi_steps=False):
        for _ in range(n_burn):
            self.sweep(record_psi=record_psi_steps)

        states_list = []
        psi_matrices_list = []
        logprob_list = []

        for _ in range(n_samples):
            for _ in range(n_between):
                self.sweep(record_psi=record_psi_steps)
            states_list.append(self.states.clone())
            psi_matrices_list.append(self.psi_matrices.clone())
            logprob_list.append(self.logprob.clone())

        return (
            torch.cat(states_list, dim=0),
            torch.cat(psi_matrices_list, dim=0),
            torch.cat(logprob_list, dim=0),
        )

    def acceptance_rate(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


@torch.no_grad()
def h_psi_tfim_batch(model, states, psi_matrices, field, J):
    if states.dim() != 3:
        raise ValueError("states must have shape (M, k, N).")
    if psi_matrices.dim() != 3:
        raise ValueError("psi_matrices must have shape (M, k, k).")

    M, k, N = states.shape
    if psi_matrices.shape != (M, k, k):
        raise ValueError(
            f"Expected psi_matrices.shape == {(M, k, k)}, got {tuple(psi_matrices.shape)}."
        )

    diag_terms = -J * torch.sum(
        states * torch.roll(states, shifts=-1, dims=2),
        dim=2,
    )
    hp = diag_terms.unsqueeze(-1).to(torch.complex128) * psi_matrices.to(torch.complex128)

    flipped_rows = states.unsqueeze(2).repeat(1, 1, N, 1)
    site_ids = torch.arange(N, device=states.device).view(1, 1, N)
    flipped_rows[
        :,
        torch.arange(k, device=states.device).view(1, k, 1),
        site_ids,
        site_ids,
    ] *= -1.0
    flipped_rows_flat = flipped_rows.reshape(M * k * N, N)
    flipped_values = model.wavefunction_matrix(flipped_rows_flat, field).reshape(M, k, N, k)
    hp = hp - field * torch.sum(flipped_values, dim=2).to(torch.complex128)
    return hp


@torch.no_grad()
def local_energy_tfim_batch(model, states, psi_matrices, field, J):
    hp = h_psi_tfim_batch(model, states, psi_matrices, field, J)
    psi_inv = torch.linalg.inv(psi_matrices)
    local_matrix = torch.matmul(psi_inv.to(torch.complex128), hp)
    return torch.einsum("mii->m", local_matrix)


def compute_D_matrix(model, states, field):
    if states.dim() != 3:
        raise ValueError("states must have shape (M, k, N).")
    if states.size(1) != model.k:
        raise ValueError(
            f"Expected states.shape[1] == model.k == {model.k}, got {states.size(1)}."
        )

    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)
    M = states.shape[0]

    D = torch.empty((M, P), dtype=torch.complex128, device=device)
    timing = {
        "d_total_s": 0.0,
        "d_forward_s": 0.0,
        "d_inv_s": 0.0,
        "d_autograd_s": 0.0,
        "d_pack_s": 0.0,
        "d_samples": M,
    }

    sync_if_cuda()
    d_total_t0 = time.perf_counter()

    for m in range(M):
        model.zero_grad(set_to_none=True)

        sync_if_cuda()
        t0 = time.perf_counter()
        log_phi_matrix = model.log_wavefunction_matrix(states[m], field)
        sync_if_cuda()
        timing["d_forward_s"] += time.perf_counter() - t0

        sync_if_cuda()
        t0 = time.perf_counter()
        phi_matrix = torch.exp(log_phi_matrix)
        phi_inv = torch.linalg.inv(phi_matrix).detach()
        sync_if_cuda()
        timing["d_inv_s"] += time.perf_counter() - t0

        d_scalar = torch.einsum("ia,ai->", phi_inv, phi_matrix)

        sync_if_cuda()
        t0 = time.perf_counter()
        grads_re = torch.autograd.grad(
            torch.real(d_scalar),
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        grads_im = torch.autograd.grad(
            torch.imag(d_scalar),
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        sync_if_cuda()
        timing["d_autograd_s"] += time.perf_counter() - t0

        sync_if_cuda()
        t0 = time.perf_counter()
        g_re = parameters_to_vector(grads_re)
        g_im = parameters_to_vector(grads_im)
        D[m] = g_re.to(torch.complex128) + 1j * g_im.to(torch.complex128)
        sync_if_cuda()
        timing["d_pack_s"] += time.perf_counter() - t0

    sync_if_cuda()
    timing["d_total_s"] = time.perf_counter() - d_total_t0
    return D, timing


def compute_O_matrix(model, states, field):
    return compute_D_matrix(model, states, field)


def compute_sr_statistics(model, states, field, E_loc):
    if E_loc.dim() != 1:
        raise ValueError("E_loc must have shape (M,).")
    if E_loc.size(0) != states.size(0):
        raise ValueError(
            f"Expected E_loc.shape[0] == states.shape[0] == {states.size(0)}, got {E_loc.size(0)}."
        )

    sync_if_cuda()
    stats_t0 = time.perf_counter()
    D, d_timing = compute_D_matrix(model, states, field)
    D_mean = torch.mean(D, dim=0, keepdim=True)
    D_centered = D - D_mean

    E_mean = torch.mean(E_loc)
    E_centered = (E_loc - E_mean) / states.size(0)

    C = (D_centered.conj().T @ D_centered) / states.size(0)
    F = torch.mean(D_centered.conj() * E_centered.unsqueeze(1), dim=0)
    T = (D @ D_centered.conj().T) / states.size(0)
    sync_if_cuda()
    stats_total_s = time.perf_counter() - stats_t0
    return {
        "D": D,
        "D_mean": D_mean,
        "D_centered": D_centered,
        "E_mean": E_mean,
        "E_centered": E_centered,
        "C": C,
        "F": F,
        "T": T,
        "timing": {
            **d_timing,
            "sr_stats_total_s": stats_total_s,
            "sr_stats_post_d_s": max(0.0, stats_total_s - d_timing["d_total_s"]),
        },
    }


class SRLinearOperator:
    def __init__(self, D_centered, diag_shift=1e-3):
        self.Dc = D_centered
        self.M = D_centered.shape[0]
        self.P = D_centered.shape[1]
        self.diag_shift = diag_shift

    def matvec(self, v):
        v_c = v.to(torch.complex128)
        tmp = self.Dc @ v_c
        out = self.Dc.conj().T @ tmp
        out = torch.real(out) / self.M
        out = out + self.diag_shift * v
        return out


class minSRLinearOperator:
    def __init__(self, D_centered, diag_shift=1e-3):
        self.Dc = D_centered
        self.M = D_centered.shape[0]
        self.P = D_centered.shape[1]
        self.diag_shift = diag_shift

    def matvec(self, v):
        v_r = v.to(self.Dc.dtype)
        tmp = self.Dc.conj().T @ v_r
        out = self.Dc @ tmp
        out = out / self.M
        out = out + self.diag_shift * v
        return out

def conjugate_gradient(A, b, x0=None, tol=1e-10, maxiter=200):
    sync_if_cuda()
    cg_t0 = time.perf_counter()

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A.matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    if torch.sqrt(rs_old) < tol:
        sync_if_cuda()
        return x, 0, {"cg_total_s": time.perf_counter() - cg_t0}

    for it in range(1, maxiter + 1):
        Ap = A.matvec(p)
        denom = torch.dot(p, Ap)
        if torch.abs(denom) < 1e-30:
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) < tol:
            sync_if_cuda()
            return x, it, {"cg_total_s": time.perf_counter() - cg_t0}

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    sync_if_cuda()
    return x, it, {"cg_total_s": time.perf_counter() - cg_t0}


def sr_step_cg(model, states, field, E_loc, lr=0.005, diag_shift=1e-3, cg_tol=1e-8, cg_maxiter=200):
    sync_if_cuda()
    sr_step_t0 = time.perf_counter()
    stats = compute_sr_statistics(model, states, field, E_loc)
    C = stats["C"]
    D_centered = stats["D_centered"]
    F = stats["F"]
    E = stats["E_centered"]
    force = torch.real(F)
    # SR
    '''
    A_sr = SRLinearOperator(D_centered, diag_shift=diag_shift)
    b_sr = -lr * force
    delta, n_cg_sr, cg_timing = conjugate_gradient(
        A_sr,
        b_sr,
        tol=cg_tol,
        maxiter=cg_maxiter,
    )
    delta_build_s = 0.0
    '''
    # minSR with concatenated real-imag sample space:
    D_centered_cat = torch.cat(
        [torch.real(D_centered), torch.imag(D_centered)],
        dim=0,
    ).to(torch.float64)
    E_cat = torch.cat(
        [torch.real(E), torch.imag(E)],
        dim=0,
    ).to(torch.float64)

    A_min_sr = minSRLinearOperator(D_centered_cat, diag_shift=diag_shift)
    b_min_sr = -lr * E_cat
    x_min_sr, n_cg_sr, cg_timing = conjugate_gradient(
        A_min_sr,
        b_min_sr,
        tol=cg_tol,
        maxiter=cg_maxiter,
    )

    sync_if_cuda()
    delta_t0 = time.perf_counter()
    delta = D_centered_cat.T @ x_min_sr
    sync_if_cuda()
    delta_build_s = time.perf_counter() - delta_t0

    param_update_s = 0.0
    sync_if_cuda()
    sr_step_total_s = time.perf_counter() - sr_step_t0

    info = {
        "E_mean": stats["E_mean"],
        "grad_norm": torch.linalg.norm(force).item(),
        "delta_norm": torch.linalg.norm(delta).item(),
        "cg_iters": n_cg_sr,
        #"cg_iters": max(n_cg_re, n_cg_im),
        #"cg_iters_re": n_cg_re,
        #"cg_iters_im": n_cg_im,
        "force_norm": torch.linalg.norm(F).item(),
        "cov_norm": torch.linalg.norm(C).item(),
        "timing": {
            **stats["timing"],
            **cg_timing,
            "delta_build_s": delta_build_s,
            "param_update_s": param_update_s,
            "sr_step_total_s": sr_step_total_s,
        },
        "delta": delta,
    }
    return info


def exact_ground_state_energy_tfim(N, J, h, k):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.float64)

    for state in range(dim):
        spins = np.array([1 if ((state >> i) & 1) else -1 for i in range(N)], dtype=np.int8)
        e_diag = -J * np.sum(spins * np.roll(spins, -1))
        H[state, state] += e_diag

        for i in range(N):
            flipped = state ^ (1 << i)
            H[state, flipped] += -h

    evals = np.linalg.eigvalsh(H)
    return evals[0:k]


def enumerate_tfim_basis_states(N):
    dim = 2 ** N
    state_ids = torch.arange(dim, device=device, dtype=torch.int64)
    bits = ((state_ids.unsqueeze(1) >> torch.arange(N, device=device)) & 1).to(torch.float64)
    return 2.0 * bits - 1.0


def build_tfim_hamiltonian_dense(N, J, h):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.float64)

    for state in range(dim):
        spins = np.array([1 if ((state >> i) & 1) else -1 for i in range(N)], dtype=np.int8)
        H[state, state] += -J * np.sum(spins * np.roll(spins, -1))

        for i in range(N):
            flipped = state ^ (1 << i)
            H[state, flipped] += -h

    return H


@torch.no_grad()
def compute_wavefunctions_on_full_basis(model, N, field, batch_size=1024):
    basis_states = enumerate_tfim_basis_states(N)
    dim = basis_states.size(0)
    phi_blocks = []

    for start in range(0, dim, batch_size):
        batch = basis_states[start : start + batch_size]
        phi_blocks.append(model.wavefunction_matrix(batch, field).to(torch.complex128))

    phi = torch.cat(phi_blocks, dim=0)
    return basis_states, phi


def solve_generalized_hermitian(H_proj, S_proj, rcond=1e-10):
    s_eval, s_evec = np.linalg.eigh(S_proj)
    s_max = np.max(np.abs(s_eval)) if s_eval.size > 0 else 0.0
    cutoff = rcond * max(1.0, s_max)
    keep = s_eval > cutoff

    if not np.any(keep):
        raise np.linalg.LinAlgError(
            f"Overlap matrix is numerically singular: max eigenvalue {s_max:.3e}, cutoff {cutoff:.3e}."
        )

    X = s_evec[:, keep] / np.sqrt(s_eval[keep])
    H_orth = X.conj().T @ H_proj @ X
    evals, evecs_orth = np.linalg.eigh(H_orth)
    coeffs = X @ evecs_orth
    return evals, coeffs, s_eval, keep


@torch.no_grad()
def project_hamiltonian_to_model_basis(model, N, J, field, batch_size=1024, rcond=1e-10):
    basis_states, phi = compute_wavefunctions_on_full_basis(model, N, field, batch_size=batch_size)
    phi_np = phi.detach().cpu().numpy()
    H_dense = build_tfim_hamiltonian_dense(N, J, field)

    S_proj = phi_np.conj().T @ phi_np
    H_proj = phi_np.conj().T @ H_dense @ phi_np
    evals, coeffs, s_eval, keep = solve_generalized_hermitian(H_proj, S_proj, rcond=rcond)

    return {
        "field": float(field),
        "basis_states": basis_states.detach().cpu().numpy(),
        "wavefunctions": phi_np,
        "H_proj": H_proj,
        "S_proj": S_proj,
        "generalized_evals": evals,
        "generalized_evecs": coeffs,
        "overlap_evals": s_eval,
        "effective_rank": int(np.sum(keep)),
    }


def train_tfim_nqs_sr(
    N=12,
    J=1.0,
    h=1.0,
    h_values=None,
    token_size=2,
    d_model=32,
    num_heads=4,
    mlp_dim=64,
    num_layers=2,
    k=1,
    n_iter=200,
    n_chains=64,
    n_burn=100,
    n_between=5,
    n_samples=4,
    sr_lr=0.05,
    diag_shift=1e-3,
    cg_tol=1e-8,
    cg_maxiter=200,
    projection_dim_cutoff=2 ** 14,
    projection_batch_size=1024,
    projection_rcond=1e-10,
    projection_output_prefix=None,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if h_values is None:
        h_values = [h]
    h_values = [float(field) for field in h_values]
    if len(h_values) == 0:
        raise ValueError("h_values must contain at least one field value.")

    model = build_vit_model(
        N=N,
        token_size=token_size,
        d_model=d_model,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        k=k,
    )
    samplers = build_samplers(
        model=model,
        N=N,
        n_chains=n_chains,
        h_values=h_values,
        seed=seed,
    )
    exact_E0 = maybe_print_exact_energies(N=N, J=J, h_values=h_values, k=k)

    energy_hist = {field: [] for field in h_values}
    var_hist = {field: [] for field in h_values}
    projection_info = {}

    for it in range(1, n_iter + 1):
        burn = n_burn
        delta_sum = None

        for field in h_values:
            sampler = samplers[field]

            sample_t0 = time.perf_counter()
            states, psi_matrices, _ = sampler.sample(
                n_burn=burn,
                n_between=n_between,
                n_samples=n_samples,
            )
            sample_time = time.perf_counter() - sample_t0
            E_loc = local_energy_tfim_batch(model, states, psi_matrices, field, J)
            E_mean = torch.mean(E_loc)
            E_var = torch.mean(torch.abs(E_loc - E_mean) ** 2).real

            sr_t0 = time.perf_counter()
            info = sr_step_cg(
                model=model,
                states=states,
                field=field,
                E_loc=E_loc,
                lr=sr_lr,
                diag_shift=diag_shift,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
            )
            sr_time = time.perf_counter() - sr_t0
            if delta_sum is None:
                delta_sum = info["delta"].clone()
            else:
                delta_sum = delta_sum + info["delta"]

            energy_hist[field].append(E_mean.real.item())
            var_hist[field].append(E_var.item())

            if it % 2 == 0 or it == 1:
                print_iteration_log(
                    it=it,
                    field=field,
                    E_mean=E_mean,
                    E_var=E_var,
                    info=info,
                    sampler=sampler,
                    sample_time=sample_time,
                    sr_time=sr_time,
                )

        if delta_sum is not None:
            update_time = apply_joint_update(model, delta_sum)
            refresh_all_samplers(samplers, h_values)

            if it % 2 == 0 or it == 1:
                print(f"    Joint update | h_count = {len(h_values)} | TUpdAll = {update_time:.3f}s")

    hilbert_dim = 2 ** N
    if hilbert_dim <= projection_dim_cutoff:
        for field in h_values:
            info = project_hamiltonian_to_model_basis(
                model=model,
                N=N,
                J=J,
                field=field,
                batch_size=projection_batch_size,
                rcond=projection_rcond,
            )
            projection_info[field] = info
            print(f"[Projection] h = {field:.6g} | wavefunctions shape = {info['wavefunctions'].shape}")
            print(f"[Projection] h = {field:.6g} | overlap eigenvalues = {info['overlap_evals']}")
            print(f"[Projection] h = {field:.6g} | projected spectrum = {info['generalized_evals']}")
            print(f"[Projection] h = {field:.6g} | effective rank = {info['effective_rank']}")

            if projection_output_prefix is not None:
                output_path = f"{projection_output_prefix}_h_{field:.6g}.npz"
                np.savez(
                    output_path,
                    field=info["field"],
                    basis_states=info["basis_states"],
                    wavefunctions=info["wavefunctions"],
                    H_proj=info["H_proj"],
                    S_proj=info["S_proj"],
                    generalized_evals=info["generalized_evals"],
                    generalized_evecs=info["generalized_evecs"],
                    overlap_evals=info["overlap_evals"],
                    effective_rank=info["effective_rank"],
                )
                print(f"[Projection] h = {field:.6g} | saved to {output_path}")
    else:
        print(
            f"[Projection] skipped because Hilbert dimension 2^{N} = {hilbert_dim} "
            f"exceeds cutoff {projection_dim_cutoff}."
        )

    return model, energy_hist, var_hist, exact_E0, projection_info


def train_tfim_nqs_sr_combined_stats(
    N=12,
    J=1.0,
    h=1.0,
    h_values=None,
    token_size=2,
    d_model=32,
    num_heads=4,
    mlp_dim=64,
    num_layers=2,
    k=1,
    n_iter=200,
    n_chains=64,
    n_burn=100,
    n_between=5,
    n_samples=4,
    sr_lr=0.05,
    diag_shift=1e-3,
    cg_tol=1e-8,
    cg_maxiter=200,
    projection_dim_cutoff=2 ** 14,
    projection_batch_size=1024,
    projection_rcond=1e-10,
    projection_output_prefix=None,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if h_values is None:
        h_values = [h]
    h_values = [float(field) for field in h_values]
    if len(h_values) == 0:
        raise ValueError("h_values must contain at least one field value.")

    model = build_vit_model(
        N=N,
        token_size=token_size,
        d_model=d_model,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        k=k,
    )
    samplers = build_samplers(
        model=model,
        N=N,
        n_chains=n_chains,
        h_values=h_values,
        seed=seed,
    )
    exact_E0 = maybe_print_exact_energies(N=N, J=J, h_values=h_values, k=k)

    energy_hist = {field: [] for field in h_values}
    var_hist = {field: [] for field in h_values}
    projection_info = {}

    for it in range(1, n_iter + 1):
        burn = n_burn
        combined_stats = None
        sample_time_total = 0.0

        for field in h_values:
            sampler = samplers[field]

            sample_t0 = time.perf_counter()
            states, psi_matrices, _ = sampler.sample(
                n_burn=burn,
                n_between=n_between,
                n_samples=n_samples,
            )
            sample_time = time.perf_counter() - sample_t0
            sample_time_total += sample_time

            E_loc = local_energy_tfim_batch(model, states, psi_matrices, field, J)
            E_mean = torch.mean(E_loc)
            E_var = torch.mean(torch.abs(E_loc - E_mean) ** 2).real

            stats = compute_sr_statistics(model, states, field, E_loc)
            combined_stats = accumulate_combined_sr_stats(combined_stats, stats)

            energy_hist[field].append(E_mean.real.item())
            var_hist[field].append(E_var.item())

            if it % 2 == 0 or it == 1:
                print(
                    f"Iter {it:4d} | h = {field:.1f} | "
                    f"E = {E_mean.real.item(): .10f} | "
                    f"Var = {E_var.item(): .6e} | "
                    f"Acc = {sampler.acceptance_rate():.3f} | "
                    f"Tsample = {sample_time:.3f}s"
                )

        if combined_stats is not None:
            sr_t0 = time.perf_counter()
            info = sr_step_cg_from_combined_stats(
                model=model,
                combined_stats=combined_stats,
                lr=sr_lr,
                diag_shift=diag_shift,
                cg_tol=cg_tol,
                cg_maxiter=cg_maxiter,
            )
            sr_time = time.perf_counter() - sr_t0
            refresh_all_samplers(samplers, h_values)

            if it % 2 == 0 or it == 1:
                timing = info["timing"]
                print(
                    f"    Combined SR | h_count = {len(h_values)} | "
                    #f"|g| = {info['grad_norm']:.3e} | "
                    f"|delta| = {info['delta_norm']:.3e} | "
                    f"CG = {info['cg_iters']} | "
                    f"TsampleAll = {sample_time_total:.3f}s | "
                    f"TSR = {sr_time:.3f}s | "
                    f"TD = {timing['d_total_s']:.3f}s | "
                    f"TAD = {timing['d_autograd_s']:.3f}s | "
                    f"TCG = {timing['cg_total_s']:.3f}s | "
                    f"TUpd = {timing['param_update_s']:.3f}s"
                )

    hilbert_dim = 2 ** N
    if hilbert_dim <= projection_dim_cutoff:
        for field in h_values:
            info = project_hamiltonian_to_model_basis(
                model=model,
                N=N,
                J=J,
                field=field,
                batch_size=projection_batch_size,
                rcond=projection_rcond,
            )
            projection_info[field] = info
            print(f"[Projection] h = {field:.6g} | wavefunctions shape = {info['wavefunctions'].shape}")
            print(f"[Projection] h = {field:.6g} | overlap eigenvalues = {info['overlap_evals']}")
            print(f"[Projection] h = {field:.6g} | projected spectrum = {info['generalized_evals']}")
            print(f"[Projection] h = {field:.6g} | effective rank = {info['effective_rank']}")

            if projection_output_prefix is not None:
                output_path = f"{projection_output_prefix}_h_{field:.6g}.npz"
                np.savez(
                    output_path,
                    field=info["field"],
                    basis_states=info["basis_states"],
                    wavefunctions=info["wavefunctions"],
                    H_proj=info["H_proj"],
                    S_proj=info["S_proj"],
                    generalized_evals=info["generalized_evals"],
                    generalized_evecs=info["generalized_evecs"],
                    overlap_evals=info["overlap_evals"],
                    effective_rank=info["effective_rank"],
                )
                print(f"[Projection] h = {field:.6g} | saved to {output_path}")
    else:
        print(
            f"[Projection] skipped because Hilbert dimension 2^{N} = {hilbert_dim} "
            f"exceeds cutoff {projection_dim_cutoff}."
        )

    return model, energy_hist, var_hist, exact_E0, projection_info



def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train ViT-based determinant NQS-VMC with SR."
    )
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--h_values", type=float, nargs="+", default=[0.7,0.9,1.2])
    parser.add_argument("--token_size", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--n_chains", type=int, default=256)
    parser.add_argument("--n_burn", type=int, default=10)
    parser.add_argument("--n_between", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--sr_lr", type=float, default=0.05)
    parser.add_argument("--diag_shift", type=float, default=1e-4)
    parser.add_argument("--cg_tol", type=float, default=1e-8)
    parser.add_argument("--cg_maxiter", type=int, default=500)
    parser.add_argument("--combine_sr_stats", action="store_true", default=True)
    parser.add_argument("--projection_dim_cutoff", type=int, default=2 ** 14)
    parser.add_argument("--projection_batch_size", type=int, default=1024)
    parser.add_argument("--projection_rcond", type=float, default=1e-10)
    parser.add_argument("--projection_output_prefix", type=str, default=None)
    parser.add_argument(
        "--comparison_plot_path",
        type=str,
        default="projection_vs_exact_energy.png",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train_fn = train_tfim_nqs_sr_combined_stats if args.combine_sr_stats else train_tfim_nqs_sr
    model, energy_hist, var_hist, exact_E0, projection_info = train_fn(
        N=args.N,
        J=args.J,
        h=args.h,
        h_values=args.h_values,
        token_size=args.token_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        k=args.k,
        n_iter=args.n_iter,
        n_chains=args.n_chains,
        n_burn=args.n_burn,
        n_between=args.n_between,
        n_samples=args.n_samples,
        sr_lr=args.sr_lr,
        diag_shift=args.diag_shift,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        projection_dim_cutoff=args.projection_dim_cutoff,
        projection_batch_size=args.projection_batch_size,
        projection_rcond=args.projection_rcond,
        projection_output_prefix=args.projection_output_prefix,
        seed=args.seed,
    )
    plot_projection_vs_exact_energies(
        projection_info=projection_info,
        exact_E0=exact_E0,
        output_path=args.comparison_plot_path,
    )

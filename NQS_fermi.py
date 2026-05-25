
# NQS_fermi.py
# Implementation of Neural Quantum States (NQS) for Fermi lattice systems
# Based on NQS_VMC.py and FCI.py

import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import torch
import torch.nn as nn
from matplotlib.ticker import MaxNLocator
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# Cache for Hamiltonian row data:
# key = (tk, sector_idx), value = (diag_coeff, offdiag_terms)
HAMILTONIAN_ROW_CACHE = {}

# ============================================================
# Parameter input module
# ============================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural Quantum States for Fermi lattice systems")

    # Momentum grid parameters (replacing spin-site)
    parser.add_argument("--num_kx", type=int, default=3, help="Number of momentum grid points in x direction")
    parser.add_argument("--num_ky", type=int, default=6, help="Number of momentum grid points in y direction")
    parser.add_argument("--num_particles", type=int, default=None, help="Particle number; default num_k // 3")
    parser.add_argument("--tk", type=int, default=0, help="Total momentum sector")
    parser.add_argument("--all_sectors", default=False, action="store_true", help="Optimize all non-empty total momentum sectors")

    # Tight-binding parameters from FCI.py
    parser.add_argument("--epsilon", type=float, default=0.0, help="Energy offset")
    parser.add_argument("--t1", type=float, default=1.0, help="Hopping parameter t1")
    parser.add_argument("--t2", type=float, default=(1 - (1 / 2) ** (1 / 2)), help="Hopping parameter t2")
    parser.add_argument("--M", type=float, default=0.0, help="Mass term")
    parser.add_argument("--phi", type=float, default=np.pi / 4, help="Phase parameter")

    # Neural network parameters (similar to NQS_VMC.py)
    parser.add_argument("--hidden_sizes", type=int, nargs='+', default=[32, 32], help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh", "relu", "gelu"], help="Activation function")
    parser.add_argument("--k", type=int, default=1, help="Number of complex wavefunctions")

    # VMC parameters
    parser.add_argument("--n_chains", type=int, default=1024, help="Number of Monte Carlo chains")
    parser.add_argument("--n_burn", type=int, default=1, help="Number of burn-in sweeps")
    parser.add_argument("--n_between", type=int, default=1, help="Sweeps between saved samples")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of saved samples")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of VMC optimization steps")
    parser.add_argument("--lr", type=float, default=5e-2, help="SR update step size")
    parser.add_argument("--diag_shift", type=float, default=1e-3, help="SR diagonal shift")
    parser.add_argument("--cg_tol", type=float, default=1e-8, help="SR conjugate-gradient tolerance")
    parser.add_argument("--cg_maxiter", type=int, default=200, help="SR conjugate-gradient max iterations")
    parser.add_argument("--logphi_real_clip", type=float, default=30.0, help="Clip range for Re(log phi) before exponentiation")
    parser.add_argument("--max_sr_step_norm", type=float, default=10.0, help="Trust-region cap on SR update norm")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--plot_log_file", type=str, default=None, help="Parse a saved training log and plot MC energy vs iteration.")
    parser.add_argument("--plot_ed_energy", type=float, default=None, help="Exact diagonalization reference energy to overlay on the MC-energy plot.")
    parser.add_argument("--plot_output", type=str, default=None, help="Optional output image path for the parsed MC-energy plot.")
    parser.add_argument("--projection_block_size", type=int, default=4096, help="Number of sector basis states processed per block in the final projection.")
    parser.add_argument("--projection_rcond", type=float, default=1e-10, help="Relative cutoff used to regularize the overlap matrix in the generalized eigensolver.")

    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Derived parameters
num_k = args.num_kx * args.num_ky
num_sites = num_k
num_particles = args.num_particles if args.num_particles is not None else num_k // 3

# Momentum grid
kx = 2 * np.pi * np.arange(args.num_kx) / args.num_kx
ky = 2 * np.pi * np.arange(args.num_ky) / args.num_ky

# ============================================================
# Tight-binding model and basis construction (from FCI.py)
# ============================================================

# Pauli Matrices
sigma = np.zeros((4, 2, 2), complex)
sigma[0] = np.array([[1, 0], [0, 1]])
sigma[1] = np.array([[0, 1], [1, 0]])
sigma[2] = np.array([[0, -1j], [1j, 0]])
sigma[3] = np.array([[1, 0], [0, -1]])

def build_band():
    """Build tight-binding single particle band."""
    H0 = np.zeros((args.num_kx, args.num_ky, 2, 2), complex)
    E0 = np.zeros((args.num_kx, args.num_ky, 2), complex)
    psi0 = np.zeros((args.num_kx, args.num_ky, 2, 2), complex)
    for i in range(args.num_kx):
        for j in range(args.num_ky):
            h11 = 2 * args.t2 * (np.cos(kx[i]) - np.cos(ky[j])) + args.M
            h12 = args.t1 * complex(np.cos(args.phi), np.sin(args.phi)) * (1 + complex(np.cos(ky[j] - kx[i]), np.sin(ky[j] - kx[i]))) + \
                  args.t1 * complex(np.cos(args.phi), -np.sin(args.phi)) * (complex(np.cos(ky[j]), np.sin(ky[j])) + complex(np.cos(kx[i]), -np.sin(kx[i])))
            H0[i, j] = np.array([[h11, h12], [h12.conjugate(), -h11]])
            E0[i, j, :], psi0[i, j, :, :] = np.linalg.eig(H0[i, j])
            if E0[i, j, 0] > E0[i, j, 1]:
                E0[i, j, [0, 1]] = E0[i, j, [1, 0]]
                psi0[i, j, :, [0, 1]] = psi0[i, j, :, [1, 0]]
    return E0, psi0

def build_basis():
    """Build interacting basis with binary indices."""
    index = np.array(list(itertools.combinations(range(num_k), num_particles)))
    base_number = index.shape[0]
    index_tk = np.zeros((num_k, base_number), int)
    number_tk = np.zeros(num_k, int)
    index_b = np.zeros(base_number, dtype=np.int64)
    for i in range(base_number):
        sumx = 0
        sumy = 0
        for j in range(num_particles):
            sumx += index[i, j] % args.num_kx
            sumy += index[i, j] // args.num_kx
            index_b[i] |= (1 << index[i, j])
        total_sum = (sumx % args.num_kx) + (sumy % args.num_ky) * args.num_kx
        index_tk[total_sum, number_tk[total_sum]] = i
        number_tk[total_sum] += 1
    return index, index_b, base_number, index_tk, number_tk

def count_inversions(arr):
    """Compute inversion number for fermion state."""
    n = len(arr)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inversions += 1
    return inversions

def count_ones(n):
    """Count occupation number (# of 1 in binary representations) of state n."""
    count = 0
    positions = []
    index = 0
    while n:
        if n & 1:
            count += 1
            positions.append(index)
        n >>= 1
        index += 1
    return count, positions

def build_interacting_hamiltonian(tk, psi0, E0, index, index_b, base_number, index_tk, number_tk):
    """Build interacting hamiltonian in occupation basis."""
    print("building interacting hamiltonian")
    Hi = np.zeros((number_tk[tk], number_tk[tk]), complex)
    invindex = np.zeros((number_tk[tk], num_k), int)
    for i in range(number_tk[tk]):
        for j in range(num_k):
            invindex[i, j] = -1
    for i in range(number_tk[tk]):
        for j in range(num_particles):
            invindex[i, index[index_tk[tk, i], j]] = j
    for i in range(number_tk[tk]):
        ki_b = index_b[index_tk[tk, i]].copy()
        ki = index[index_tk[tk, i]].copy()
        for p in range(num_particles):
            Hi[i, i] += args.epsilon * E0[ki[p] % args.num_kx, ki[p] // args.num_kx, 0]

        # ---scattering interaction---
        for j in range(i):
            kj_b = index_b[index_tk[tk, j]].copy()
            kj = index[index_tk[tk, j]].copy()
            dif_b = ki_b ^ kj_b
            count, position = count_ones(int(dif_b))
            if count != 4:
                continue
            pq = []
            mn = []
            for l in range(4):
                if invindex[j, position[l]] != -1:
                    pq.append(invindex[j, position[l]])
                if invindex[i, position[l]] != -1:
                    mn.append(invindex[i, position[l]])
            for m_order in [(0, 1), (1, 0)]:
                for p_order in [(0, 1), (1, 0)]:
                    k2 = kj[pq[p_order[0]]]
                    k4 = kj[pq[p_order[1]]]
                    k1 = ki[mn[m_order[0]]]
                    k3 = ki[mn[m_order[1]]]

                    k2y = k2 // args.num_kx
                    k2x = k2 % args.num_kx
                    k4y = k4 // args.num_kx
                    k4x = k4 % args.num_kx
                    k1y = k1 // args.num_kx
                    k1x = k1 % args.num_kx
                    k3y = k3 // args.num_kx
                    k3x = k3 % args.num_kx

                    V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
                        (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
                    index_temp = kj.copy()
                    index_temp[pq[p_order[0]]] = k1
                    index_temp[pq[p_order[1]]] = k3
                    sign = (-1) ** (count_inversions(index_temp) - count_inversions(ki))
                    Hi[i, j] += sign * V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

            Hi[j, i] = Hi[i, j].conjugate()

        # ---self interaction---
        for p in range(num_particles):
            for q in range(num_particles):
                k2 = ki[p]
                k4 = ki[q]
                k1 = ki[q]
                k3 = ki[p]

                k2y = k2 // args.num_kx
                k2x = k2 % args.num_kx
                k4y = k4 // args.num_kx
                k4x = k4 % args.num_kx
                k1y = k1 // args.num_kx
                k1x = k1 % args.num_kx
                k3y = k3 // args.num_kx
                k3x = k3 % args.num_kx

                V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
                    (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
                Hi[i, i] += -V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

    return Hi

# ============================================================
# Utilities for fermion occupation configurations
# ============================================================
def bitstrings_to_occupations(bitstrings):
    """Convert basis-state bitstrings to occupation vectors in {0, 1}."""
    bitstrings = np.asarray(bitstrings, dtype=np.int64)
    occ = ((bitstrings[:, None] >> np.arange(num_sites, dtype=np.int64)) & 1).astype(np.float64)
    return occ

def sector_basis_bitstrings(index_b, index_tk, number_tk, tk):
    """Return binary basis labels for one total-momentum sector."""
    states = np.zeros(number_tk[tk], dtype=np.int64)
    for i in range(number_tk[tk]):
        states[i] = index_b[index_tk[tk, i]]
    return states

def build_sector_neighbors(Hi, atol=1e-12):
    """Cache nonzero Hamiltonian matrix elements for fast local-energy evaluation."""
    neighbors = []
    for i in range(Hi.shape[0]):
        cols = np.where(np.abs(Hi[i]) > atol)[0]
        neighbors.append((cols.astype(np.int64), Hi[i, cols].astype(np.complex128)))
    return neighbors

def build_invindex_sector(index, index_tk, number_tk, tk):
    """Map occupied momentum labels to their positions inside each sector basis state."""
    invindex = np.full((number_tk[tk], num_k), -1, dtype=int)
    for i in range(number_tk[tk]):
        for j in range(num_particles):
            invindex[i, index[index_tk[tk, i], j]] = j
    return invindex

# Import ComplexMLP from NQS_VMC.py
class ComplexMLP(nn.Module):
    def __init__(self, N, hidden_sizes=(64, 64), activation="tanh", k=1):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive.")

        acts = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
        }
        act = acts[activation]

        layers = []
        in_dim = N
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            in_dim = h

        self.N = N
        self.backbone = nn.Sequential(*layers)
        self.k = k
        self.real_head = nn.Linear(in_dim, k)
        self.imag_head = nn.Linear(in_dim, k)

    def forward(self, s):
        # s: (batch, N), entries are fermion occupations encoded as 0/1
        if s.dim() != 2:
            raise ValueError("s must have shape (batch, N).")
        if s.size(1) != self.N:
            raise ValueError(f"Expected s.shape[1] == {self.N}, got {s.size(1)}.")
        x = self.backbone(s)
        re = self.real_head(x)
        im = self.imag_head(x)
        return re + 1j * im

    def log_wavefunction_matrix(self, states):
        """Return the matrix LogPhi_{a,i} = log phi_i(s_a)."""
        if states.dim() != 2:
            raise ValueError("states must have shape (batch, N).")
        if states.size(1) != self.N:
            raise ValueError(f"Expected states.shape[1] == {self.N}, got {states.size(1)}.")
        return self.forward(states)

    def wavefunction_matrix(self, states):
        """Return the matrix Phi_{a,i} = phi_i(s_a)."""
        log_phi = self.log_wavefunction_matrix(states)
        re = torch.clamp(torch.real(log_phi), min=-args.logphi_real_clip, max=args.logphi_real_clip)
        im = torch.imag(log_phi)
        return torch.exp(re).to(torch.complex128) * torch.exp(1j * im)

    def log_psi(self, states):
        """Scalar log-wavefunction used by the fermion occupation ansatz."""
        out = self.forward(states)
        if out.size(-1) != 1:
            raise ValueError("log_psi requires k=1. Use wavefunction_matrix for k>1.")
        return out[:, 0]

# Minimal fermionic NQS wrapper
class FermiNQS(nn.Module):
    def __init__(self, num_sites, num_particles, hidden_sizes=(64, 64), activation="tanh", k=1):
        super().__init__()
        self.num_sites = num_sites
        self.num_particles = num_particles
        self.k = k
        self.ansatz = ComplexMLP(num_sites, hidden_sizes=hidden_sizes, activation=activation, k=k)

    def forward(self, configurations):
        return self.ansatz.forward(configurations)

    def log_wavefunction_matrix(self, configurations):
        if configurations.dim() != 2:
            raise ValueError("configurations must have shape (batch, num_sites).")
        if configurations.size(1) != self.num_sites:
            raise ValueError(
                f"Expected configurations.shape[1] == {self.num_sites}, got {configurations.size(1)}."
            )
        if torch.any((configurations != 0) & (configurations != 1)):
            raise ValueError("FermiNQS expects occupation-number inputs with entries 0 or 1.")
        return self.ansatz.log_wavefunction_matrix(configurations)

    def wavefunction_matrix(self, configurations):
        return torch.exp(self.log_wavefunction_matrix(configurations))

    def log_psi(self, configurations):
        if self.k != 1:
            raise ValueError("log_psi is only defined for k=1. Use wavefunction_matrix for k>1.")
        return self.ansatz.log_psi(configurations)

    def psi(self, configurations):
        if self.k != 1:
            raise ValueError("psi is only defined for k=1. Use wavefunction_matrix for k>1.")
        return torch.exp(self.log_psi(configurations))


def compute_wavefunction_matrices(model, states):
    """Build wavefunction matrices for a batch of chains.

    Args:
        states: Tensor of shape (n_chains, k, N).

    Returns:
        Complex tensor of shape (n_chains, k, k).
    """
    if states.dim() != 3:
        raise ValueError("states must have shape (n_chains, k, N).")
    n_chains, k, n_sites = states.shape
    if k != model.k:
        raise ValueError(f"Expected states.shape[1] == model.k == {model.k}, got {k}.")
    if n_sites != model.num_sites:
        raise ValueError(f"Expected states.shape[2] == model.num_sites == {model.num_sites}, got {n_sites}.")
    flat_states = states.reshape(n_chains * k, n_sites)
    phi_flat = model.wavefunction_matrix(flat_states)
    return phi_flat.reshape(n_chains, k, model.k)


def compute_logdet_and_logprob(psi_matrices):
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

# ============================================================
# Metropolis sampler for fermion occupation configurations
# Probability weight is |psi(n)|^2
# The implementation follows the structure of NQS_VMC.py, but the
# Markov chain lives directly inside one fixed total-momentum sector.
# ============================================================
class MetropolisSampler:
    def __init__(self, model, N, index_b, index_tk, number_tk, tk, n_chains, seed=1234):
        self.model = model
        self.N = N
        self.k = model.k
        self.tk = tk
        self.n_chains = n_chains
        self.device = next(model.parameters()).device
        self.sector_size = int(number_tk[tk])
        if self.sector_size <= 0:
            raise ValueError(f"Momentum sector tk={tk} is empty.")
        if self.sector_size < self.k:
            raise ValueError(
                f"Momentum sector tk={tk} has size {self.sector_size}, smaller than k={self.k}."
            )

        self.sector_positions = np.arange(self.sector_size, dtype=np.int64)
        self.sector_bitstrings = sector_basis_bitstrings(index_b, index_tk, number_tk, tk)

        gen_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=gen_device)
        self.generator.manual_seed(seed)

        self.positions = torch.empty((n_chains, self.k), dtype=torch.int64, device=self.device)
        for chain_id in range(n_chains):
            self.positions[chain_id] = torch.randperm(
                self.sector_size,
                generator=self.generator,
                device=self.device,
            )[: self.k]
        states = torch.as_tensor(
            bitstrings_to_occupations(self.sector_bitstrings[self.positions.cpu().numpy().reshape(-1)]),
            dtype=torch.float64,
            device=self.device,
        ).reshape(n_chains, self.k, N)

        self.states = states
        self.refresh_cache()
        self.n_accepted = 0
        self.n_proposed = 0

    @torch.no_grad()
    def refresh_cache(self):
        """Rebuild cached wavefunction data for the current states and current model parameters."""
        self.psi_matrices = compute_wavefunction_matrices(self.model, self.states)
        self.psi_inv = torch.linalg.inv(self.psi_matrices)
        self.logdet, self.logprob = compute_logdet_and_logprob(self.psi_matrices)

    @torch.no_grad()
    def sweep(self):
        """
        One sweep = N proposals per chain.
        Each proposal replaces one row by another basis state in the same
        total-momentum sector. The chain weight is |det Phi|^2.
        """
        n_chains, k, N = self.states.shape
        chain_ids = torch.arange(n_chains, device=self.device)

        for _ in range(N):
            if self.sector_size == 1:
                self.n_proposed += n_chains
                continue

            config_indices = torch.randint(
                0, k, (n_chains,), generator=self.generator, device=self.device
            )
            proposed_positions = self.positions.clone()

            for chain_id in range(n_chains):
                row_id = int(config_indices[chain_id].item())
                current_position = int(self.positions[chain_id, row_id].item())
                occupied_positions = set(self.positions[chain_id].tolist())
                occupied_positions.remove(current_position)

                candidate = int(
                    torch.randint(0, self.sector_size, (1,), generator=self.generator, device=self.device).item()
                )
                while candidate == current_position or candidate in occupied_positions:
                    candidate = int(
                        torch.randint(0, self.sector_size, (1,), generator=self.generator, device=self.device).item()
                    )
                proposed_positions[chain_id, row_id] = candidate

            proposed_row_positions = proposed_positions[chain_ids, config_indices].cpu().numpy()
            proposed_rows = torch.as_tensor(
                bitstrings_to_occupations(self.sector_bitstrings[proposed_row_positions]),
                dtype=torch.float64,
                device=self.device,
            )

            new_row_values = self.model.wavefunction_matrix(proposed_rows)
            det_ratio = row_replacement_det_ratio(new_row_values, self.psi_inv, config_indices)
            logprob_new = self.logprob + 2.0 * torch.log(torch.abs(det_ratio))
            logdet_new = self.logdet + torch.log(det_ratio)

            log_acc = logprob_new - self.logprob
            rand_log = torch.log(
                torch.rand(n_chains, generator=self.generator, device=self.device)
            )
            accept = rand_log < torch.minimum(log_acc, torch.zeros_like(log_acc))

            if accept.any():
                accept_ids = chain_ids[accept]
                accept_rows = config_indices[accept]
                self.states[accept_ids, accept_rows] = proposed_rows[accept]
                self.psi_matrices[accept_ids, accept_rows] = new_row_values[accept]
                self.logdet[accept_ids] = logdet_new[accept]
                self.logprob[accept_ids] = logprob_new[accept_ids]
                self.positions[accept_ids, accept_rows] = proposed_positions[accept_ids, accept_rows]
                self.psi_inv[accept_ids] = torch.linalg.inv(self.psi_matrices[accept_ids])

            self.n_accepted += accept.sum().item()
            self.n_proposed += n_chains

    @torch.no_grad()
    def sample(self, n_burn, n_between, n_samples):
        """
        Returns:
            states  : (n_samples * n_chains, k, N)
            positions : (n_samples * n_chains, k)
            psi_matrices : (n_samples * n_chains, k, k)
            logprob : (n_samples * n_chains,)
        """
        for _ in range(n_burn):
            self.sweep()

        states_list = []
        positions_list = []
        psi_matrices_list = []
        logprob_list = []

        for _ in range(n_samples):
            for _ in range(n_between):
                self.sweep()
            states_list.append(self.states.clone())
            positions_list.append(self.positions.clone())
            psi_matrices_list.append(self.psi_matrices.clone())
            logprob_list.append(self.logprob.clone())

        return (
            torch.cat(states_list, dim=0),
            torch.cat(positions_list, dim=0),
            torch.cat(psi_matrices_list, dim=0),
            torch.cat(logprob_list, dim=0),
        )

    def acceptance_rate(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

def hamiltonian_row_terms(sector_idx, psi0, E0, index, index_b, index_tk, number_tk, tk, invindex):
    """Return diagonal coefficient and off-diagonal connected terms for one basis state."""
    cache_key = (int(tk), int(sector_idx))
    cached = HAMILTONIAN_ROW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ki = index[index_tk[tk, sector_idx]].copy()
    ki_b = index_b[index_tk[tk, sector_idx]].copy()
    diag_coeff = 0.0j
    offdiag_terms = []

    for j in range(number_tk[tk]):
        if j == sector_idx:
            continue
        kj = index[index_tk[tk, j]].copy()
        kj_b = index_b[index_tk[tk, j]].copy()
        dif_b = ki_b ^ kj_b
        count, position = count_ones(int(dif_b))
        if count != 4:
            continue

        pq = []
        mn = []
        for l in range(4):
            if invindex[j, position[l]] != -1:
                pq.append(invindex[j, position[l]])
            if invindex[sector_idx, position[l]] != -1:
                mn.append(invindex[sector_idx, position[l]])
        if len(pq) != 2 or len(mn) != 2:
            continue

        hij = 0.0j
        k2y = kj[pq[0]] // args.num_kx
        k2x = kj[pq[0]] % args.num_kx
        k4y = kj[pq[1]] // args.num_kx
        k4x = kj[pq[1]] % args.num_kx
        k1y = ki[mn[0]] // args.num_kx
        k3y = ki[mn[1]] // args.num_kx
        k1x = ki[mn[0]] % args.num_kx
        k3x = ki[mn[1]] % args.num_kx
        V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
            (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
        index_temp = kj.copy()
        index_temp[pq[0]] = k1y * args.num_kx + k1x
        index_temp[pq[1]] = k3y * args.num_kx + k3x
        sign = (-1) ** (count_inversions(index_temp) - count_inversions(ki))
        hij += sign * V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

        k2y = kj[pq[1]] // args.num_kx
        k2x = kj[pq[1]] % args.num_kx
        k4y = kj[pq[0]] // args.num_kx
        k4x = kj[pq[0]] % args.num_kx
        k1y = ki[mn[0]] // args.num_kx
        k3y = ki[mn[1]] // args.num_kx
        k1x = ki[mn[0]] % args.num_kx
        k3x = ki[mn[1]] % args.num_kx
        V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
            (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
        index_temp = kj.copy()
        index_temp[pq[1]] = k1y * args.num_kx + k1x
        index_temp[pq[0]] = k3y * args.num_kx + k3x
        sign = (-1) ** (count_inversions(index_temp) - count_inversions(ki))
        hij += sign * V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

        k2y = kj[pq[0]] // args.num_kx
        k2x = kj[pq[0]] % args.num_kx
        k4y = kj[pq[1]] // args.num_kx
        k4x = kj[pq[1]] % args.num_kx
        k1y = ki[mn[1]] // args.num_kx
        k3y = ki[mn[0]] // args.num_kx
        k1x = ki[mn[1]] % args.num_kx
        k3x = ki[mn[0]] % args.num_kx
        V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
            (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
        index_temp = kj.copy()
        index_temp[pq[0]] = k1y * args.num_kx + k1x
        index_temp[pq[1]] = k3y * args.num_kx + k3x
        sign = (-1) ** (count_inversions(index_temp) - count_inversions(ki))
        hij += sign * V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

        k2y = kj[pq[1]] // args.num_kx
        k2x = kj[pq[1]] % args.num_kx
        k4y = kj[pq[0]] // args.num_kx
        k4x = kj[pq[0]] % args.num_kx
        k1y = ki[mn[1]] // args.num_kx
        k3y = ki[mn[0]] // args.num_kx
        k1x = ki[mn[1]] % args.num_kx
        k3x = ki[mn[0]] % args.num_kx
        V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
            (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
        index_temp = kj.copy()
        index_temp[pq[1]] = k1y * args.num_kx + k1x
        index_temp[pq[0]] = k3y * args.num_kx + k3x
        sign = (-1) ** (count_inversions(index_temp) - count_inversions(ki))
        hij += sign * V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

        if abs(hij) > 0.0:
            offdiag_terms.append((j, hij))

    for p in range(num_particles):
        for q in range(num_particles):
            k2y = ki[p] // args.num_kx
            k2x = ki[p] % args.num_kx
            k4y = ki[q] // args.num_kx
            k4x = ki[q] % args.num_kx
            m = ki[q]
            n = ki[p]
            k1y = m // args.num_kx
            k3y = n // args.num_kx
            k1x = m % args.num_kx
            k3x = n % args.num_kx

            V = (1 + complex(np.cos(ky[k4y] - ky[k3y]), np.sin(ky[k4y] - ky[k3y]))) * \
                (1 + complex(np.cos(kx[k4x] - kx[k3x]), -np.sin(kx[k4x] - kx[k3x])))
            diag_coeff += -V * psi0[k1x, k1y, 0, 0].conjugate() * psi0[k2x, k2y, 0, 0] * psi0[k3x, k3y, 1, 0].conjugate() * psi0[k4x, k4y, 1, 0] / num_k

    result = (diag_coeff, tuple(offdiag_terms))
    HAMILTONIAN_ROW_CACHE[cache_key] = result
    return result


# (H psi) and local energy for the fermion NQS
@torch.no_grad()
def h_psi_batch(model, states, sector_positions, psi_matrices, psi0, E0, index, index_b, index_tk, number_tk, tk, invindex=None):
    """Build (H Psi) for the fermion determinant ansatz.

    Args:
        states: Tensor of shape (M, k, N).
        sector_positions: integer array of shape (M, k), basis indices inside the fixed sector.
        psi_matrices: Tensor of shape (M, k, k), with Psi[m, a, i] = phi_i(s_a).
    """
    if states.dim() != 3:
        raise ValueError("states must have shape (M, k, N).")
    if psi_matrices.dim() != 3:
        raise ValueError("psi_matrices must have shape (M, k, k).")

    M, k, N = states.shape
    if k != model.k:
        raise ValueError(f"Expected states.shape[1] == model.k == {model.k}, got {k}.")
    if psi_matrices.shape != (M, k, k):
        raise ValueError(f"Expected psi_matrices.shape == {(M, k, k)}, got {tuple(psi_matrices.shape)}.")

    sector_positions = np.asarray(sector_positions, dtype=np.int64)
    if sector_positions.shape != (M, k):
        raise ValueError(f"Expected sector_positions.shape == {(M, k)}, got {sector_positions.shape}.")
    if invindex is None:
        invindex = build_invindex_sector(index, index_tk, number_tk, tk)

    hp = torch.zeros((M, k, k), dtype=torch.complex128, device=states.device)
    neighbor_row_cache = {}

    # Reuse rows already evaluated in psi_matrices before doing any extra forward passes.
    for m in range(M):
        for a in range(k):
            neighbor_row_cache[int(sector_positions[m, a])] = psi_matrices[m, a].clone()

    for m in range(M):
        for a in range(k):
            sector_idx = int(sector_positions[m, a])
            diag_coeff, offdiag_terms = hamiltonian_row_terms(
                sector_idx, psi0, E0, index, index_b, index_tk, number_tk, tk, invindex
            )
            hp[m, a] += diag_coeff * psi_matrices[m, a]

            for neighbor_idx, hij in offdiag_terms:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx in neighbor_row_cache:
                    neighbor_row = neighbor_row_cache[neighbor_idx]
                else:
                    neighbor_state = torch.as_tensor(
                        bitstrings_to_occupations(np.array([index_b[index_tk[tk, neighbor_idx]]], dtype=np.int64)),
                        dtype=torch.float64,
                        device=states.device,
                    )
                    neighbor_row = model.wavefunction_matrix(neighbor_state)[0]
                    neighbor_row_cache[neighbor_idx] = neighbor_row.clone()
                hp[m, a] += hij * neighbor_row

    return hp


@torch.no_grad()
def E_loc(model, states, sector_positions, psi_matrices, psi0, E0, index, index_b, index_tk, number_tk, tk, invindex=None):
    """Compute E_loc = Tr(Psi^{-1} H Psi) on a batch of sampled chains."""
    hp = h_psi_batch(
        model,
        states,
        sector_positions,
        psi_matrices,
        psi0,
        E0,
        index,
        index_b,
        index_tk,
        number_tk,
        tk,
        invindex=invindex,
    )
    psi_inv = torch.linalg.inv(psi_matrices)
    local_matrix = torch.matmul(psi_inv.to(torch.complex128), hp)
    return torch.einsum("mii->m", local_matrix)


def get_param_vector(model):
    return parameters_to_vector([p for p in model.parameters() if p.requires_grad])


def set_param_vector(model, vec):
    vector_to_parameters(vec, [p for p in model.parameters() if p.requires_grad])


def compute_D_matrix(model, states):
    """Compute D_l = Tr(Psi^{-1} d_l Psi) for each sampled chain."""
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

    for m in range(M):
        model.zero_grad(set_to_none=True)
        log_phi_matrix = model.log_wavefunction_matrix(states[m])
        phi_matrix = torch.exp(log_phi_matrix)
        phi_inv = torch.linalg.inv(phi_matrix).detach()
        d_scalar = torch.einsum("ia,ai->", phi_inv, phi_matrix)

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

        g_re = parameters_to_vector(grads_re)
        g_im = parameters_to_vector(grads_im)
        D[m] = g_re.to(torch.complex128) + 1j * g_im.to(torch.complex128)

    return D


def compute_sr_statistics(model, states, E_loc_values):
    if E_loc_values.dim() != 1:
        raise ValueError("E_loc_values must have shape (M,).")
    if E_loc_values.size(0) != states.size(0):
        raise ValueError(
            f"Expected E_loc_values.shape[0] == states.shape[0] == {states.size(0)}, got {E_loc_values.size(0)}."
        )

    D = compute_D_matrix(model, states)
    D_mean = torch.mean(D, dim=0, keepdim=True)
    D_centered = D - D_mean
    E_mean = torch.mean(E_loc_values)
    E_centered = E_loc_values - E_mean
    C = (D_centered.conj().T @ D_centered) / states.size(0)
    F = torch.mean(D_centered.conj() * E_centered.unsqueeze(1), dim=0)

    return {
        "D": D,
        "D_mean": D_mean,
        "D_centered": D_centered,
        "E_mean": E_mean,
        "E_centered": E_centered,
        "C": C,
        "F": F,
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


def conjugate_gradient(A, b, x0=None, tol=1e-10, maxiter=200):
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()

    r = b - A.matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)

    if torch.sqrt(rs_old) < tol:
        return x, 0

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
            return x, it
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, it


def sr_step_cg(model, states, E_loc_values, diag_shift=1e-3, cg_tol=1e-8, cg_maxiter=200):
    stats = compute_sr_statistics(model, states, E_loc_values)
    force = torch.real(stats["F"])
    A = SRLinearOperator(stats["D_centered"], diag_shift=diag_shift)
    b = -force
    delta, n_cg = conjugate_gradient(A, b, tol=cg_tol, maxiter=cg_maxiter)

    return {
        "E_mean": stats["E_mean"],
        "grad_norm": torch.linalg.norm(force).item(),
        "delta_norm": torch.linalg.norm(delta).item(),
        "cg_iters": n_cg,
        "force_norm": torch.linalg.norm(stats["F"]).item(),
        "cov_norm": torch.linalg.norm(stats["C"]).item(),
        "delta": delta,
    }

@torch.no_grad()
def sample_and_evaluate(model, sampler, psi0, E0, index, index_b, index_tk, number_tk, tk, invindex, n_burn, n_between, n_samples):
    t_sample_start = time.perf_counter()
    states, positions, psi_matrices, logprob = sampler.sample(n_burn, n_between, n_samples)
    t_sample = time.perf_counter() - t_sample_start

    t_eloc_start = time.perf_counter()
    e_loc = E_loc(
        model,
        states,
        positions.cpu().numpy(),
        psi_matrices,
        psi0,
        E0,
        index,
        index_b,
        index_tk,
        number_tk,
        tk,
        invindex=invindex,
    )
    t_eloc = time.perf_counter() - t_eloc_start
    return {
        "states": states,
        "positions": positions,
        "psi_matrices": psi_matrices,
        "logprob": logprob,
        "E_loc": e_loc,
        "E_mean": torch.mean(e_loc),
        "E_real_mean": torch.real(e_loc).mean().item(),
        "E_imag_mean": torch.imag(e_loc).mean().item(),
        "E_real_std": torch.real(e_loc).std(unbiased=False).item(),
        "acceptance": sampler.acceptance_rate(),
        "timing": {
            "sample": t_sample,
            "eloc": t_eloc,
        },
    }


def optimize_vmc(model, sampler, psi0, E0, index, index_b, index_tk, number_tk, tk, invindex, n_steps, lr, n_burn, n_between, n_samples):
    history = []

    for step in range(1, n_steps + 1):
        t_step_start = time.perf_counter()
        batch_before = sample_and_evaluate(
            model,
            sampler,
            psi0,
            E0,
            index,
            index_b,
            index_tk,
            number_tk,
            tk,
            invindex,
            n_burn,
            n_between,
            n_samples,
        )

        t_sr_start = time.perf_counter()
        sr_info = sr_step_cg(
            model,
            batch_before["states"],
            batch_before["E_loc"],
            diag_shift=args.diag_shift,
            cg_tol=args.cg_tol,
            cg_maxiter=args.cg_maxiter,
        )
        t_sr = time.perf_counter() - t_sr_start

        t_update_start = time.perf_counter()
        theta_old = get_param_vector(model)
        delta = sr_info["delta"]
        delta_norm = torch.linalg.norm(delta).item()
        trust_region_scale = 1.0
        if delta_norm > args.max_sr_step_norm:
            trust_region_scale = args.max_sr_step_norm / delta_norm
            delta = delta * trust_region_scale

        step_scale = lr
        backtracks = 0
        update_ok = False
        last_error = None

        applied_update_norm = None
        effective_lr = None

        for _ in range(12):
            theta_new = theta_old + step_scale * delta
            set_param_vector(model, theta_new)
            try:
                sampler.refresh_cache()
                if (
                    torch.isfinite(torch.real(sampler.psi_matrices)).all()
                    and torch.isfinite(torch.imag(sampler.psi_matrices)).all()
                    and torch.isfinite(torch.real(sampler.logprob)).all()
                ):
                    applied_update_norm = torch.linalg.norm(theta_new - theta_old).item()
                    effective_lr = step_scale * trust_region_scale
                    update_ok = True
                    break
                last_error = RuntimeError("Non-finite values found in sampler cache after parameter update.")
            except Exception as exc:
                last_error = exc

            backtracks += 1
            step_scale *= 0.5

        if not update_ok:
            set_param_vector(model, theta_old)
            sampler.refresh_cache()
            raise RuntimeError(
                f"SR update failed after {backtracks} backtracking attempts at step {step}. "
                f"Last error: {last_error}"
            )
        t_update = time.perf_counter() - t_update_start

        t_eval_after_start = time.perf_counter()
        batch_after = sample_and_evaluate(
            model,
            sampler,
            psi0,
            E0,
            index,
            index_b,
            index_tk,
            number_tk,
            tk,
            invindex,
            0,
            0,
            1,
        )
        t_eval_after = time.perf_counter() - t_eval_after_start
        t_step = time.perf_counter() - t_step_start

        history.append(
            {
                "step": step,
                "energy_real_before": batch_before["E_real_mean"],
                "energy_imag_before": batch_before["E_imag_mean"],
                "energy_std_before": batch_before["E_real_std"],
                "energy_real_after": batch_after["E_real_mean"],
                "energy_imag_after": batch_after["E_imag_mean"],
                "energy_std_after": batch_after["E_real_std"],
                "acceptance_before": batch_before["acceptance"],
                "acceptance_after": batch_after["acceptance"],
                "delta_norm": sr_info["delta_norm"],
                "applied_update_norm": applied_update_norm,
                "grad_norm": sr_info["grad_norm"],
                "cg_iters": sr_info["cg_iters"],
                "trust_region_scale": trust_region_scale,
                "applied_lr": step_scale,
                "effective_lr": effective_lr,
                "backtracks": backtracks,
                "time_sample": batch_before["timing"]["sample"],
                "time_eloc": batch_before["timing"]["eloc"],
                "time_sr": t_sr,
                "time_update": t_update,
                "time_eval_after": t_eval_after,
                "time_total": t_step,
            }
        )
        if step%100==0:
            print(
            f"step={step:4d} "
            f"E_before={batch_before['E_real_mean']:.10f} + {batch_before['E_imag_mean']:.3e}i "
            f"E_after={batch_after['E_real_mean']:.10f} + {batch_after['E_imag_mean']:.3e}i "
            f"std_before={batch_before['E_real_std']:.10f} "
            f"std_after={batch_after['E_real_std']:.10f} "
            f"acc_before={batch_before['acceptance']:.4f} "
            f"acc_after={batch_after['acceptance']:.4f} "
            f"|delta|={sr_info['delta_norm']:.3e} "
            f"|dtheta|={applied_update_norm:.3e} "
            f"cg={sr_info['cg_iters']} "
            f"tr={trust_region_scale:.3e} "
            f"lr={step_scale:.3e} "
            f"lr_eff={effective_lr:.3e} "
            f"bt={backtracks}"
            )
            print(
            f"  timing: sample={batch_before['timing']['sample']:.3f}s "
            f"eloc={batch_before['timing']['eloc']:.3f}s "
            f"sr={t_sr:.3f}s "
            f"update={t_update:.3f}s "
            f"eval_after={t_eval_after:.3f}s "
            f"total={t_step:.3f}s"
            )

    return history


def plot_sector_energies(sector_results):
    """Plot variational energies versus total momentum sector, following FCI.py style."""
    x = []
    y = []
    for result in sector_results:
        tk = result["tk"]
        x.append(tk)
        y.append(result["initial_stats"]["E_real_mean"])
        for item in result["history"]:
            x.append(tk)
            y.append(item["energy_real"])

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color="red", marker="x", label="NQS Energy")
    plt.title("Low-energy spectrum")
    plt.xlabel("Total Momentum")
    plt.ylabel("E")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()


def parse_mc_energy_log(log_path):
    initial_pattern = re.compile(
        r"initial E=(?P<E_real>[-+0-9.eE]+)\s+\+\s+(?P<E_imag>[-+0-9.eE]+)i\s+std=(?P<std>[-+0-9.eE]+)"
    )
    step_pattern = re.compile(
        r"step=\s*(?P<step>\d+)\s+"
        r"E_before=(?P<E_before_real>[-+0-9.eE]+)\s+\+\s+(?P<E_before_imag>[-+0-9.eE]+)i\s+"
        r"E_after=(?P<E_after_real>[-+0-9.eE]+)\s+\+\s+(?P<E_after_imag>[-+0-9.eE]+)i\s+"
        r"std_before=(?P<std_before>[-+0-9.eE]+)\s+"
        r"std_after=(?P<std_after>[-+0-9.eE]+)\s+"
        r"acc_before=(?P<acc_before>[-+0-9.eE]+)\s+"
        r"acc_after=(?P<acc_after>[-+0-9.eE]+)"
    )

    initial = None
    steps = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if initial is None:
                match = initial_pattern.search(line)
                if match:
                    initial = {
                        "E_real": float(match.group("E_real")),
                        "E_imag": float(match.group("E_imag")),
                        "std": float(match.group("std")),
                    }
                    continue

            match = step_pattern.search(line)
            if match:
                steps.append(
                    {
                        "step": int(match.group("step")),
                        "E_before_real": float(match.group("E_before_real")),
                        "E_before_imag": float(match.group("E_before_imag")),
                        "E_after_real": float(match.group("E_after_real")),
                        "E_after_imag": float(match.group("E_after_imag")),
                        "std_before": float(match.group("std_before")),
                        "std_after": float(match.group("std_after")),
                        "acc_before": float(match.group("acc_before")),
                        "acc_after": float(match.group("acc_after")),
                    }
                )

    if initial is None:
        raise ValueError(f"Could not find 'initial E=...' line in log file: {log_path}")
    if not steps:
        raise ValueError(f"Could not find any optimization step lines in log file: {log_path}")

    return initial, steps


def plot_mc_energy_from_log(log_path, ed_energy=None, output_path=None):
    initial, steps = parse_mc_energy_log(log_path)

    x = [0] + [item["step"] for item in steps]
    y_after = [initial["E_real"]] + [item["E_after_real"] for item in steps]
    y_before = [initial["E_real"]] + [item["E_before_real"] for item in steps]
    yerr_after = [initial["std"]] + [item["std_after"] for item in steps]
    acc_after = [steps[0]["acc_before"]] + [item["acc_after"] for item in steps]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, y_after, color="tab:blue", linewidth=1.8, label="MC Energy (after SR step)")
    ax1.plot(x, y_before, color="tab:orange", linewidth=1.0, alpha=0.6, linestyle="--", label="MC Energy (before SR step)")
    ax1.fill_between(
        x,
        np.array(y_after) - np.array(yerr_after),
        np.array(y_after) + np.array(yerr_after),
        color="tab:blue",
        alpha=0.15,
        label="MC std",
    )

    if ed_energy is not None:
        ax1.axhline(ed_energy, color="red", linestyle="-.", linewidth=1.5, label=f"ED Exact = {ed_energy:.10f}")

    ax1.set_title("MC Energy and Acceptance vs Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Energy")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(x, acc_after, color="tab:green", linewidth=1.2, alpha=0.85, label="Acceptance")
    ax2.set_ylabel("Acceptance")
    ax2.set_ylim(0.0, 1.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"saved MC-energy plot to: {output_path}")
    else:
        plt.show()


def solve_generalized_hermitian(H_proj, S_proj, rcond=1e-10):
    s_eval, s_evec = np.linalg.eigh(S_proj)
    s_max = np.max(np.abs(s_eval)) if s_eval.size > 0 else 0.0
    if s_max <= 0.0:
        raise ValueError("The projected overlap matrix is singular with zero spectrum.")

    keep = s_eval > (rcond * s_max)
    if not np.any(keep):
        raise ValueError(
            f"No overlap eigenvalues survived the cutoff rcond={rcond}. "
            f"Raw overlap eigenvalues: {s_eval}"
        )

    X = s_evec[:, keep] / np.sqrt(s_eval[keep])[None, :]
    H_orth = X.conj().T @ H_proj @ X
    evals, evecs_orth = np.linalg.eigh(H_orth)
    coeffs = X @ evecs_orth
    return evals, coeffs, s_eval, keep


@torch.no_grad()
def project_sector_hamiltonian_to_model_basis(
    model,
    psi0,
    E0,
    index,
    index_b,
    index_tk,
    number_tk,
    tk,
    block_size=4096,
    rcond=1e-10,
    verbose=True,
):
    sector_size = int(number_tk[tk])
    k = model.k
    invindex = build_invindex_sector(index, index_tk, number_tk, tk)

    H_proj = np.zeros((k, k), dtype=np.complex128)
    S_proj = np.zeros((k, k), dtype=np.complex128)

    n_blocks = (sector_size + block_size - 1) // block_size

    for block_id, start in enumerate(range(0, sector_size, block_size), start=1):
        end = min(start + block_size, sector_size)
        sector_positions_1d = np.arange(start, end, dtype=np.int64)
        basis_ids = index_tk[tk, sector_positions_1d]
        bitstrings = index_b[basis_ids]
        states_single = torch.as_tensor(
            bitstrings_to_occupations(bitstrings),
            dtype=torch.float64,
            device=device,
        )
        phi_blk = model.wavefunction_matrix(states_single).to(torch.complex128)  # (B, k)

        B = states_single.shape[0]
        states_blk = states_single.unsqueeze(1).repeat(1, k, 1)  # (B, k, N)
        psi_matrices_blk = phi_blk.unsqueeze(1).repeat(1, k, 1)  # (B, k, k)
        sector_positions_blk = np.repeat(sector_positions_1d[:, None], k, axis=1)

        hp_blk = h_psi_batch(
            model,
            states_blk,
            sector_positions_blk,
            psi_matrices_blk,
            psi0,
            E0,
            index,
            index_b,
            index_tk,
            number_tk,
            tk,
            invindex=invindex,
        )

        phi_blk_np = phi_blk.detach().cpu().numpy()
        hphi_blk_np = hp_blk[:, 0, :].detach().cpu().numpy()

        S_proj += phi_blk_np.conj().T @ phi_blk_np
        H_proj += phi_blk_np.conj().T @ hphi_blk_np

        if verbose and (block_id == 1 or block_id % 10 == 0 or block_id == n_blocks):
            print(
                f"[Projection] block {block_id:6d}/{n_blocks:6d} "
                f"| sector states [{start}, {end}) accumulated."
            )

    evals, coeffs, s_eval, keep = solve_generalized_hermitian(H_proj, S_proj, rcond=rcond)
    return {
        "H_proj": H_proj,
        "S_proj": S_proj,
        "generalized_evals": evals,
        "generalized_evecs": coeffs,
        "overlap_evals": s_eval,
        "effective_rank": int(np.sum(keep)),
        "sector_size": sector_size,
        "block_size": block_size,
        "tk": tk,
    }


def run_sector_workflow(tk, psi0, E0, index, index_b, index_tk, number_tk):
    invindex = build_invindex_sector(index, index_tk, number_tk, tk)
    sector_size = int(number_tk[tk])

    print(
        f"sector tk={tk}: num_k={num_k}, num_particles={num_particles}, "
        f"k={args.k}, sector_size={sector_size}"
    )

    model = FermiNQS(
        num_sites=num_sites,
        num_particles=num_particles,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        k=args.k,
    ).to(device)

    sampler = MetropolisSampler(
        model=model,
        N=num_sites,
        index_b=index_b,
        index_tk=index_tk,
        number_tk=number_tk,
        tk=tk,
        n_chains=args.n_chains,
        seed=args.seed,
    )

    stats = sample_and_evaluate(
        model,
        sampler,
        psi0,
        E0,
        index,
        index_b,
        index_tk,
        number_tk,
        tk,
        invindex,
        args.n_burn,
        args.n_between,
        args.n_samples,
    )
    print(
        f"initial E={stats['E_real_mean']:.10f} + {stats['E_imag_mean']:.3e}i "
        f"std={stats['E_real_std']:.10f} acc={stats['acceptance']:.4f}"
    )

    history = []
    if args.n_steps > 0:
        history = optimize_vmc(
            model,
            sampler,
            psi0,
            E0,
            index,
            index_b,
            index_tk,
            number_tk,
            tk,
            invindex,
            args.n_steps,
            args.lr,
            args.n_burn,
            args.n_between,
            args.n_samples,
        )

    projection_info = project_sector_hamiltonian_to_model_basis(
        model=model,
        psi0=psi0,
        E0=E0,
        index=index,
        index_b=index_b,
        index_tk=index_tk,
        number_tk=number_tk,
        tk=tk,
        block_size=args.projection_block_size,
        rcond=args.projection_rcond,
        verbose=True,
    )
    print("[Projection] overlap eigenvalues =", projection_info["overlap_evals"])
    print("[Projection] generalized eigenvalues =", projection_info["generalized_evals"])
    print("[Projection] effective rank =", projection_info["effective_rank"])
    print("[Projection] sector_size =", projection_info["sector_size"])
    print("[Projection] block_size =", projection_info["block_size"])

    return {
        "tk": tk,
        "sector_size": sector_size,
        "model": model,
        "sampler": sampler,
        "initial_stats": stats,
        "history": history,
        "projection_info": projection_info,
    }

if __name__ == "__main__":
    if args.plot_log_file is not None:
        plot_mc_energy_from_log(
            log_path=args.plot_log_file,
            ed_energy=args.plot_ed_energy,
            output_path=args.plot_output,
        )
        raise SystemExit(0)

    E0, psi0 = build_band()
    index, index_b, base_number, index_tk, number_tk = build_basis()
    if args.all_sectors:
        sector_results = []
        for tk in range(num_k):
            if number_tk[tk] == 0:
                continue
            if number_tk[tk] < args.k:
                print(
                    f"skip sector tk={tk}: sector_size={number_tk[tk]} is smaller than k={args.k}"
                )
                continue
            sector_results.append(
                run_sector_workflow(tk, psi0, E0, index, index_b, index_tk, number_tk)
            )
        print(f"completed {len(sector_results)} sectors")
        if sector_results:
            plot_sector_energies(sector_results)
    else:
        if number_tk[args.tk] < args.k:
            raise ValueError(
                f"sector tk={args.tk} has size {number_tk[args.tk]}, smaller than k={args.k}."
            )
        run_sector_workflow(args.tk, psi0, E0, index, index_b, index_tk, number_tk)

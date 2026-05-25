import argparse
import time
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math

class MLP_SubspaceNQS(nn.Module):
    def __init__(self, N, k, hidden=256, depth=2, dtype=torch.float64, device="cuda"):
        super().__init__()
        self.N = N
        self.k = k
        self.real_dtype = dtype
        self.complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.device = torch.device(device)

        layers = []
        in_dim = N
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.Tanh()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 2 * k)]
        self.net = nn.Sequential(*layers).to(self.device, dtype=self.real_dtype)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def f(self, s):
        out = self.net(s)  # (B, 2k), real
        re = out[:, :self.k]
        im = out[:, self.k:]
        return re.to(self.complex_dtype) + 1j * im.to(self.complex_dtype)


def all_spin_configs_pm1(N, device, dtype):
    M = 1 << N
    idx = torch.arange(M, device=device, dtype=torch.int64)
    bits = (idx[:, None] >> torch.arange(N, device=device, dtype=torch.int64)[None, :]) & 1
    s = bits.to(dtype) * 2.0 - 1.0
    return s, idx


def diag_energy_tfim(s, J):
    s_shift = torch.roll(s, shifts=-1, dims=1)
    return -J * torch.sum(s * s_shift, dim=1)


def build_psi_all(model, s_all, batch_eval=65536):
    M = s_all.shape[0]
    k = model.k
    psi_all = torch.empty((M, k), device=model.device, dtype=model.complex_dtype)
    for start in range(0, M, batch_eval):
        end = min(M, start + batch_eval)
        f = model.f(s_all[start:end])
        psi_all[start:end] = torch.exp(f)
    return psi_all


def compute_SH_exact_from_psi(psi_all, s_all, J, Gamma, eps=1e-6):
    device = psi_all.device
    cdtype = psi_all.dtype
    M, k = psi_all.shape
    N = s_all.shape[1]

    E_diag = diag_energy_tfim(s_all, J=J).to(cdtype)
    Hpsi = E_diag[:, None] * psi_all

    idx_base = torch.arange(M, device=device, dtype=torch.int64)
    for i in range(N):
        flip_idx = idx_base ^ (1 << i)
        Hpsi = Hpsi - Gamma * psi_all[flip_idx]

    psi_dag = psi_all.conj().transpose(0, 1)
    S = psi_dag @ psi_all
    H = psi_dag @ Hpsi

    S = 0.5 * (S + S.conj().transpose(0, 1))
    H = 0.5 * (H + H.conj().transpose(0, 1))

    S_reg = S
    return S_reg, H


def compute_SH_exact(model, N, J, Gamma, eps=1e-6, batch_eval=65536):
    s_all, _ = all_spin_configs_pm1(N, device=model.device, dtype=model.real_dtype)
    psi_all = build_psi_all(model, s_all, batch_eval=batch_eval)
    S, H = compute_SH_exact_from_psi(psi_all, s_all, J, Gamma, eps=eps)
    return S, H


def normalized_overlap_matrix(S, delta=1e-12):
    d = torch.real(torch.diag(S))
    inv_sqrt_d = torch.rsqrt(d).to(S.dtype)
    S_tilde = inv_sqrt_d[:, None] * S * inv_sqrt_d[None, :]
    S_tilde = 0.5 * (S_tilde + S_tilde.conj().transpose(0, 1))
    return S_tilde


def cholesky_whitened_H(S, H, eig_floor=1e-12):

    k = S.shape[0]
    eye = torch.eye(k, device=S.device, dtype=S.dtype)

    S = 0.5 * (S + S.conj().transpose(0, 1))
    S = S + eig_floor * eye

    L = torch.linalg.cholesky(S, upper=False)
    L_inv = torch.linalg.solve_triangular(L, eye, upper=False)

    Ht = L_inv @ H @ L_inv.conj().transpose(0, 1)
    Ht = 0.5 * (Ht + Ht.conj().transpose(0, 1))
    return Ht, L_inv


def generalized_ritz_solve(S, H, eig_floor=1e-12):
    #S = normalized_overlap_matrix(S, delta=eig_floor)
    Ht, L_inv = cholesky_whitened_H(S, H, eig_floor=eig_floor)

    if torch.is_grad_enabled():
        evals = torch.linalg.eigvalsh(Ht)
        coeffs = None
    else:
        evals, y = torch.linalg.eigh(Ht)
        coeffs = L_inv.conj().transpose(0, 1) @ y
    return evals, coeffs


def trace_from_SH(S, H, eig_floor=1e-12):
    Ht, _ = cholesky_whitened_H(S, H, eig_floor=eig_floor)
    return torch.real(torch.trace(Ht))


def freeE_from_SH(S, H, beta=1.0, eig_floor=1e-12):
    Ht, _ = cholesky_whitened_H(S, H, eig_floor=eig_floor)
    eval_H = torch.linalg.eigvalsh(Ht)
    F = -torch.log(torch.sum(torch.exp(-beta * eval_H))) / beta
    return torch.real(F)


def exact_diag_tfim(N, J=1.0, Gamma=1.0):
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.float64)

    def spin_z(state, i):
        return 1.0 if (state >> i) & 1 else -1.0

    for s in range(dim):
        e = 0.0
        for i in range(N):
            zi = spin_z(s, i)
            zj = spin_z(s, (i + 1) % N)
            e += -J * zi * zj
        H[s, s] += e
        for i in range(N):
            s_flip = s ^ (1 << i)
            H[s_flip, s] += -Gamma

    w = np.linalg.eigh(H)[0]
    return w


def reconstruct_ritz_wavefunctions(psi_all, coeffs, num_levels):
    phi = psi_all @ coeffs[:, :num_levels]
    phi = phi.transpose(0, 1).contiguous()
    return phi





def normalized_S_condition_penalty(
    S,
    norm_delta=1e-12,
    eig_delta=1e-8,
    cond_threshold=10.0,
    sharpness=100.0,
    power=4,
):
    S_tilde = normalized_overlap_matrix(S, delta=norm_delta)
    evals = torch.linalg.eigvalsh(S_tilde)

    lam_min = torch.min(evals)
    lam_max = torch.max(evals)
    cond = lam_max / (lam_min + eig_delta)

    log_cond = torch.log(cond)
    log_thr = math.log(cond_threshold)
    excess = torch.nn.functional.softplus(sharpness * (log_cond - log_thr)) / sharpness
    penalty = excess ** power

    return penalty, S_tilde, evals

def get_total_loss_from_SH(
    S, H, evals,
    loss_type="freeE",
    m=1,
    beta=1.0,
    eig_floor=1e-12,
    S_penalty_weight=0.0,
    S_norm_delta=1e-12,
    S_eig_delta=1e-8,
):
    #S = normalized_overlap_matrix(S, delta=S_norm_delta)

    if loss_type == "sum":
        phys_loss = torch.sum(evals[:m])
    elif loss_type == "trace":
        phys_loss = trace_from_SH(S, H, eig_floor=eig_floor)
    elif loss_type == "freeE":
        phys_loss = freeE_from_SH(S, H, beta=beta, eig_floor=eig_floor)
    else:
        raise ValueError(f"Unknown loss_type = {loss_type}")

    penalty = torch.zeros((), device=S.device, dtype=evals.dtype)
    S_tilde = None
    evals_tilde = None

    if S_penalty_weight > 0:
        penalty, S_tilde, evals_tilde = normalized_S_condition_penalty(
            S,
            norm_delta=S_norm_delta,
            eig_delta=S_eig_delta,
        )

    total_loss = phys_loss + S_penalty_weight * penalty
    return total_loss, phys_loss, penalty, S_tilde, evals_tilde


def save_checkpoint(path, model, optimizer, gamma_index, gamma_value, args):
    ckpt = {
        "gamma_index": gamma_index,
        "gamma_value": gamma_value,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    p = argparse.ArgumentParser()

    # Physical parameters
    p.add_argument("--N", type=int, default=14)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--Gamma_min", type=float, default=0.0)
    p.add_argument("--Gamma_max", type=float, default=3.0)
    p.add_argument("--Gamma_num", type=int, default=31)

    # Network / subspace parameters
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--depth", type=int, default=2)

    # Training / optimization
    p.add_argument("--max_iters", type=int, default=1000)
    p.add_argument("--min_iters", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=5.0)

    # Numerical options
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--batch_eval", type=int, default=65536)
    p.add_argument("--loss_type", type=str, default="trace", choices=["freeE", "trace", "sum"])
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eig_floor", type=float, default=1e-12)

    # Normalized-S condition penalty
    p.add_argument("--S_penalty_weight", type=float, default=1.0)
    p.add_argument("--S_norm_delta", type=float, default=1e-12)
    p.add_argument("--S_eig_delta", type=float, default=1e-8)

    # Runtime
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--print_every", type=int, default=50)

    # Output
    p.add_argument("--outdir", type=str, default="out_gamma_scan")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--plot_levels", type=int, default=3)
    p.add_argument("--save_levels", type=int, default=3)
    p.add_argument("--ckpt_name", type=str, default="scan_checkpoint.pt")

    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    wf_dir = os.path.join(args.outdir, "wavefunctions")
    os.makedirs(wf_dir, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""

    k = args.k
    m = k if args.m is None else min(args.m, k)
    plot_levels = max(1, min(args.plot_levels, k))
    save_levels = max(1, min(args.save_levels, k))

    gamma_list = np.linspace(args.Gamma_min, args.Gamma_max, args.Gamma_num, dtype=np.float64)

    model = MLP_SubspaceNQS(
        args.N,
        k,
        hidden=args.hidden,
        depth=args.depth,
        dtype=dtype,
        device=args.device,
    )
    opt = optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.outdir, args.ckpt_name)

    s_all, idx_all = all_spin_configs_pm1(args.N, device=model.device, dtype=model.real_dtype)
    spins_np = s_all.detach().cpu().numpy()
    basis_index_np = idx_all.detach().cpu().numpy()

    gamma_done = []
    nn_energies = []
    ed_energies = []
    losses = []
    phys_losses = []
    S_penalties = []
    gamma_times = []
    gamma_iters = []
    min_eig_Stilde_hist = []
    max_eig_Stilde_hist = []
    cond_Stilde_hist = []

    total_t0 = time.time()

    for ig, Gamma in enumerate(gamma_list):
        print(args.N)
        print("=" * 100)
        print(f"Gamma point {ig + 1}/{len(gamma_list)} : Gamma = {Gamma:.8f}")
        print("=" * 100)
        gamma_t0 = time.time()

        best_loss = float("inf")
        best_state_dict = None
        no_improve_count = 0
        actual_iters = 0

        for it in range(1, args.max_iters + 1):
            actual_iters = it
            opt.zero_grad(set_to_none=True)

            S, H = compute_SH_exact(
                model,
                N=args.N,
                J=args.J,
                Gamma=Gamma,
                eps=args.eps,
                batch_eval=args.batch_eval,
            )
            evals, coeffs = generalized_ritz_solve(S, H, eig_floor=args.eig_floor)
            args.S_penalty_weight=0 #!!!
            loss, phys_loss, S_penalty, S_tilde, evals_tilde = get_total_loss_from_SH(
                S, H, evals,
                loss_type=args.loss_type,
                m=m,
                beta=args.beta,
                eig_floor=args.eig_floor,
                S_penalty_weight=args.S_penalty_weight,
                S_norm_delta=args.S_norm_delta,
                S_eig_delta=args.S_eig_delta,
            )

            loss.backward()

            if args.clip is not None and args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)

            opt.step()

            loss_val = float(loss.item())

            if np.isfinite(best_loss):
                rel_improve = (best_loss - loss_val) / max(1.0, abs(best_loss))
            else:
                rel_improve = np.inf

            if rel_improve > args.tol:
                best_loss = loss_val
                best_state_dict = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if evals_tilde is not None:
                min_eig_tilde = torch.min(evals_tilde).item()
                max_eig_tilde = torch.max(evals_tilde).item()
                cond_tilde = max_eig_tilde / (min_eig_tilde + args.S_eig_delta)
            else:
                min_eig_tilde = float("nan")
                max_eig_tilde = float("nan")
                cond_tilde = float("nan")

            if it % args.print_every == 0 or it == 1:
                show = evals[:min(6, k)].detach().cpu().numpy()
                print(
                    f"  iter {it:5d} | "
                    f"loss={loss_val:.10f} | "
                    f"phys={phys_loss.item():.10f} | "
                    f"S_pen={S_penalty.item():.10f} | "
                    f"minEig(Stilde)={min_eig_tilde:.4e} | "
                    f"maxEig(Stilde)={max_eig_tilde:.4e} | "
                    f"cond(Stilde)={cond_tilde:.4e} | "
                    f"best_loss={best_loss:.10f} | "
                    f"no_improve={no_improve_count:4d} | "
                    f"Ritz[:min(6,k)]={np.array2string(show, precision=8)}"
                )

            if it >= args.min_iters and no_improve_count >= args.patience:
                print(
                    f"  Early stop at iter {it:d} "
                    f"(min_iters={args.min_iters}, patience={args.patience}, tol={args.tol:.2e})"
                )
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        with torch.no_grad():
            psi_all = build_psi_all(model, s_all, batch_eval=args.batch_eval)
            S, H = compute_SH_exact_from_psi(psi_all, s_all, J=args.J, Gamma=Gamma, eps=args.eps)
            evals, coeffs = generalized_ritz_solve(S, H, eig_floor=args.eig_floor)

            final_loss, final_phys_loss, final_S_penalty, S_tilde, evals_tilde = get_total_loss_from_SH(
                S, H, evals,
                loss_type=args.loss_type,
                m=m,
                beta=args.beta,
                eig_floor=args.eig_floor,
                S_penalty_weight=args.S_penalty_weight,
                S_norm_delta=args.S_norm_delta,
                S_eig_delta=args.S_eig_delta,
            )

            evals_S = torch.linalg.eigvalsh(S)
            phi = reconstruct_ritz_wavefunctions(psi_all, coeffs, save_levels)

        evals_np = evals.detach().cpu().numpy()
        coeffs_np = coeffs.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()
        psi_all_np = psi_all.detach().cpu().numpy()

        final_loss_val = float(final_loss.item())
        final_phys_loss_val = float(final_phys_loss.item())
        final_S_penalty_val = float(final_S_penalty.item())

        if evals_tilde is not None:
            evals_tilde_np = evals_tilde.detach().cpu().numpy()
            min_eig_tilde = float(np.min(evals_tilde_np))
            max_eig_tilde = float(np.max(evals_tilde_np))
            cond_tilde = max_eig_tilde / (min_eig_tilde + args.S_eig_delta)
            S_tilde_np = S_tilde.detach().cpu().numpy()
        else:
            evals_tilde_np = None
            min_eig_tilde = np.nan
            max_eig_tilde = np.nan
            cond_tilde = np.nan
            S_tilde_np = None

        #ed = exact_diag_tfim(args.N, J=args.J, Gamma=Gamma)
        #print("exact diag",ed[:k])
        gamma_done.append(Gamma)
        nn_energies.append(evals_np.copy())
        #ed_energies.append(ed[:k].copy())
        losses.append(final_loss_val)
        phys_losses.append(final_phys_loss_val)
        S_penalties.append(final_S_penalty_val)
        min_eig_Stilde_hist.append(min_eig_tilde)
        max_eig_Stilde_hist.append(max_eig_tilde)
        cond_Stilde_hist.append(cond_tilde)

        wf_path = os.path.join(wf_dir, f"wf_gamma_{ig:04d}{suffix}.npz")
        np.savez_compressed(
            wf_path,
            gamma=Gamma,
            ritz_evals=evals_np,
            #ed_evals=ed,
            coeffs=coeffs_np,
            phi=phi_np,
            psi_basis=psi_all_np,
            spins=spins_np,
            basis_index=basis_index_np,
            loss=final_loss_val,
            phys_loss=final_phys_loss_val,
            S_penalty=final_S_penalty_val,
            S_eigs=evals_S.detach().cpu().numpy(),
            S_tilde=S_tilde_np,
            S_tilde_eigs=evals_tilde_np,
            S_tilde_min_eig=min_eig_tilde,
            S_tilde_max_eig=max_eig_tilde,
            S_tilde_cond=cond_tilde,
            actual_iters=actual_iters,
            params=dict(vars(args)),
        )

        save_checkpoint(ckpt_path, model, opt, ig, Gamma, args)

        gamma_elapsed = time.time() - gamma_t0
        gamma_times.append(gamma_elapsed)
        gamma_iters.append(actual_iters)

        print(f"Finished Gamma = {Gamma:.8f} in {gamma_elapsed:.2f} s")
        print(f"Final total loss = {final_loss_val:.10f}")
        print(f"Final physical loss = {final_phys_loss_val:.10f}")
        print(f"Final normalized-S cond penalty = {final_S_penalty_val:.10f}")
        print(f"Final cond(Stilde) = {cond_tilde:.6e}")
        print(f"Saved wavefunction data to: {wf_path}")
        print(f"Checkpoint updated: {ckpt_path}")
        args.min_iter = 50

    total_elapsed = time.time() - total_t0
    print(f"\nAll scan finished. Total elapsed = {total_elapsed:.2f} s")

    gamma_done = np.array(gamma_done, dtype=np.float64)
    nn_energies = np.array(nn_energies, dtype=np.float64)
    ed_energies = np.array(ed_energies, dtype=np.float64)
    losses = np.array(losses, dtype=np.float64)
    phys_losses = np.array(phys_losses, dtype=np.float64)
    S_penalties = np.array(S_penalties, dtype=np.float64)
    gamma_times = np.array(gamma_times, dtype=np.float64)
    gamma_iters = np.array(gamma_iters, dtype=np.int64)
    min_eig_Stilde_hist = np.array(min_eig_Stilde_hist, dtype=np.float64)
    max_eig_Stilde_hist = np.array(max_eig_Stilde_hist, dtype=np.float64)
    cond_Stilde_hist = np.array(cond_Stilde_hist, dtype=np.float64)

    scan_path = os.path.join(args.outdir, f"scan_results{suffix}.npz")
    np.savez_compressed(
        scan_path,
        gamma=gamma_done,
        nn_energies=nn_energies,
        ed_energies=ed_energies,
        loss=losses,
        phys_loss=phys_losses,
        S_penalty=S_penalties,
        gamma_times=gamma_times,
        gamma_iters=gamma_iters,
        S_tilde_min_eig=min_eig_Stilde_hist,
        S_tilde_max_eig=max_eig_Stilde_hist,
        S_tilde_cond=cond_Stilde_hist,
        params=dict(vars(args)),
        elapsed=total_elapsed,
    )

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    for n in range(plot_levels):
        ax1.plot(gamma_done, nn_energies[:, n], marker="o", ms=3, label=f"NN E[{n}]")
        #ax1.plot(gamma_done, ed_energies[:, n], linestyle="--", label=f"ED E[{n}]")
    ax1.set_xlabel(r"$\Gamma$")
    ax1.set_ylabel("Energy")
    ax1.set_title("TFIM energy spectrum vs transverse field")
    ax1.legend()
    fig1.tight_layout()
    fig1_path = os.path.join(args.outdir, f"energy_vs_gamma{suffix}.png")
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.plot(gamma_done, gamma_times, marker="o", ms=3)
    ax2.set_xlabel(r"$\Gamma$")
    ax2.set_ylabel("Training time per Gamma (s)")
    ax2.set_title("Training time vs transverse field")
    fig2.tight_layout()
    fig2_path = os.path.join(args.outdir, f"time_vs_gamma{suffix}.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(gamma_done, gamma_iters, marker="o", ms=3)
    ax3.set_xlabel(r"$\Gamma$")
    ax3.set_ylabel("Actual training iterations")
    ax3.set_title("Converged iterations vs transverse field")
    fig3.tight_layout()
    fig3_path = os.path.join(args.outdir, f"iters_vs_gamma{suffix}.png")
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    fig4 = plt.figure(figsize=(8, 6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(gamma_done, cond_Stilde_hist, marker="o", ms=3)
    ax4.set_xlabel(r"$\Gamma$")
    ax4.set_ylabel(r"$\kappa(\tilde S)$")
    ax4.set_title("Condition number of normalized overlap matrix")
    ax4.set_yscale("log")
    fig4.tight_layout()
    fig4_path = os.path.join(args.outdir, f"cond_Stilde_vs_gamma{suffix}.png")
    fig4.savefig(fig4_path, dpi=200)
    plt.close(fig4)

    print("\nSaved files:")
    print("  ", scan_path)
    print("  ", fig1_path)
    print("  ", fig2_path)
    print("  ", fig3_path)
    print("  ", fig4_path)
    print("  ", ckpt_path)
    print("  ", wf_dir)


if __name__ == "__main__":
    main()

# SNQS.py 说明文档

## 文件用途

`SNQS.py` 实现了一个基于多输出 MLP 的子空间 Neural Quantum State。程序对一维周期边界横场 Ising 模型进行 Gamma 扫描，枚举完整自旋基，构造投影重叠矩阵 `S` 和哈密顿矩阵 `H`，再通过 Cholesky 白化求解 Ritz 广义本征值问题。

主要流程：

1. 构建全部 `2^N` 个自旋构型，取值为 `-1/+1`。
2. 用 `MLP_SubspaceNQS` 输出 `k` 个复数波函数分量。
3. 对每个 `Gamma` 点训练模型。
4. 计算投影能谱、损失和重叠矩阵条件数。
5. 保存扫描结果、波函数数据、checkpoint 和图像。

## 运行方式

```bash
python SNQS.py
```

示例：

```bash
python SNQS.py --N 12 --k 4 --Gamma_min 0 --Gamma_max 2 --Gamma_num 21 --max_iters 500 --device cuda --outdir out_snqs
```

## 输入参数设置

| 参数 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `--N` | int | `14` | 自旋链长度，Hilbert 空间维度为 `2^N` |
| `--J` | float | `1.0` | Ising 相互作用强度 |
| `--Gamma_min` | float | `0.0` | 横场扫描起点 |
| `--Gamma_max` | float | `3.0` | 横场扫描终点 |
| `--Gamma_num` | int | `31` | 横场采样点数 |
| `--k` | int | `6` | 子空间波函数数量，即模型复输出维度 |
| `--m` | int/None | `None` | `loss_type=sum` 时累加前 `m` 个能级；为空时使用 `k` |
| `--hidden` | int | `1024` | MLP 隐藏层宽度 |
| `--depth` | int | `2` | MLP 隐藏层层数 |
| `--max_iters` | int | `1000` | 每个 Gamma 点最大优化迭代数 |
| `--min_iters` | int | `100` | 早停前最小迭代数 |
| `--patience` | int | `30` | 无显著改进后的早停等待步数 |
| `--tol` | float | `1e-4` | 判断 loss 相对改进的阈值 |
| `--lr` | float | `1e-3` | Adam 学习率 |
| `--clip` | float | `5.0` | 梯度裁剪最大范数；小于等于 0 时不裁剪 |
| `--eps` | float | `1e-6` | 矩阵构造相关预留正则参数 |
| `--batch_eval` | int | `65536` | 全基评估波函数时的 batch 大小 |
| `--loss_type` | str | `trace` | 损失类型，可选 `freeE`、`trace`、`sum` |
| `--beta` | float | `1.0` | `freeE` 损失中的反温度 |
| `--eig_floor` | float | `1e-12` | Cholesky 白化前给 `S` 添加的对角正则 |
| `--S_penalty_weight` | float | `1.0` | 归一化重叠矩阵条件数惩罚权重；代码训练中当前被强制置为 0 |
| `--S_norm_delta` | float | `1e-12` | 归一化重叠矩阵时使用的稳定项 |
| `--S_eig_delta` | float | `1e-8` | 条件数计算中最小本征值稳定项 |
| `--seed` | int | `1` | 随机种子 |
| `--device` | str | `cuda` 或 `cpu` | 运行设备，默认有 CUDA 则用 `cuda` |
| `--dtype` | str | `float64` | 实数精度，可选 `float32`、`float64` |
| `--print_every` | int | `50` | 训练日志打印间隔 |
| `--outdir` | str | `out_gamma_scan` | 输出目录 |
| `--tag` | str | 空字符串 | 输出文件名后缀 |
| `--plot_levels` | int | `3` | 能谱图绘制的能级数 |
| `--save_levels` | int | `3` | 保存 Ritz 波函数的能级数 |
| `--ckpt_name` | str | `scan_checkpoint.pt` | checkpoint 文件名 |

## 主要类和函数

| 名称 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `MLP_SubspaceNQS(N, k, hidden, depth, dtype, device)` | 系统尺寸、输出维度、网络结构和设备 | PyTorch 模型 | 多输出复波函数 ansatz，实部和虚部分别由 MLP 的前后 `k` 个输出给出 |
| `MLP_SubspaceNQS.f(s)` | `s: (B, N)` 自旋构型 | `(B, k)` 复数 log 波函数 | 返回 `f(s)`，后续通过 `exp(f)` 得到波函数 |
| `all_spin_configs_pm1(N, device, dtype)` | 自旋数 | `s, idx` | 枚举所有 `-1/+1` 自旋构型和对应整数编号 |
| `diag_energy_tfim(s, J)` | 自旋构型、耦合强度 | 对角能量 | 计算 `-J sum_i s_i s_{i+1}` |
| `build_psi_all(model, s_all, batch_eval)` | 模型、全基构型 | `(2^N, k)` 波函数矩阵 | 分 batch 计算 `exp(model.f(s))` |
| `compute_SH_exact_from_psi(psi_all, s_all, J, Gamma, eps)` | 波函数、全基、自旋参数 | `S, H` | 显式构造投影重叠矩阵和哈密顿矩阵 |
| `compute_SH_exact(model, N, J, Gamma, eps, batch_eval)` | 模型和物理参数 | `S, H` | 从模型直接构造 `S, H` |
| `cholesky_whitened_H(S, H, eig_floor)` | 投影矩阵 | 白化哈密顿量、`L_inv` | 用 Cholesky 对 `S` 白化 |
| `generalized_ritz_solve(S, H, eig_floor)` | `S, H` | 本征值、系数 | 求解广义 Ritz 能谱 |
| `trace_from_SH(S, H, eig_floor)` | `S, H` | 标量 | 返回白化哈密顿量 trace |
| `freeE_from_SH(S, H, beta, eig_floor)` | `S, H` 和反温度 | 标量自由能 | 计算 `-log(sum exp(-beta E))/beta` |
| `exact_diag_tfim(N, J, Gamma)` | TFIM 参数 | 精确本征值 | 构造 dense Hamiltonian 并精确对角化 |
| `reconstruct_ritz_wavefunctions(psi_all, coeffs, num_levels)` | 基函数和 Ritz 系数 | Ritz 波函数 | 重构前若干能级波函数 |
| `normalized_S_condition_penalty(...)` | `S` 和条件数参数 | 惩罚、归一化矩阵、本征值 | 计算归一化 `S` 的条件数惩罚 |
| `get_total_loss_from_SH(...)` | `S, H, evals` 和损失配置 | 总损失、物理损失、惩罚项 | 根据 `loss_type` 汇总优化目标 |
| `save_checkpoint(path, model, optimizer, gamma_index, gamma_value, args)` | 路径、模型、优化器、参数 | `.pt` 文件 | 保存训练状态 |

## 输出文件

默认输出目录为 `out_gamma_scan`：

| 文件/目录 | 说明 |
|---|---|
| `scan_results*.npz` | Gamma 扫描汇总结果，包括能谱、loss、耗时、迭代数、条件数等 |
| `wavefunctions/wf_gamma_*.npz` | 每个 Gamma 点的波函数、Ritz 系数、基态信息和重叠矩阵信息 |
| `energy_vs_gamma*.png` | 能谱随 Gamma 变化图 |
| `time_vs_gamma*.png` | 每个 Gamma 点训练耗时图 |
| `iters_vs_gamma*.png` | 收敛迭代数图 |
| `cond_Stilde_vs_gamma*.png` | 归一化重叠矩阵条件数图 |
| `scan_checkpoint.pt` | 最新 Gamma 点的模型和优化器 checkpoint |

## 注意事项

- 程序枚举完整 Hilbert 空间，内存和时间随 `2^N` 增长；`N=14` 已经是较大的默认设置。
- 当前训练循环中存在 `args.S_penalty_weight=0 #!!!`，会覆盖命令行传入的 `--S_penalty_weight`。
- `ed_energies` 的精确对角化代码被注释，保存时会得到空数组。

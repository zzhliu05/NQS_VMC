# NQS_transformer/main.py 说明文档

## 文件用途

`NQS_transformer/main.py` 是基于 ViT 波函数的 TFIM NQS-VMC 训练入口。它使用 `ViT.py` 中的 `ViTWaveFunction` 作为复波函数模型，通过 Metropolis 采样和 SR/minSR 优化训练，并可在 Hilbert 空间较小时将 Hamiltonian 投影到模型子空间，与精确对角化能谱对比。

主要流程：

1. 创建 ViT 波函数模型。
2. 为一个或多个横场 `h_values` 创建采样器。
3. 对每个横场采样，计算局域能量和 SR 统计量。
4. 支持逐场累加更新或合并统计量后统一更新。
5. 训练结束后执行投影广义本征求解。
6. 保存投影结果和投影能谱对比图。

## 运行方式

```bash
cd NQS_transformer
python main.py
```

示例：

```bash
python main.py --N 10 --h_values 0.7 0.9 1.2 --k 2 --n_iter 100 --n_chains 256
```

## 输入参数设置

| 参数 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `--N` | int | `10` | TFIM 自旋链长度 |
| `--J` | float | `1.0` | Ising 相互作用强度 |
| `--h` | float | `0.5` | 单横场默认值；当 `h_values` 为空时使用 |
| `--h_values` | float list | `[0.7, 0.9, 1.2]` | 同时训练/评估的横场列表 |
| `--token_size` | int | `5` | 每个 token 包含的自旋站点数；必须整除 `N` |
| `--d_model` | int | `16` | Transformer token embedding 维度 |
| `--num_heads` | int | `2` | 多头注意力头数；必须整除 `d_model` |
| `--mlp_dim` | int | `64` | Transformer block 中 MLP 隐藏维度 |
| `--num_layers` | int | `2` | Transformer encoder block 层数 |
| `--k` | int | `2` | 复波函数输出数量和 determinant 矩阵大小 |
| `--n_iter` | int | `100` | 训练迭代次数 |
| `--n_chains` | int | `256` | Monte Carlo 链数量 |
| `--n_burn` | int | `10` | 每轮采样前 burn-in sweep 数 |
| `--n_between` | int | `1` | 保存样本之间的 sweep 数 |
| `--n_samples` | int | `1` | 每轮保存样本数 |
| `--sr_lr` | float | `0.05` | SR/minSR 学习率 |
| `--diag_shift` | float | `1e-4` | SR 线性系统对角正则 |
| `--cg_tol` | float | `1e-8` | 共轭梯度收敛阈值 |
| `--cg_maxiter` | int | `500` | 共轭梯度最大迭代次数 |
| `--combine_sr_stats` | bool | `True` | 是否将多个 `h` 的 SR 统计量合并后统一更新 |
| `--projection_dim_cutoff` | int | `2^14` | Hilbert 维度超过该值时跳过完整基投影 |
| `--projection_batch_size` | int | `1024` | 投影时全基波函数评估 batch 大小 |
| `--projection_rcond` | float | `1e-10` | 投影广义本征问题的重叠矩阵相对截断阈值 |
| `--projection_output_prefix` | str/None | `None` | 投影 `.npz` 输出文件前缀；为空则不保存投影数据文件 |
| `--comparison_plot_path` | str | `projection_vs_exact_energy.png` | 投影能谱与精确能谱对比图路径 |
| `--seed` | int | `42` | 随机种子 |

## 主要类

| 类 | 初始化输入 | 说明 |
|---|---|---|
| `ViTAdapter` | `num_sites, token_size, d_model, num_heads, mlp_dim, num_layers, k, bias, dropout` | 将 `ViTWaveFunction` 适配到训练流程需要的 `log_wavefunction_matrix` 和 `wavefunction_matrix` 接口 |
| `MetropolisSampler` | `model, N, n_chains, gamma, seed` | TFIM 自旋构型采样器，状态取值为 `-1/+1` |
| `SRLinearOperator` | `D_centered, diag_shift` | 标准 SR 线性算子 |
| `minSRLinearOperator` | `D_centered, diag_shift` | minSR 线性算子，在线性系统中工作于样本空间 |

## 主要函数

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `sync_if_cuda()` | 无 | 无 | 若可用 CUDA，则同步 GPU 计时 |
| `build_vit_model(...)` | 模型结构参数 | `ViTAdapter` | 构建 ViT 波函数模型并打印参数量 |
| `build_samplers(model, N, n_chains, h_values, seed)` | 模型和横场列表 | 采样器字典 | 为每个横场创建一个 Metropolis sampler |
| `maybe_print_exact_energies(N, J, h_values, k)` | 小尺寸 TFIM 参数 | 精确能量字典 | `N <= 14` 时执行精确对角化 |
| `plot_projection_vs_exact_energies(...)` | 投影结果、精确能量、输出路径 | PNG 图 | 绘制投影能谱与精确能谱对比 |
| `apply_joint_update(model, delta_sum)` | 模型、合计更新向量 | 耗时 | 将多个横场的 SR 更新一次性加到参数上 |
| `refresh_all_samplers(samplers, h_values)` | 采样器字典 | 无 | 更新模型参数后刷新所有 sampler cache |
| `print_iteration_log(...)` | 训练统计 | 无 | 打印单次迭代日志 |
| `initialize_combined_sr_stats(stats)` | 单场 SR 统计 | 合并统计字典 | 初始化跨横场统计量 |
| `accumulate_combined_sr_stats(combined_stats, stats)` | 合并统计、新统计 | 更新后统计 | 累加多个横场的 `D_centered` 和能量中心化量 |
| `sr_step_cg_from_combined_stats(...)` | 模型、合并统计、SR 参数 | 更新信息 | 基于合并统计执行 minSR 更新 |
| `compute_wavefunction_matrix(model, states, gamma)` | 状态和横场 | 波函数矩阵 | 调用模型计算波函数 |
| `compute_wavefunction_matrices(model, states, gamma)` | `(n_chains, k, N)` 状态 | `(n_chains, k, k)` 矩阵 | 批量构造 determinant 矩阵 |
| `compute_logdet_and_logprob(psi_matrices)` | 波函数矩阵 | `logdet, logprob` | 计算 determinant 和采样权重 |
| `row_replacement_det_ratio(...)` | 新行、逆矩阵、行索引 | determinant ratio | Metropolis 行替换接受率核心 |
| `get_param_vector(model)` | 模型 | 参数向量 | 将参数展平 |
| `set_param_vector(model, vec)` | 模型、向量 | 无 | 将向量写回模型参数 |
| `num_params(model)` | 模型 | 参数量 | 统计参数数量 |
| `h_psi_tfim_batch(model, states, psi_matrices, field, J)` | 样本、波函数矩阵和 TFIM 参数 | `H psi` | 计算 TFIM Hamiltonian 作用 |
| `local_energy_tfim_batch(...)` | 同上 | local energy | 计算局域能量 |
| `compute_D_matrix(model, states, field)` | 模型、样本、横场 | 导数矩阵 `D` | 对 log determinant 关于参数求导 |
| `compute_O_matrix(model, states, field)` | 同上 | 导数矩阵 | 当前直接调用 `compute_D_matrix` |
| `compute_sr_statistics(...)` | 模型、样本、横场、局域能量 | SR 统计字典 | 构造 force、协方差、中心化导数 |
| `conjugate_gradient(...)` | 线性算子、右端向量 | 解、迭代数、耗时 | CG 求解线性系统 |
| `sr_step_cg(...)` | 模型、样本、横场、局域能量、SR 参数 | 更新信息 | 单横场 minSR 更新向量计算 |
| `exact_ground_state_energy_tfim(N, J, h, k)` | TFIM 参数 | 前 `k` 个精确能级 | dense Hamiltonian 精确对角化 |
| `enumerate_tfim_basis_states(N)` | 自旋数 | 全基自旋构型 | 枚举 `2^N` 个 `-1/+1` 构型 |
| `build_tfim_hamiltonian_dense(N, J, h)` | TFIM 参数 | dense Hamiltonian | 构建完整 Hamiltonian 矩阵 |
| `compute_wavefunctions_on_full_basis(...)` | 模型、全基参数 | 基态、波函数矩阵 | 分 batch 计算全 Hilbert 空间波函数 |
| `solve_generalized_hermitian(...)` | `H_proj, S_proj, rcond` | 投影能谱和系数 | 正则化求解广义本征问题 |
| `project_hamiltonian_to_model_basis(...)` | 模型、TFIM 参数 | 投影结果字典 | 投影 Hamiltonian 到模型输出子空间 |
| `train_tfim_nqs_sr(...)` | 训练和模型参数 | 模型、历史、精确能量、投影结果 | 对每个横场分别计算更新，再 joint update |
| `train_tfim_nqs_sr_combined_stats(...)` | 同上 | 同上 | 先合并多个横场统计量，再执行统一 SR 更新 |
| `build_arg_parser()` | 无 | argparse parser | 定义命令行参数 |

## 输出文件

| 输出 | 条件 | 说明 |
|---|---|---|
| `projection_vs_exact_energy.png` | 默认生成 | 投影能谱和精确能谱对比图 |
| `{projection_output_prefix}_h_{field}.npz` | 设置 `--projection_output_prefix` | 保存投影矩阵、波函数、广义本征值等 |

## 注意事项

- 默认 `--combine_sr_stats` 的 `default=True` 且使用 `store_true`，因此命令行中无法通过不传参数关闭它；若需要关闭，需要修改默认值或添加反向参数。
- `token_size` 必须整除 `N`。
- `d_model` 必须能被 `num_heads` 整除。
- 完整基投影的计算量随 `2^N` 增长，超过 `projection_dim_cutoff` 会自动跳过。

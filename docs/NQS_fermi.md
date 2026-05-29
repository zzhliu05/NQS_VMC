# NQS_fermi.py 说明文档

## 文件用途

`NQS_fermi.py` 实现费米格点体系的 Neural Quantum State 变分 Monte Carlo。代码构建动量空间基、单粒子紧束缚能带和相互作用 Hamiltonian，并在指定总动量扇区中使用 Metropolis 采样与 stochastic reconfiguration 进行优化。

主要能力：

- 构建二维动量网格 `num_kx * num_ky`。
- 按粒子数生成费米占据基。
- 按总动量 `tk` 划分 Hilbert 空间扇区。
- 使用 `FermiNQS`/`ComplexMLP` 表示复波函数。
- 使用行替换 determinant ratio 进行 Metropolis 更新。
- 使用 CG 求解 SR 更新。
- 可解析训练日志并绘制 MC 能量曲线。
- 训练后将 Hamiltonian 投影到模型子空间并求解广义本征值。

## 运行方式

训练单个动量扇区：

```bash
python NQS_fermi.py --tk 0
```

训练所有非空扇区：

```bash
python NQS_fermi.py --all_sectors --k 1 --n_steps 200
```

仅解析日志并绘图：

```bash
python NQS_fermi.py --plot_log_file train.log --plot_ed_energy -12.34 --plot_output mc_energy.png
```

## 输入参数设置

| 参数 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `--num_kx` | int | `3` | x 方向动量格点数 |
| `--num_ky` | int | `6` | y 方向动量格点数 |
| `--num_particles` | int/None | `None` | 费米子数；为空时使用 `num_k // 3` |
| `--tk` | int | `0` | 总动量扇区编号 |
| `--all_sectors` | bool | `False` | 是否优化所有非空总动量扇区 |
| `--epsilon` | float | `0.0` | 单粒子能量偏置 |
| `--t1` | float | `1.0` | 紧束缚 hopping 参数 `t1` |
| `--t2` | float | `1 - sqrt(1/2)` | 紧束缚 hopping 参数 `t2` |
| `--M` | float | `0.0` | 质量项 |
| `--phi` | float | `pi/4` | 相位参数 |
| `--hidden_sizes` | int list | `[32, 32]` | MLP 隐藏层宽度列表，例如 `--hidden_sizes 64 64 32` |
| `--activation` | str | `tanh` | 激活函数，可选 `tanh`、`relu`、`gelu` |
| `--k` | int | `1` | 复波函数输出数量；也决定 determinant 矩阵大小 |
| `--n_chains` | int | `1024` | Monte Carlo 链数量 |
| `--n_burn` | int | `1` | burn-in sweep 数 |
| `--n_between` | int | `1` | 两次保存样本之间的 sweep 数 |
| `--n_samples` | int | `1` | 每轮保存的样本数 |
| `--n_steps` | int | `200` | VMC/SR 优化步数 |
| `--lr` | float | `5e-2` | SR 参数更新步长 |
| `--diag_shift` | float | `1e-3` | SR 协方差矩阵对角正则 |
| `--cg_tol` | float | `1e-8` | 共轭梯度收敛阈值 |
| `--cg_maxiter` | int | `200` | 共轭梯度最大迭代次数 |
| `--logphi_real_clip` | float | `30.0` | 对 `Re(log phi)` 做指数前裁剪，防止溢出 |
| `--max_sr_step_norm` | float | `10.0` | SR 更新向量范数上限 |
| `--seed` | int | `1234` | 随机种子 |
| `--plot_log_file` | str/None | `None` | 训练日志路径；设置后只解析日志并绘图 |
| `--plot_ed_energy` | float/None | `None` | 日志图中的精确对角化参考能量 |
| `--plot_output` | str/None | `None` | 日志图输出路径；为空则直接显示 |
| `--projection_block_size` | int | `4096` | 最终投影时每个 block 处理的扇区基态数 |
| `--projection_rcond` | float | `1e-10` | 广义本征问题中重叠矩阵的相对截断阈值 |

## 全局派生参数

| 名称 | 计算方式 | 说明 |
|---|---|---|
| `num_k` | `num_kx * num_ky` | 总动量格点数 |
| `num_sites` | `num_k` | NQS 输入维度 |
| `num_particles` | 参数值或 `num_k // 3` | 粒子数 |
| `kx, ky` | 均匀离散动量 | 动量网格坐标 |

## 主要类

| 类 | 初始化输入 | 说明 |
|---|---|---|
| `ComplexMLP` | `N, hidden_sizes, activation, k` | 复数 MLP ansatz，分别用两个线性头输出实部和虚部 |
| `FermiNQS` | `num_sites, num_particles, hidden_sizes, activation, k` | 费米占据配置的 NQS 包装器，检查输入为 0/1 占据数 |
| `MetropolisSampler` | `model, N, index_b, index_tk, number_tk, tk, n_chains, seed` | 固定总动量扇区内的 Metropolis 采样器 |
| `SRLinearOperator` | `D_centered, diag_shift` | SR 线性算子，提供 CG 所需的 `matvec` |

## 主要函数

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `parse_arguments()` | 无 | `args` | 读取命令行参数 |
| `build_band()` | 使用全局 `args` | `E0, psi0` | 构建单粒子紧束缚能带和本征矢 |
| `build_basis()` | 使用全局粒子数和动量网格 | `index, index_b, base_number, index_tk, number_tk` | 构建费米占据基并按总动量分扇区 |
| `count_inversions(arr)` | 排列数组 | 逆序数 | 计算费米符号所需的逆序数 |
| `count_ones(n)` | 整数 bitstring | `count, positions` | 统计占据数和占据位置 |
| `build_interacting_hamiltonian(...)` | 扇区、能带、基信息 | `Hi` | 构建某总动量扇区的相互作用 Hamiltonian 矩阵 |
| `bitstrings_to_occupations(bitstrings)` | bitstring 数组 | `0/1` 占据矩阵 | 将整数基态转成 NQS 输入 |
| `sector_basis_bitstrings(...)` | 基信息和 `tk` | bitstring 数组 | 返回指定扇区的基态标签 |
| `build_sector_neighbors(Hi, atol)` | Hamiltonian 矩阵 | 非零邻接表 | 缓存非零矩阵元 |
| `build_invindex_sector(...)` | 基信息和 `tk` | 反索引矩阵 | 将动量标签映射到每个基态中的占据位置 |
| `compute_wavefunction_matrices(model, states)` | `(n_chains, k, N)` 状态 | `(n_chains, k, k)` 波函数矩阵 | 批量计算 determinant 所需矩阵 |
| `compute_logdet_and_logprob(psi_matrices)` | 波函数矩阵 | `logdet, logprob` | 计算 determinant 和采样概率 |
| `row_replacement_det_ratio(...)` | 新行、逆矩阵、行索引 | determinant ratio | 用于快速 Metropolis 接受率 |
| `hamiltonian_row_terms(...)` | 扇区基态编号和模型数据 | 对角项、非对角项 | 计算并缓存 Hamiltonian 一行的非零项 |
| `h_psi_batch(...)` | 模型、样本、Hamiltonian 数据 | `H psi` | 批量计算 Hamiltonian 作用后的波函数 |
| `E_loc(...)` | 样本和 `H psi` 数据 | local energy | 计算局域能量 |
| `compute_D_matrix(model, states)` | 模型和样本 | 导数矩阵 `D` | 对参数求导，构造 SR 统计量 |
| `compute_sr_statistics(...)` | 模型、样本、局域能量 | SR 统计字典 | 计算 `D_centered`、force、协方差等 |
| `conjugate_gradient(...)` | 线性算子和右端向量 | 解向量、迭代数 | CG 求解 SR 线性系统 |
| `sr_step_cg(...)` | 模型、样本、局域能量、SR 参数 | SR 更新信息 | 计算一次 SR 更新向量 |
| `sample_and_evaluate(...)` | 模型、采样器和 Hamiltonian 数据 | 样本统计 | 采样并计算能量、方差、接受率和耗时 |
| `optimize_vmc(...)` | 模型、采样器、训练参数 | `history` | 执行多步 VMC/SR 优化 |
| `plot_sector_energies(sector_results)` | 多扇区结果 | 图像 | 绘制不同动量扇区能量 |
| `parse_mc_energy_log(log_path)` | 日志路径 | 初始统计、step 列表 | 解析训练日志中的 MC 能量 |
| `plot_mc_energy_from_log(...)` | 日志、参考能量、输出路径 | 图像 | 绘制训练能量和接受率曲线 |
| `solve_generalized_hermitian(...)` | `H_proj, S_proj, rcond` | 本征值、系数、重叠谱 | 正则化求解投影广义本征问题 |
| `project_sector_hamiltonian_to_model_basis(...)` | 模型、扇区和 Hamiltonian 数据 | 投影结果字典 | 将扇区 Hamiltonian 投影到模型基 |
| `run_sector_workflow(...)` | `tk` 和基/能带数据 | 扇区结果字典 | 单个扇区完整流程入口 |

## 输出和日志

该脚本主要通过标准输出打印：

- 初始局域能量、虚部、标准差、接受率。
- 每 100 步的 SR 训练统计。
- 采样、局域能量、SR、更新等耗时。
- 投影重叠矩阵本征值。
- 投影广义本征能谱。
- 有效秩、扇区大小和 block size。

`--plot_output` 可保存日志解析图；训练流程本身当前未默认保存模型 checkpoint。

## 注意事项

- 程序启动时会立即解析命令行参数并设置全局变量，因此作为模块导入时也会读取 `sys.argv`。
- 默认 dtype 为 `torch.float64`。
- 默认设备为有 CUDA 时使用 GPU，否则使用 CPU。
- `--all_sectors` 会跳过空扇区以及扇区大小小于 `k` 的扇区。

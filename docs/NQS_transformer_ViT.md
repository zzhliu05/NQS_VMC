# NQS_transformer/ViT.py 说明文档

## 文件用途

`ViT.py` 定义了用于一维自旋构型的 Vision Transformer 风格复波函数模型。它将长度为 `num_sites` 的自旋配置切分成固定长度 token，并把横场参数 `gamma` 附加到每个 token 后投影到 `d_model` 维空间，再经过 Transformer encoder，最后输出复数 log 波函数。

该文件不包含命令行入口，通常由 `NQS_transformer/main.py` 导入使用。

## 模型输入约定

| 输入 | 支持形状 | 说明 |
|---|---|---|
| `x` | `(num_sites,)` 或 `(batch_size, num_sites)` | 自旋构型，训练代码中通常取值为 `-1/+1` |
| `gamma` | 标量 | 所有 batch 和 token 共享同一个横场参数 |
| `gamma` | `(batch_size,)` | 每个样本一个横场参数 |
| `gamma` | `(batch_size, num_tokens)` | 每个样本、每个 token 一个横场参数 |
| `gamma` | `(batch_size, num_tokens, 1)` | 已扩展好的 token-wise 参数 |

## 结构参数设置

这些参数由 `main.py` 的 `ViTAdapter` 或直接构造类时传入。

| 参数 | 类型 | 说明 |
|---|---:|---|
| `num_sites` | int | 输入站点数，必须为正数 |
| `token_size` | int | 每个 token 的站点数，必须为正数且整除 `num_sites` |
| `d_model` | int | token embedding 和 Transformer 隐层维度，必须为正数 |
| `num_heads` | int | 注意力头数，必须为正数且整除 `d_model` |
| `mlp_dim` | int | Transformer block 内部 MLP 隐藏维度，必须为正数 |
| `num_layers` | int | Transformer encoder block 层数，必须为正数 |
| `num_outputs` | int | 复波函数输出数量，必须为正数 |
| `bias` | bool | 线性层是否使用 bias |
| `dropout` | float | dropout 概率 |

## 主要函数和类

| 名称 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `count_model_parameters(model, verbose=True)` | PyTorch 模型 | 参数总数 | 统计模型参数量，可打印到终端 |
| `LinearTokenEmbedding` | `num_sites, token_size, d_model, bias` | 模块 | 将站点配置切分成 token，并附加 `gamma` 后线性投影 |
| `LinearTokenEmbedding.tokenize(x)` | `(num_sites,)` 或 `(B, num_sites)` | `(B, num_tokens, token_size)` | 切分输入配置 |
| `LinearTokenEmbedding.append_gamma(tokens, gamma)` | token 和横场 | `(B, num_tokens, token_size + 1)` | 将横场参数拼接到每个 token |
| `LinearTokenEmbedding.forward(x, gamma)` | 配置和横场 | `(B, num_tokens, d_model)` | token 化、拼接横场并投影 |
| `ViTEmbedding` | `num_sites, token_size, d_model, bias` | 模块 | 对 `LinearTokenEmbedding` 的轻量封装 |
| `MultiHeadSelfAttention` | `d_model, num_heads, bias, dropout` | 模块 | 标准多头自注意力层 |
| `MultiHeadSelfAttention.forward(x)` | `(B, num_tokens, d_model)` | 同形状张量 | 执行 scaled dot-product attention |
| `TransformerEncoderBlock` | `d_model, num_heads, mlp_dim, bias, dropout` | 模块 | Pre-LN attention + MLP 残差块 |
| `TransformerEncoderBlock.forward(x)` | token embedding | token embedding | 一层 Transformer 编码 |
| `TransformerEncoder` | `num_sites, token_size, d_model, num_heads, mlp_dim, num_layers, bias, dropout` | 模块 | embedding、位置参数、多层 encoder 和最终 LayerNorm |
| `TransformerEncoder.forward(x, gamma)` | 配置和横场 | `(B, num_tokens, d_model)` | 输出编码后的 token 表示 |
| `ViTWaveFunction` | `num_sites, token_size, d_model, num_heads, mlp_dim, num_layers, num_outputs, bias, dropout` | 模块 | 复数 log 波函数模型 |
| `ViTWaveFunction.encode(x, gamma)` | 配置和横场 | token 表示 | 调用 encoder |
| `ViTWaveFunction.pooled_features(x, gamma)` | 配置和横场 | `(B, d_model)` | 对 token 维度求和池化 |
| `ViTWaveFunction.log_wavefunction_matrix(x, gamma)` | 配置和横场 | `(B, num_outputs)` 复张量 | 输出复数 log 波函数 |
| `ViTWaveFunction.forward(x, gamma)` | 配置和横场 | `(B, num_outputs)` 复张量 | 等价于 `log_wavefunction_matrix` |
| `ViTWaveFunction.wavefunction_matrix(x, gamma)` | 配置和横场 | `(B, num_outputs)` 复张量 | 返回 `exp(log_wavefunction_matrix)` |
| `ViTWaveFunction.complex_components(x, gamma)` | 配置和横场 | 复数 log 波函数 | 兼容旧接口的别名 |
| `ViTWaveFunction.phi(x, gamma)` | 配置和横场 | 波函数 | `wavefunction_matrix` 的别名 |
| `ViTWaveFunction.log_phi(x, gamma)` | 配置和横场 | log 波函数 | `log_wavefunction_matrix` 的别名 |

## 输出维度说明

若 `num_sites=10`、`token_size=5`、`d_model=16`、`num_outputs=2`：

1. 输入 `x` 形状为 `(B, 10)`。
2. token 化后形状为 `(B, 2, 5)`。
3. 拼接 `gamma` 后形状为 `(B, 2, 6)`。
4. 线性投影后形状为 `(B, 2, 16)`。
5. Transformer encoder 输出形状为 `(B, 2, 16)`。
6. token 求和池化后形状为 `(B, 16)`。
7. 实部和虚部 head 输出形状均为 `(B, 2)`。
8. 最终复数 log 波函数形状为 `(B, 2)`。

## 注意事项

- `LinearTokenEmbedding` 要求 `num_sites % token_size == 0`。
- `MultiHeadSelfAttention` 要求 `d_model % num_heads == 0`。
- `ViTWaveFunction.forward` 返回的是复数 log 波函数，不是指数后的波函数；需要物理波函数时使用 `wavefunction_matrix` 或 `phi`。

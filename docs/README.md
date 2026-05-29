# NQS_VMC 代码说明文档索引

本目录为项目源码的说明文档，覆盖当前工作区中的 4 个 Python 源码文件。重点包含脚本用途、运行方式、输入参数设置、主要类和函数说明、输出文件说明。

## 源码与文档对应关系

| 源码文件 | 说明文档 | 主要用途 |
|---|---|---|
| `SNQS.py` | [SNQS.md](SNQS.md) | 基于 MLP 的子空间 NQS，对横场 Ising 模型执行精确全基枚举、Ritz 广义本征求解和 Gamma 扫描 |
| `NQS_fermi.py` | [NQS_fermi.md](NQS_fermi.md) | 费米格点体系的 NQS-VMC，支持固定总动量扇区或所有非空扇区优化 |
| `NQS_transformer/main.py` | [NQS_transformer_main.md](NQS_transformer_main.md) | 基于 ViT 波函数的 TFIM NQS-VMC/SR 训练入口 |
| `NQS_transformer/ViT.py` | [NQS_transformer_ViT.md](NQS_transformer_ViT.md) | ViT 波函数模型、Transformer 编码器和 token embedding 模块 |

## 环境依赖

代码主要依赖：

- Python 3
- NumPy
- PyTorch
- Matplotlib

其中 `SNQS.py` 和 `NQS_transformer/main.py` 默认优先使用 CUDA；无 GPU 时会回退到 CPU，或可通过命令行参数显式指定。

## 常用运行入口

```bash
python SNQS.py
python NQS_fermi.py
python NQS_transformer/main.py
```

详细参数见各文件对应文档。

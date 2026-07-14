# StarVLA-RobustLab

本项目研究视觉语言动作模型（VLA）在视觉、语言、机器人状态与动作执行扰动下的鲁棒性，并探索基于动作预测分歧的风险感知动态 action horizon。

当前状态：**Phase 0 — 上游审计与环境预检**。尚未产生训练或评测结果，所有实验指标均为 `TBD`。

## 学习主线

每一阶段都按“概念 → 源码调用链 → 最小实验 → 验收问题”推进：

1. VLA 基础：理解 observation、VLM hidden state、action head、action chunk。
2. 推理闭环：理解 policy server、LIBERO client、动作反归一化和环境执行。
3. 训练基础：理解监督动作回归、冻结/解冻、DDP 与 DeepSpeed ZeRO。
4. 鲁棒性评测：理解可复现扰动、paired seeds、置信区间和失败分类。
5. 动态执行：理解动作分歧、阈值校准及性能—计算成本权衡。
6. 研究表达：通过消融、失败分析和局限性形成可信的简历材料。

## 当前机器与正式实验环境

当前 Windows 工作站适合源码学习、框架开发与 CPU 单元测试，但不满足项目说明书中的正式训练假设。真实训练与 LIBERO GPU 评测计划在 Linux/CUDA 环境执行；具体差异见 `reports/upstream_audit.md`。

## Phase 0 使用

在 Git Bash、WSL 或 Linux 中执行：

```bash
bash scripts/preflight.sh
```

严格模式（正式训练机验收，任一关键项缺失即失败）：

```bash
STRICT=1 EXPECTED_GPU_COUNT=8 MIN_GPU_MEMORY_MIB=38000 bash scripts/preflight.sh
```

项目范围和阶段验收标准见 `PROJECT_SPEC.md`。


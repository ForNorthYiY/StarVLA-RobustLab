# StarVLA-RobustLab 项目开发说明书

> 面向 Codex 的可执行项目指导文档  
> 项目定位：基于 StarVLA 的完整 VLA 训练、评测与方法改进项目  
> 计算资源：单机 8 × RTX 4090 40GB  
> 推荐主线：Qwen3-VL-4B + StarVLA-OFT + LIBERO + 鲁棒性评测 + 风险感知动态执行

---

## 0. 文档用途

本文件同时承担四个角色：

1. **项目需求文档**：说明要做什么、为什么做、做到什么程度。
2. **Codex 总任务说明**：可直接作为 Codex 的长期上下文。
3. **实验执行手册**：规定训练、评测、消融和记录方式。
4. **简历项目验收标准**：确保最终成果不是“运行了官方代码”，而是具备个人贡献、实验结论和工程完整性。

Codex 在开始任何修改前，必须先阅读本文件、当前仓库 `README.md`、StarVLA 官方 Quick Start、目标配置文件以及将要修改的源码。不得根据旧版本教程臆造目录、类名或配置字段。

---

# 1. 项目概述

## 1.1 项目名称

**StarVLA-RobustLab：面向视觉语言动作模型的鲁棒性评测与风险感知动态执行系统**

英文名称：

**StarVLA-RobustLab: Robustness Evaluation and Risk-Aware Dynamic Execution for Vision-Language-Action Models**

## 1.2 项目目标

以 StarVLA 为训练和部署底座，在 LIBERO 中完成以下闭环：

```text
数据准备
  → VLA 模型训练
  → Checkpoint 加载
  → Policy Server / 仿真推理
  → 标准环境评测
  → 扰动环境评测
  → 失败诊断
  → 风险感知动态执行改进
  → 消融实验
  → 技术报告、视频和简历材料
```

项目不能止步于复现。最终必须至少包含两项个人工作：

- 一套模块化、渐进式的 VLA 鲁棒性评测工具；
- 一个基于动作预测分歧的风险感知动态执行模块。

## 1.3 核心研究问题

### RQ1：不同类型扰动如何影响 VLA？

研究视觉、相机、语言、机器人状态和动作执行扰动对成功率、动作平滑度及失败阶段的影响。

### RQ2：VLM 微调范围如何影响 clean 性能与鲁棒性？

对比：

- 仅训练 action head；
- 部分解冻 VLM；
- 全参数微调。

### RQ3：固定 action horizon 是否适合所有状态？

研究模型预测稳定时长动作块、模型不确定时短动作块或立即重规划，是否能改善扰动环境成功率。

### RQ4：鲁棒性提升带来多少计算代价？

同时报告：

- 成功率；
- policy forward 次数；
- 平均推理延迟；
- 每条轨迹总耗时；
- 动作平滑度。

---

# 2. 技术背景与上游事实

StarVLA 将 VLA 抽象为可替换的 foundation-model backbone 与 action head，并统一训练和评测接口。当前官方仓库列出的主要动作生成范式包括：

- `QwenOFT`：MLP 并行连续动作回归；
- `QwenFAST`：自回归离散动作 token；
- `QwenPI_v3`：Flow Matching 动作专家；
- `QwenGR00T`：VLM System 2 + Flow Matching System 1。

官方 Quick Start 使用 LeRobot 格式的 LIBERO 数据，并给出 `Qwen3-VL-4B-Instruct`、7 维 LIBERO 动作、action horizon 8 等示例。StarVLA 的公开接口强调：

```python
framework.forward(raw_batch)
framework.predict_action(raw_observation)
```

其中数据加载器应输出模型无关的原始字典，例如图像、语言、动作和可选状态。

**注意：StarVLA 正在快速迭代。** 本文中的字段和命令是项目设计基线，不是让 Codex 盲目覆盖仓库。Codex 必须以当前 checkout 的实际代码为准，若发现差异，先形成审计报告，再做兼容性修改。

---

# 3. 硬件与训练策略

## 3.1 硬件假设

```text
GPU: 8 × RTX 4090 40GB
单卡显存: 40GB
总物理显存: 320GB
部署形式: 单机多卡
推荐系统: Ubuntu 22.04
本地磁盘: 建议 NVMe，预留至少 1TB
```

分布式训练中 320GB 并不是一个统一显存池。DDP 会复制模型，ZeRO-2/3 才会切分优化器状态、梯度或参数。

## 3.2 推荐主模型

```text
Backbone: Qwen3-VL-4B-Instruct
Framework: QwenOFT
Benchmark: LIBERO
Action dimension: 7
Action horizon: 8
Precision: BF16
Attention: Flash Attention 2
Distributed: Accelerate + DeepSpeed
```

OFT 作为第一主线的原因：

- 架构和损失简单，便于真正理解 VLA 数据流；
- 推理快，适合进行大量鲁棒性 rollout；
- 方便实现动作分歧、动态执行和失败分析；
- 比 Flow Matching 更容易定位训练问题。

## 3.3 训练层级

### T0：只训练 Action Head

- 冻结整个 Qwen3-VL；
- 训练 OFT action head；
- 用于管线验证和最低成本基线。

### T1：部分解冻

- 训练 action head；
- 解冻多模态投影层；
- 解冻 VLM 最后 2～4 层；
- 主实验优先选择此版本。

### T2：全参数微调

- 采用 DeepSpeed ZeRO-3；
- 作为研究对照，不应成为第一轮训练；
- 若性能提升有限，可形成“部分解冻更具性价比”的有效结论。

## 3.4 初始训练参数

以下仅为 smoke test 起始配置，必须通过实测显存和吞吐调整：

```yaml
precision: bf16
per_device_batch_size: 2
gradient_accumulation_steps: 1
gradient_checkpointing: true
action_head_lr: 1.0e-4
vlm_lr: 5.0e-6
weight_decay: 0.01
max_grad_norm: 1.0
action_dim: 7
action_horizon: 8
```

训练策略：

1. T0 从 ZeRO-2 开始；
2. T1 优先 ZeRO-2，OOM 或显存不均衡时切 ZeRO-3；
3. T2 使用 ZeRO-3，暂不启用 CPU offload；
4. 先做 20 step、200 step，再做正式训练；
5. 每次改变 batch、分辨率或 ZeRO stage 都记录吞吐和峰值显存。

---

# 4. 项目边界

## 4.1 必须完成

- 官方 checkpoint 的 LIBERO 推理闭环；
- 至少一个 LIBERO suite 的自主训练；
- 至少三类扰动，每类三个强度等级；
- 统一评测和结果保存；
- 风险感知动态 action horizon；
- 至少四组核心消融；
- 可复现 README；
- 技术报告和演示视频；
- 简历项目描述。

## 4.2 建议完成

- OFT 与 Flow Matching 小规模对照；
- 多 suite 联合训练；
- 自动失败阶段标注工具；
- W&B 实验面板；
- 单元测试和集成测试；
- 将个人扩展以独立 package 或清晰 patch 层接入 StarVLA。

## 4.3 第一版不做

- 从零预训练 VLM；
- 同时复现 FAST、OFT、PI、GR00T 四条路线；
- 在没有 clean baseline 的情况下直接加入 RL；
- 宣称动态 action chunk 是完全原创方法；
- 为追求数量而训练所有 benchmark；
- 对真实机器人部署做无根据承诺。

---

# 5. 代码仓库设计

## 5.1 推荐结构

```text
StarVLA-RobustLab/
├── README.md
├── PROJECT_SPEC.md                 # 本文档副本
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
├── third_party/
│   └── starVLA/                    # 固定 commit 的上游仓库
├── robustlab/
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py
│   │   └── validation.py
│   ├── perturbations/
│   │   ├── base.py
│   │   ├── visual.py
│   │   ├── camera.py
│   │   ├── language.py
│   │   ├── proprioception.py
│   │   ├── action_noise.py
│   │   └── registry.py
│   ├── execution/
│   │   ├── base.py
│   │   ├── fixed_horizon.py
│   │   ├── replan_every_step.py
│   │   ├── disagreement.py
│   │   └── risk_aware_horizon.py
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── rollout.py
│   │   ├── metrics.py
│   │   ├── result_schema.py
│   │   └── failure_taxonomy.py
│   ├── adapters/
│   │   ├── starvla_policy.py
│   │   └── libero_adapter.py
│   ├── logging/
│   │   ├── jsonl_logger.py
│   │   └── wandb_logger.py
│   └── visualization/
│       ├── robustness_curve.py
│       ├── heatmap.py
│       ├── action_plot.py
│       └── video_overlay.py
├── configs/
│   ├── train/
│   │   ├── oft_head_only.yaml
│   │   ├── oft_partial_unfreeze.yaml
│   │   └── oft_full_finetune.yaml
│   ├── eval/
│   │   ├── clean.yaml
│   │   ├── visual_noise.yaml
│   │   ├── camera_shift.yaml
│   │   ├── language_paraphrase.yaml
│   │   └── combined.yaml
│   └── execution/
│       ├── fixed_8.yaml
│       ├── fixed_4.yaml
│       ├── replan_1.yaml
│       └── risk_aware.yaml
├── scripts/
│   ├── preflight.sh
│   ├── setup_starvla.sh
│   ├── smoke_test.sh
│   ├── train_oft.sh
│   ├── eval_clean.sh
│   ├── eval_robustness.sh
│   ├── run_ablation.sh
│   └── build_report.py
├── tests/
│   ├── test_perturbations.py
│   ├── test_metrics.py
│   ├── test_execution_policy.py
│   └── test_result_schema.py
├── runs/                           # gitignore
├── checkpoints/                    # gitignore
├── data/                           # gitignore，使用软链接
├── assets/
│   ├── architecture/
│   ├── figures/
│   └── videos/
└── reports/
    ├── experiment_log.md
    └── technical_report.md
```

## 5.2 上游代码管理原则

Codex 必须遵循：

1. `third_party/starVLA` 固定具体 commit；
2. 不在上游默认分支直接开发；
3. 创建项目分支，如 `feat/robustlab-integration`；
4. 优先通过 adapter、配置和最小 patch 扩展；
5. 必须修改上游时，将 patch 按功能拆分并记录原因；
6. 不复制整个 StarVLA 源码到个人模块；
7. 不把数据、模型权重和实验输出提交 Git。

建议记录：

```bash
git -C third_party/starVLA rev-parse HEAD > STARVLA_COMMIT.txt
```

---

# 6. 阶段化开发计划

每个阶段只有在验收通过后才能进入下一阶段。

## Phase 0：上游审计与环境预检

### 目标

确认实际 StarVLA 版本、目录、配置字段和运行入口。

### Codex 任务

1. 阅读：
   - 根目录 README；
   - `docs/starVLA_guideline.md`；
   - LIBERO 示例目录；
   - `QwenOFT.py`；
   - 训练入口；
   - 数据加载器；
   - DeepSpeed 配置。
2. 输出 `reports/upstream_audit.md`，必须包括：
   - commit hash；
   - 当前分支；
   - Python、PyTorch、CUDA、Transformers、Flash Attention 版本；
   - QwenOFT 类与入口；
   - 训练命令；
   - 评测命令；
   - 配置字段实际名称；
   - checkpoint 格式；
   - 已知风险和版本差异。
3. 编写 `scripts/preflight.sh`。

### Preflight 最少检查

```text
nvidia-smi
nvidia-smi topo -m
nvcc -V
python version
PyTorch version
PyTorch CUDA runtime
GPU count and memory
BF16 availability
transformers version
flash-attn import
accelerate version
deepseed/deepspeed version
磁盘剩余空间
共享内存大小
```

### 验收

- 8 张 GPU 均可见；
- 单卡显示约 40GB；
- BF16 可用；
- Flash Attention 可导入；
- audit 中没有未经确认的路径或类名。

---

## Phase 1：官方 Checkpoint 推理闭环

### 目标

在不训练的情况下跑通：

```text
LIBERO observation
  → StarVLA policy
  → action chunk
  → environment execution
  → result/video
```

### 必须记录

每个 episode 保存：

```json
{
  "run_id": "...",
  "model_id": "...",
  "checkpoint": "...",
  "task_suite": "libero_spatial",
  "task_id": 0,
  "episode_id": 0,
  "seed": 0,
  "success": true,
  "steps": 123,
  "policy_calls": 16,
  "mean_inference_ms": 42.1,
  "max_inference_ms": 48.5,
  "execution_strategy": "fixed_8",
  "perturbation": null,
  "video_path": "..."
}
```

### 验收

- 至少 20 个 episode；
- 有成功和失败视频；
- checkpoint 可重复加载；
- action shape、状态 shape 和归一化方式被明确记录；
- 相同 seed 在允许误差范围内可复现。

---

## Phase 2：OFT 训练基线

### 目标

独立完成至少一个 LIBERO suite 的训练。

### 训练顺序

#### 2A. 单卡模型 smoke test

- 构造 fake batch；
- 跑 `forward()`；
- 检查 loss 为有限数；
- 检查 action 输出 shape；
- 检查指定模块冻结状态。

#### 2B. 两卡 20 step

验证：

- DistributedSampler 或等效逻辑有效；
- 不同 rank 不读取完全相同 batch；
- loss 正常；
- checkpoint 能保存和加载。

#### 2C. 八卡 200 step

记录：

- 每卡峰值 allocated/reserved memory；
- steps/s；
- samples/s；
- GPU 利用率；
- 数据加载时间；
- 通信开销；
- loss 曲线。

#### 2D. 正式训练

先从 `libero_spatial` 或 `libero_goal` 开始。

### 基线实验

| 实验 | VLM 设置 | 目的 |
|---|---|---|
| T0 | 全冻结，仅 action head | 管线与最低成本基线 |
| T1 | 解冻最后 2 层 + projector | 小规模适配 |
| T2 | 解冻最后 4 层 + projector | 主候选模型 |
| T3 | 全参数微调 | 上限与成本对照 |

### 验收

- 至少一个自训练 checkpoint 能在 LIBERO 推理；
- clean success rate 显著高于随机策略；
- 训练日志包含完整配置、commit、seed 和硬件；
- 能回答图像、语言、状态如何进入模型，动作如何输出和反归一化。

---

## Phase 3：鲁棒性评测框架

### 目标

构建与模型解耦的 perturbation pipeline。

## 3.1 统一扰动接口

```python
class Perturbation(Protocol):
    name: str

    def apply(self, observation, *, rng, severity: int):
        """返回新 observation，不允许原地修改输入。"""
```

每个扰动必须：

- 可配置；
- 可指定 seed；
- 不修改原始 observation；
- 记录实际采样参数；
- 支持 severity 0/1/2/3；
- severity 0 等价于 clean。

## 3.2 第一版扰动

### A. 视觉扰动

- brightness；
- contrast；
- Gaussian noise；
- blur；
- random occlusion。

示例强度：

```text
Gaussian sigma: 5 / 15 / 30
Blur kernel: 3 / 5 / 9
Brightness factor: 0.85 / 0.65 / 0.45
Occlusion ratio: 5% / 12% / 20%
```

### B. 相机/图像几何扰动

优先先在输入图像层实现可控近似：

- horizontal shift；
- vertical shift；
- small rotation；
- crop + resize。

后续再扩展到仿真器真实相机位姿变化，并将两者区分命名为：

- `image_geometry_*`；
- `sim_camera_pose_*`。

### C. 语言扰动

- 同义改写；
- 礼貌/冗余表达；
- 词序变化；
- 目标描述扩展。

语言改写必须保证任务语义不变。第一版使用人工维护 paraphrase 表，不要在正式评测时在线调用语言模型，以保证复现。

### D. 状态扰动

- proprioception Gaussian noise；
- 初始状态小偏移；
- gripper state noise。

### E. 动作执行扰动

- action Gaussian noise；
- action delay；
- 随机丢弃一次动作；
- scale bias。

动作扰动应在环境执行前施加，而不是改模型输出日志中的原始预测。

## 3.3 结果组织

```text
results/
└── <model_id>/
    └── <suite>/
        └── <perturbation_name>/
            └── severity_<n>/
                ├── episodes.jsonl
                ├── aggregate.json
                └── videos/
```

### 验收

- 至少三类扰动；
- 每类三个 severity；
- 每个设置至少 20 个 episode（正式结果建议 50）；
- 可从单个 YAML 一键运行；
- clean 与 severity 0 结果一致；
- 所有随机量有 seed。

---

## Phase 4：风险感知动态执行

## 4.1 动机

固定执行完整 action chunk 的优点是减少推理次数，但在模型输出不可靠时，错误会持续多个控制步。每步重新规划更稳健，但计算成本高。

目标是在两者间动态权衡：

```text
预测一致 → 执行更多动作
预测分歧 → 少执行、尽快重规划
```

## 4.2 动作分歧计算

对同一 observation 生成 M 个预测：

\[
A^{(1)}, A^{(2)}, \ldots, A^{(M)} \in \mathbb{R}^{H\times D}
\]

OFT 本身通常是确定性的，可通过轻微且语义保持的 test-time augmentation 构造多个视图，例如：

- 非破坏性亮度微扰；
- 极小 crop/resize；
- 多相机视角子集；
- 若模型保留 dropout，可实验性启用 MC dropout，但必须单独标记。

仅比较前 K 步：

\[
\bar A_{1:K}=\frac{1}{M}\sum_{m=1}^{M} A^{(m)}_{1:K}
\]

\[
u_t = \frac{1}{M}\sum_{m=1}^{M}
\left\|W\odot\left(A^{(m)}_{1:K}-\bar A_{1:K}\right)\right\|_2^2
\]

其中 W 用于平衡平移、旋转与夹爪维度，避免量纲差异。

## 4.3 动态 horizon

```text
u < tau_low       → execute 8
u < tau_high      → execute 4
otherwise         → execute 1
```

必须支持配置：

```yaml
execution:
  name: risk_aware
  candidate_horizons: [1, 4, 8]
  num_predictions: 4
  compare_prefix: 3
  tau_low: null
  tau_high: null
  calibration_file: null
```

## 4.4 阈值校准

不得直接在测试集上手工挑最好阈值。

建议流程：

1. 在 calibration episodes 收集 `u_t`；
2. 分析 clean 成功轨迹、clean 失败轨迹和扰动轨迹分布；
3. 用分位数或验证集网格搜索选择阈值；
4. 固定阈值后评测 test episodes。

## 4.5 必须对照

| 策略 | 描述 |
|---|---|
| E8 | 每次预测，固定执行 8 步 |
| E4 | 固定执行 4 步 |
| E1 | 每步重新规划 |
| ER | 风险感知执行 1/4/8 步 |

## 4.6 验收

- ER 在至少一种扰动下优于 E8；
- 同时报告相对 E1 节省的 policy calls；
- 不允许只挑有效任务汇报；
- 输出不确定性与失败概率的关系图；
- 对无提升或负提升的扰动给出分析。

---

## Phase 5：正式实验矩阵

## 5.1 微调策略

| ID | 模型 | 可训练模块 |
|---|---|---|
| F0 | Qwen3-VL-4B + OFT | action head |
| F1 | Qwen3-VL-4B + OFT | action head + projector + last 2 blocks |
| F2 | Qwen3-VL-4B + OFT | action head + projector + last 4 blocks |
| F3 | Qwen3-VL-4B + OFT | full model |

## 5.2 数据增强

| ID | 视觉增强 | 状态噪声 | 语言改写 |
|---|---:|---:|---:|
| R0 | × | × | × |
| R1 | ✓ | × | × |
| R2 | ✓ | ✓ | × |
| R3 | ✓ | ✓ | ✓ |

## 5.3 执行策略

| ID | Horizon |
|---|---|
| E8 | 固定 8 |
| E4 | 固定 4 |
| E1 | 固定 1 |
| ER | 风险感知 1/4/8 |

## 5.4 最小核心结果

必须完成：

```text
Baseline = F0 + R0 + E8
Main     = F_best + R_best + ER
```

并至少包含：

1. `F0/F1/F2` 微调范围对比；
2. `R0/R3` 数据增强对比；
3. `E8/E4/E1/ER` 执行策略对比；
4. clean 与三种扰动的结果；
5. 计算开销对比。

## 5.5 可选动作头对比

资源允许后加入：

```text
QwenOFT vs QwenPI_v3
```

只在相同 suite、数据、训练预算和评测协议下比较。不要因为模型范式不同而使用不一致的数据处理。

---

# 7. 指标定义

## 7.1 成功率

\[
SR = \frac{N_{success}}{N_{episodes}}
\]

## 7.2 成功率下降

\[
\Delta SR = SR_{clean} - SR_{perturbed}
\]

## 7.3 鲁棒性保持率

\[
RR = \frac{SR_{perturbed}}{\max(SR_{clean}, \epsilon)}
\]

## 7.4 扰动曲线面积

对于 severity 0～K：

\[
AURC = \frac{1}{K+1}\sum_{k=0}^{K} SR_k
\]

需要在报告中明确：这里的 AURC 是项目自定义的 robustness curve 平均值，不应与其他领域同名指标混淆。

## 7.5 动作平滑度

使用二阶差分：

\[
J_{smooth}=\frac{1}{T-2}\sum_{t=2}^{T-1}
\left\|a_{t+1}-2a_t+a_{t-1}\right\|_2
\]

分别报告：

- translation smoothness；
- rotation smoothness；
- gripper switching count。

## 7.6 效率

- mean/p95 inference latency；
- policy calls per episode；
- environment steps per episode；
- wall-clock seconds per successful episode；
- max GPU memory；
- training samples/s。

## 7.7 统计要求

正式结果：

- 每个设置至少 50 episodes，条件允许时 100；
- 使用相同 seed 集合做 paired comparison；
- 报告 bootstrap 95% confidence interval；
- 多随机种子训练时报告均值和标准差；
- 不以单次最好 checkpoint 代替完整模型选择协议。

---

# 8. 失败诊断

## 8.1 失败分类

```text
F1 目标物体选择错误
F2 场景/空间位置判断错误
F3 接近轨迹失败
F4 夹爪对准失败
F5 抓取失败
F6 抓取后掉落
F7 搬运碰撞或偏离
F8 放置位置错误
F9 长时序子任务顺序错误
F10 动作块切换不连续
F11 语言理解错误
F12 超时
F13 仿真或系统错误（不计入模型失败，单独报告）
```

## 8.2 标注规范

- 每条失败轨迹允许一个 primary cause 和最多两个 secondary causes；
- 系统错误不得混入模型成功率；
- 自动规则只能给建议标签，最终抽样人工复核；
- 至少人工分析 100 条失败轨迹或全部失败中的较小者。

## 8.3 输出

- 扰动类型 × 失败原因热力图；
- 各执行策略失败类型分布；
- 成功/失败动作轨迹对比；
- 典型案例视频。

---

# 9. 可复现性与日志

每次运行必须保存：

```text
project git commit
StarVLA commit
config snapshot
command line
hostname
GPU model and count
CUDA/PyTorch/Transformers versions
seed
checkpoint hash or path
数据集名称和版本
开始/结束时间
stdout/stderr log
W&B run ID（若启用）
```

建议每个 run 目录：

```text
runs/<run_id>/
├── config_resolved.yaml
├── command.txt
├── environment.txt
├── git_state.txt
├── metrics.jsonl
├── summary.json
├── checkpoints/
└── logs/
```

不得只依赖 W&B 云端记录，必须保留本地可恢复日志。

---

# 10. 测试要求

## 10.1 单元测试

### 扰动测试

- severity 0 保持输入不变；
- 相同 seed 输出相同；
- 输入未被原地修改；
- 图像 dtype/range/shape 合法；
- 状态和动作 shape 不变。

### 指标测试

使用人工构造数据验证：

- success rate；
- robustness retention；
- smoothness；
- bootstrap CI；
- latency aggregation。

### 动态执行测试

- 低分歧返回 horizon 8；
- 中分歧返回 4；
- 高分歧返回 1；
- NaN/Inf 分歧安全回退为 1；
- 候选 horizon 不超过预测 chunk 长度。

## 10.2 集成测试

- mock policy + mock env 完整 rollout；
- StarVLA adapter 输出 shape 检查；
- 结果 JSON schema 校验；
- 中断后可从已完成 episode 继续；
- 多进程写结果不会相互覆盖。

## 10.3 训练测试

- 仅 action head 时，VLM 梯度必须为空；
- 部分解冻时，仅目标层有梯度；
- 每个 rank 的 batch 有差异；
- checkpoint 加载后相同输入输出接近；
- loss 为 NaN 时自动保存诊断 batch，而不是静默继续。

---

# 11. Codex 工作规则

Codex 必须遵守以下规则：

1. **先读后改**：修改前列出相关文件和数据流。
2. **小步提交**：一个 commit 只做一个逻辑功能。
3. **不臆造 API**：不确定时搜索当前仓库。
4. **不删除上游行为**：新增配置默认不改变官方 baseline。
5. **不硬编码路径**：路径进入 YAML 或环境变量。
6. **不硬编码 GPU 数量**：训练和评测均从参数读取。
7. **每个功能带测试**。
8. **失败要显式**：使用清晰异常，不吞掉错误。
9. **先 smoke test 再大规模运行**。
10. **不伪造实验数字**：没有运行结果时使用 `TBD`。
11. **不自动开始昂贵正式训练**：先生成命令和检查项，由用户执行或明确授权。
12. **不要重写整个上游文件**：尽量最小 patch。

---

# 12. 可直接交给 Codex 的总提示词

下面内容可作为第一次进入项目时的 Codex Prompt：

```text
你现在是 StarVLA-RobustLab 项目的主要工程助手。请先完整阅读 PROJECT_SPEC.md，以及 third_party/starVLA 中的 README、docs/starVLA_guideline.md、LIBERO 示例、QwenOFT framework、训练入口、LeRobot dataloader 和 DeepSpeed 配置。

项目硬件是单机 8×RTX 4090 40GB。主线是 Qwen3-VL-4B + QwenOFT + LIBERO，个人贡献是鲁棒性扰动评测和基于动作预测分歧的风险感知动态 action horizon。

本轮只执行 Phase 0，不要开始正式训练，也不要大范围修改源码。请完成：
1. 审计当前 StarVLA commit、分支、目录和实际配置字段；
2. 梳理 observation → forward/predict_action → action chunk → LIBERO execution 的调用链；
3. 创建 reports/upstream_audit.md；
4. 创建 scripts/preflight.sh；
5. 列出环境安装和 smoke test 命令；
6. 为所有新增脚本提供错误检查；
7. 运行不昂贵的静态检查或单元测试；
8. 最后汇报修改文件、验证结果、尚未解决的问题和下一步命令。

严格要求：不得臆造路径/API；不得伪造实验结果；不得修改上游默认行为；不得执行耗时训练。
```

---

# 13. 分阶段 Codex 提示词

## 13.1 Phase 1 Prompt

```text
依据 PROJECT_SPEC.md 执行 Phase 1：官方 checkpoint 的 LIBERO 推理闭环。

先验证 upstream_audit.md 仍与当前 commit 一致。实现或整理一个最小可复现的评测入口，保存每个 episode 的 JSONL、聚合指标和视频。必须记录 action/state shape、归一化方式、checkpoint 路径、seed、推理延迟和 policy call 次数。

先运行最小 smoke test，不要直接跑大规模评测。若 checkpoint 或依赖缺失，输出明确下载/安装命令，不得用假数据冒充真实 LIBERO 结果。完成后更新 README 和 reports/experiment_log.md。
```

## 13.2 Phase 2 Prompt

```text
依据 PROJECT_SPEC.md 执行 Phase 2：QwenOFT 训练基线。

先实现 head-only 配置，不改变官方配置。必须自动验证冻结模块、可训练参数量、梯度状态和输出 shape。依次准备单卡 fake-batch smoke test、两卡 20-step 测试和八卡 200-step 测试命令。不要直接启动正式长训练。

为每次训练保存 resolved config、环境版本、git commit、峰值显存、吞吐和 loss。发现当前 StarVLA 与项目文档字段不一致时，以当前源码为准并更新 audit，不得猜测。
```

## 13.3 Phase 3 Prompt

```text
依据 PROJECT_SPEC.md 实现 robustlab/perturbations 和统一鲁棒性 evaluator。

先实现 visual Gaussian noise、image horizontal shift、language paraphrase、proprioception noise 四类扰动，每类支持 severity 0/1/2/3、独立 seed、参数记录和非原地修改。实现 registry 和 YAML 配置解析。

同时编写单元测试：severity 0 identity、seed reproducibility、shape/dtype/range、no in-place mutation。评测输出使用结构化 JSONL，支持中断续跑和按 model/suite/perturbation/severity 分目录保存。
```

## 13.4 Phase 4 Prompt

```text
依据 PROJECT_SPEC.md 实现风险感知动态 action horizon。

先定义独立 ExecutionPolicy 接口，再实现 FixedHorizon、ReplanEveryStep 和 RiskAwareHorizon。RiskAwareHorizon 对同一 observation 通过可配置的轻量 test-time augmentations 得到多个 action chunks，计算带维度权重的 prefix disagreement，并输出 horizon 1/4/8。

必须支持阈值校准文件、NaN 安全回退、详细诊断日志和单元测试。不要在测试集上硬编码最优阈值。保持 StarVLA 模型代码不变，优先在执行层实现。
```

## 13.5 正式实验 Prompt

```text
根据 PROJECT_SPEC.md 中 Phase 5 的实验矩阵生成正式实验计划和运行脚本，但在运行前先估算 episode 数、GPU 分配、预计输出文件数和磁盘需求。

确保所有对照使用相同任务、seed 和评测协议。实现 8 GPU 并行评测时，每个进程使用独立端口、独立结果目录和确定性 seed 切片。聚合脚本必须检查缺失 episode、重复 episode 和系统错误。

不得选择性丢弃失败结果。生成最终表格、置信区间、鲁棒性曲线、热力图、动作平滑度图和典型失败案例清单。
```

---

# 14. 八卡评测调度建议

训练时使用全部 8 卡。评测时优先使用“每卡一个独立模型服务 + 独立 LIBERO client”进行任务切片。

示例：

```text
GPU 0: clean / seed group 0
GPU 1: visual severity 1
GPU 2: visual severity 2
GPU 3: visual severity 3
GPU 4: camera perturbations
GPU 5: language perturbations
GPU 6: state/action perturbations
GPU 7: combined perturbations
```

更严谨的方式是按 episode shard，而不是让 GPU 与扰动永久绑定，以避免某张卡负载极不均衡。

每个 worker 必须拥有：

- 唯一 `CUDA_VISIBLE_DEVICES`；
- 唯一 server port；
- 唯一 worker ID；
- 不重叠 episode IDs；
- 可合并的结果 schema。

---

# 15. 项目里程碑

## M1：可运行

- 官方 checkpoint 在 LIBERO 上运行；
- 有 rollout 视频和结构化日志。

## M2：可训练

- 自训练 OFT checkpoint 可评测；
- 冻结/部分解冻逻辑验证通过。

## M3：可研究

- 鲁棒性评测框架完成；
- 有 clean-to-perturbed degradation 曲线。

## M4：有个人方法

- 风险感知动态 horizon 完成；
- 有 E8/E4/E1/ER 对照。

## M5：可写简历

- 有真实数值；
- 有完整消融；
- 有 GitHub README；
- 有技术报告和演示视频；
- 能解释失败和局限。

---

# 16. 最终交付物

```text
1. GitHub 仓库
2. 可复现安装与运行 README
3. 环境审计报告
4. 训练配置与日志
5. 自训练 checkpoint（可只提供下载说明）
6. 鲁棒性评测模块
7. 风险感知动态执行模块
8. 自动聚合和绘图脚本
9. Clean/扰动/消融结果表
10. 至少 20 个精选 rollout 视频
11. 6～10 页技术报告
12. 2～3 分钟演示视频
13. 中英文简历描述
14. 面试讲解材料
```

---

# 17. 简历描述模板

只有在获得真实实验结果后才替换方括号。

## 中文版

**StarVLA-RobustLab：视觉语言动作模型鲁棒性评测与风险感知执行系统**

- 基于 StarVLA、Qwen3-VL-4B 与 LIBERO 搭建 VLA 数据处理、分布式微调、策略服务和仿真评测闭环，在 8×RTX 4090 40GB 上完成 OFT 连续动作策略的多卡训练与部署。
- 设计相机/视觉、语言、机器人状态与动作执行的渐进式扰动框架，构建成功率、鲁棒性保持率、动作平滑度、推理延迟和失败阶段等指标，并实现可复现的并行 rollout 与自动报告工具。
- 实现基于多视图动作预测分歧的风险感知动态 action horizon，在 LIBERO-[suite] 的扰动环境下将平均成功率由 [A]% 提升至 [B]%，相较逐步重规划减少 [C]% policy calls，并完成冻结范围、数据增强和执行策略消融。

## English Version

**StarVLA-RobustLab: Robustness Evaluation and Risk-Aware Execution for Vision-Language-Action Models**

- Built an end-to-end VLA pipeline on StarVLA, Qwen3-VL-4B, and LIBERO, covering LeRobot data processing, distributed OFT policy fine-tuning, policy serving, and simulation evaluation on 8×RTX 4090 40GB GPUs.
- Developed a progressive robustness benchmark for visual/camera, language, proprioceptive, and action-execution perturbations, with reproducible parallel rollouts and metrics for success, robustness retention, action smoothness, latency, and failure stages.
- Implemented disagreement-based risk-aware dynamic action horizons; improved perturbed-environment success from [A]% to [B]% on LIBERO-[suite] while reducing policy calls by [C]% versus per-step replanning, supported by ablations on fine-tuning scope, augmentation, and execution strategies.

---

# 18. 面试时必须能回答的问题

1. StarVLA 为什么适合作为底座，而不是项目本身的创新？
2. QwenOFT 如何从 VLM hidden states 得到连续 action chunk？
3. `future_action_window_size` 和 `action_horizon` 在当前版本中如何定义？
4. 数据集中的动作如何归一化，推理时如何反归一化？
5. 为什么固定 horizon 8 可能在扰动下失败？
6. OFT 是确定性模型，你如何构造动作不确定性？
7. 动作各维量纲不同，为什么不能直接计算未加权方差？
8. 动态重规划成功率提高是否只是因为调用模型更多？
9. 如何公平比较 E1 与 ER 的计算成本？
10. 为什么不能在测试集调阈值？
11. 哪类扰动最伤害模型？具体失败发生在哪个阶段？
12. 部分解冻为何可能比全参数微调更好？
13. 8×40GB 下为什么仍需 ZeRO？
14. 你如何证明多卡训练没有重复读取相同数据？
15. 项目目前距离真实机器人部署还缺什么？

---

# 19. 风险与备选方案

## 风险 A：Qwen3-VL-4B 训练不稳定或官方配置变化

备选：

- 固定可运行 commit；
- 使用官方已验证 checkpoint；
- 先用 Qwen2.5-VL-3B 或 Florence-2 做管线验证；
- 保留主实验回到 Qwen3-VL-4B。

## 风险 B：全参数微调成本过高

备选：

- 将 T2 部分解冻作为主模型；
- 全参数实验缩短训练预算，定位为成本对照；
- 不为了“全参”牺牲完整评测。

## 风险 C：动态 horizon 没有提升

这是有效研究结果。进一步检查：

- 分歧是否与失败概率相关；
- test-time augmentation 是否破坏语义；
- 阈值是否在独立验证集校准；
- 风险是否主要来自错误目标理解，而非局部控制；
- 按任务阶段或动作维度重新定义分歧。

可将最终贡献转为：

> 系统评估了动作分歧作为 VLA 风险指标的适用边界，并识别其对局部控制错误有效、对语义目标错误无效。

## 风险 D：仿真评测速度慢

备选：

- 8 卡并行 worker；
- 先做 20 episode 筛选，正式配置再做 50～100；
- 缓存任务和模型初始化；
- 将视频保存限制为成功/失败代表样本，所有 episode 仍保留数值日志。

---

# 20. 参考资料

1. StarVLA 官方仓库：`https://github.com/starVLA/starVLA`
2. StarVLA Quick Start：仓库中的 `docs/starVLA_guideline.md`
3. StarVLA 技术报告：arXiv `2604.05014`
4. RLinf StarVLA + GRPO 示例：RLinf Documentation 中的 “RL on StarVLA Models”

由于 StarVLA 处于活跃开发期，正式运行前必须重新核对官方仓库最新文档，并把实际 commit 固定到项目记录中。

---

# 21. 第一条执行指令

将本文件保存到项目根目录为 `PROJECT_SPEC.md`，然后把第 12 节“总提示词”发送给 Codex。Codex 完成 Phase 0 审计后，再进入安装、checkpoint 推理和训练，不要一次性让 Codex 同时实现所有模块。

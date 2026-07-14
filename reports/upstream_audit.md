# Phase 0 上游审计

审计日期：2026-07-14

## 1. 固定上游版本

- 仓库：`https://github.com/starVLA/starVLA`
- 分支：`starVLA`（官方说明中的稳定分支）
- 固定提交：`3422b9f2387b6f682cf02802904a77b23ab13afd`
- 本地位置：`third_party/starVLA`
- 项目仓库不直接修改上游默认行为；个人功能优先放在 `robustlab/` 与 adapter 层。

版本漂移：项目说明书要求的 `docs/starVLA_guideline.md` 在该稳定提交中不存在。实际 LIBERO 文档为 `examples/LIBERO/README.md`。开发时以固定提交源码为准。

## 2. 当前工作站预检

| 项目 | 实测 | Phase 0 结论 |
|---|---|---|
| OS | Windows | 不作为正式训练环境 |
| Python | 3.13.12 | 不建议；上游目标为 Python 3.10，部分固定依赖不支持 3.13 |
| GPU | 1 × RTX 5060 Ti | 可做有限本地实验，不满足 8 卡假设 |
| GPU 显存 | 16311 MiB | 无法按文档直接微调 Qwen3-VL-4B |
| NVIDIA driver | 576.80 | 驱动可见 |
| CUDA Toolkit / nvcc | 未找到 | 正式环境待安装/验证 |
| C 盘剩余空间 | 约 7.6 GB | 不可下载模型、数据集或保存训练产物 |
| PyTorch / CUDA runtime / BF16 | 2.12.0+cpu / `None` / False | 当前 Python 仅安装 CPU PyTorch，不可用于 GPU smoke test |
| Transformers | 4.51.3 | 与上游固定的 4.57.0 不一致 |
| FlashAttention / Accelerate / DeepSpeed | 未安装 / 1.13.0 / 未安装 | 与上游环境不一致 |

结论：当前机器用于读源码、开发鲁棒性模块和运行轻量单测。模型、LIBERO 数据和 checkpoint 不应下载到当前 C 盘。正式训练需要 Linux、充足 NVMe 空间和 CUDA GPU 环境。

## 3. 上游依赖事实

- `pyproject.toml` 声明 Python `>=3.10`，但工具配置目标为 Python 3.10。
- `requirements.txt` 固定 `transformers==4.57.0`、`accelerate==1.5.2`、`deepspeed==0.16.9`、`numpy==1.26.4`、`torchvision==0.21.0`。
- 文件未固定主 PyTorch 版本；注释给出 Qwen3.5 环境示例 `torch==2.6.0+cu124`，不能自动视为 Qwen3-VL/OFT 的唯一合法版本。
- Flash Attention 只在注释示例中给出版本，正式安装前需结合 GPU、CUDA 和 PyTorch 选择匹配 wheel/build。
- LIBERO 使用独立 Python 3.10 环境，官方脚本安装 MuJoCo、LIBERO 及 websocket 等评测依赖。

## 4. QwenOFT 实际接口与数据流

实现文件：`starVLA/model/framework/VLM4A/QwenOFT.py`。

- 实际类名：`Qwenvl_OFT`，继承 `baseframework`。
- 训练接口：`forward(examples=...) -> {"action_loss": tensor}`。
- 推理接口：`predict_action(examples=...) -> {"normalized_actions": ndarray}`。
- `action_horizon` 是动作块长度的单一事实来源；源码明确把旧字段 `future_action_window_size` / `past_action_window_size` 视为兼容别名。
- 默认动作维度为 7；LIBERO 配置中 `action_horizon: 8`。
- 模型在指令末尾加入与 chunk 长度相同的动作占位 token，取得这些 token 对应的 VLM hidden states，再由 MLP action head 并行回归 `[B, H, D]` 连续动作。
- 训练使用 L1 loss，对标签最后 `action_horizon` 步计算回归损失。

概念解释：VLM 不直接输出机器人可执行的文本 token；动作占位 token 的 hidden states 是“由图像和语言条件化后的动作查询向量”，MLP action head 再把每个查询向量映射为 7 维连续动作。

## 5. LIBERO 推理闭环

实际调用链：

```text
LIBERO obs
  -> eval_libero.py 组织双相机图像、状态、instruction
  -> ModelClient.step()
  -> WebsocketClientPolicy.predict_action()
  -> deployment/model_server/server_policy.py
  -> StarVLA framework.predict_action()
  -> 服务端使用 dataset_statistics 反归一化
  -> 返回 unnormalized action chunk
  -> 客户端按 action_chunk_size 缓存并逐步取动作
  -> 拼接 xyz + rotation + gripper
  -> env.step(action)
```

重要版本差异：旧设计可能在 LIBERO client 中加载 `dataset_statistics.json` 并反归一化；当前固定提交已把反归一化移到 server 端，握手元数据提供 `action_chunk_size` 和可用的 normalization keys。个人 adapter 不应重复反归一化。

## 6. 数据与归一化

- LIBERO 数据配置位于 `examples/LIBERO/train_files/data_registry/data_config.py`。
- 动作为 7 维：`x/y/z/roll/pitch/yaw/gripper`。
- `action_indices = list(range(8))` 表示取当前步开始的 8 步动作窗口，不是 8 维动作。
- 连续 6 个位姿增量维度使用 `min_max` 变换；夹爪维度未在相同映射中声明，需继续依据 transform 与统计文件确认其编码。
- `QwenOFT.predict_action()` 返回归一化动作；部署 wrapper 负责依据 checkpoint 旁的 `dataset_statistics.json` 反归一化。

## 7. 训练与评测入口

训练入口（实际）：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes <GPU_COUNT> \
  starVLA/training/train_starvla.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
  --framework.name QwenOFT
```

配置的真实顶层字段为 `run_id`、`run_root_dir`、`seed`、`framework`、`datasets`、`trainer`。冻结字段为 `trainer.freeze_modules`，分组学习率在 `trainer.learning_rate`。

评测采用两个环境：

```bash
# starVLA 环境
bash examples/LIBERO/eval_files/run_policy_server.sh

# LIBERO 环境
LIBERO_HOME=/path/to/LIBERO bash examples/LIBERO/eval_files/eval_libero.sh
```

不得直接复制上游 `run_libero_train.sh` 运行：固定提交中的示例变量目前设置为 `QwenPI` 和 Qwen3.5-0.8B，且包含集群特定 NCCL 网卡配置；项目主线必须创建独立、经审计的 QwenOFT 启动脚本。

## 8. Checkpoint 格式

- 示例 checkpoint：`<run_root>/<run_id>/checkpoints/steps_<N>_pytorch_model.pt`。
- run 根目录同时需要 `dataset_statistics.json`，供部署端反归一化。
- 官方 Qwen3-VL-OFT LIBERO checkpoint 可从 Hugging Face `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` 获取；当前工作站未下载。

## 9. 风险与待验证项

1. Windows、Python 3.13、磁盘空间和单卡 16GB 均不满足正式实验条件。
2. 项目说明书与稳定分支已有字段/文档差异，后续每阶段需重新核对固定提交。
3. 上游示例训练脚本与 YAML 默认 framework 不一致，必须显式覆盖 `QwenOFT`。
4. 当前尚未创建 Python 3.10/CUDA 环境；实测为 CPU-only PyTorch，BF16 CUDA 不可用，FlashAttention 与 DeepSpeed 未安装。
5. 尚未下载 checkpoint、LIBERO 和数据，未产生任何成功率或视频。
6. 上游稳定分支仍会变化；本项目以记录的 commit 为复现基线，不随意拉取最新提交。

## 10. Phase 0 验收状态

- 固定并审计上游 commit：通过。
- 确认真实接口与调用链：通过（静态源码审计）。
- 环境满足 8×40GB/BF16/FlashAttention：未通过，需在正式 Linux 训练机验证。
- 昂贵训练或评测：未执行。

# 学习日志

## Phase 0：先理解系统边界

### 你现在应掌握

1. **VLA 是什么**：把图像、语言和可选机器人状态映射为连续动作，而不是普通视觉问答。
2. **Action chunk 是什么**：一次预测未来 H 步动作，减少模型调用，但会增加开环误差累积风险。
3. **训练与执行为何分层**：模型输出归一化动作；部署层反归一化；环境 adapter 负责机器人/仿真器特定格式。
4. **为什么要固定 commit**：研究结论必须对应确定的源码、字段和 checkpoint 语义。
5. **为什么先 smoke test**：先证明 shape、loss、冻结逻辑和保存加载正确，再消耗多卡预算。

### 自测题

1. `action_indices = range(8)` 为什么不是动作维度为 8？
2. 为什么当前 LIBERO client 不应该再次反归一化？
3. QwenOFT 的 8 个动作 token 与 8 步 action chunk 是什么关系？
4. 固定执行 8 步相比每步重规划有什么收益和风险？
5. 为什么当前 Windows 机器上的单元测试通过，仍不能证明 LIBERO 训练可运行？

### 下一阶段前的实践

- 在 Linux 训练机运行 `scripts/preflight.sh` 严格模式。
- 手工画出一次 episode 从 observation 到 `env.step` 的调用链。
- 阅读 `QwenOFT.forward` 与 `predict_action`，比较训练时有标签和推理时无标签的区别。


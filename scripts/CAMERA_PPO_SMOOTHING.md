# 纯视觉 PPO 动作平滑

原模型不变：`2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt`。没有使用蒸馏。

## 直接查看

在 Pick_Anything 根目录：

```bash
bash scripts/play_camera_smooth.sh
```

该脚本使用现有 isaaclab_61 Python、GPU0、1 个环境、Kit GUI 和实时播放。
机械臂动作使用一次控制周期一次的 EMA：`filtered = 0.7 * command + 0.3 * previous_filtered`。
夹爪保留原动作；每个环境 reset 时清空各自滤波状态。相机分辨率、控制频率、物理、动作尺度和观测维数都与原模型一致。
脚本禁止自动覆盖原有导出模型。

可覆盖设备、环境数或可视化方式，例如：

```bash
bash scripts/play_camera_smooth.sh --device cuda:1
```

平滑不是写进旧模型权重的；使用原来的 Camera-v8 任务或单独调用旧 JIT 时，仍然会得到原始动作。部署到其他执行器时需要复用同样的机械臂滤波、reset 和原相对目标语义。

2026-10-09 同口径 128 局验收：原模型成功 111/128，机械臂平滑后 114/128；应用动作变化平方和下降 61.6%，实际臂速度反向比例下降 54.1%。这次无需改权重即可减轻抖动。成功率差异不能视为已证明统计显著提升。完整结果见 `photos/camera_ppo_diagnosis/smoothing_results.md`；GUI 显示性能尚未实测。

## 保留的任务

- `Pick-Apple-SO101-Staged-Camera-v8`：已验证模型的原始动作执行方式。
- `Pick-Apple-SO101-Staged-Camera-Filtered-v0`：已验证的机械臂 EMA 平滑回放。

已移除仅用于奖励消融的 `Smooth` 入口以及其他历史摄像头 PPO 实验入口。
Filtered 保留验收时的配置（包括 action_rate 权重 -0.1）；权重不影响
已有模型的确定性推理。滤波部署不需要改模型权重。
其独立 runner 的学习率为 3e-5，默认加载 actor/critic 和迭代号并重建优化器。
这只是续训配置，尚未验收通过该奖励重新训练的新模型。

## 评估与回归检查

```bash
python scripts/eval_camera_smooth.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Filtered-v0 \
  --num_envs 128 --device cuda:0 --seed 42 --viz none \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt \
  --output photos/camera_ppo_diagnosis/filtered128_alpha07.json
python scripts/test_camera_smoothing.py
```

评估只计每个环境的第一个 HOME 完整任务，结束后退出；同时统计策略原始指令、滤波后实际应用指令和机械臂实际速度。Camera-v8、Filtered 和蒸馏 v10 的环境及 runner 参数与清理前保持一致。

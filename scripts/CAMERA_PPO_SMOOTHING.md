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

## 是否需要重训

直接加滤波不需要改模型权重，可加载原检查点运行，但必须验证任务成功率；评估数据放在 `photos/camera_ppo_diagnosis/`。
如果要让策略自身学会更平滑的输出，应从该检查点继续纯视觉 PPO 微调，不必从零训练。

新增两个隔离任务：

- `Pick-Apple-SO101-Staged-Camera-Smooth-v0`：与 Camera-v8 的动作执行相同，仅把最终 action_rate 权重提高到 -0.1，用于单独学习平滑输出。
- `Pick-Apple-SO101-Staged-Camera-Filtered-v0`：同样的平滑奖励，加机械臂 EMA；训练、推理共同使用滤波，夹爪保持原执行方式。

两者 actor/critic 结构和观测都兼容 model_13000；学习率 3e-5，固定课程，继承纯 PPO。默认加载模型和原迭代号，重建优化器，避免旧优化器覆盖新学习率。显式 load_cfg 可以用于恢复优化器；此配置默认设计为从原模型开启新的微调实验。

可选续训命令（以下是训练方案，尚未代表已训练并验收的新权重）：

```bash
conda activate isaaclab_61
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Filtered-v0 \
  --num_envs 256 --device cuda:0 --seed 42 --max_iterations 1000 \
  --run_name cam_v12_arm_smooth \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt \
  --viz none
```

输出在独立的 `logs/rsl_rl/pick_apple_staged_so101_camera_smooth/`。每 100 轮存档；默认沿用迭代号，所以续训 1000 轮的文件编号约为 13000–14000。
实际选择应同时看完整任务成功率、应用动作变化平方和、实际关节速度反向率。-0.1 是初始实验值，不是已经验证的最优奖励权重。

## 评估与回归检查

```bash
python scripts/eval_camera_smooth.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Filtered-v0 \
  --num_envs 128 --device cuda:0 --seed 42 --viz none \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt \
  --output photos/camera_ppo_diagnosis/filtered128_alpha07.json
python scripts/test_camera_smoothing.py
```

评估只计每个环境的第一个 HOME 完整任务，结束后退出；同时统计策略原始指令、滤波后实际应用指令和机械臂实际速度。原有 Camera-v8 和其他训练任务没有改动。

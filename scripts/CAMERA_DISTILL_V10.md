# 摄像头蒸馏 v10

唯一任务入口：`Pick-Apple-SO101-Staged-Camera-Distill-v10`。
独立配置位于 `pick_anything/tasks/pick_apple/camera_distill_v10_cfg.py`，不依赖历史蒸馏任务。

学生使用 256×192 腕部 RGB 和 24 维本体感受；教师使用 111 维状态观测。
动作尺度为机械臂 0.033、夹爪 0.02，90 秒 episode，八阶段均匀出生。
标准 RSL-RL `Distillation` 保留教师原始动作标签，学生执行也不裁剪至 ±1，
以匹配无动作裁剪的平滑教师。学生从零初始化，rollout std 为 0.05。

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Distill-v10 \
  --num_envs 32 --device cuda:0 --seed 42 --max_iterations 2 --viz none \
  --run_name distill_v10_unclipped_smoke \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt
python scripts/test_distill_v10.py
```

通过短程仿真后，可增加环境数及迭代数；默认配置为 4000 轮，每 250 轮保存。
checkpoint 和训练输出仍保存在被 Git 忽略的 `logs/` 中，需要本地已有教师权重。

完整任务验收使用学生 checkpoint：

```bash
python scripts/eval_pick_home.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Distill-v10 \
  --num_envs 256 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera_distill/<run>/model_XXXX.pt \
  --external_callback scripts.eval_pick_home.install_episode_counter
```

目前仅通过 CPU 动作契约和真实教师权重加载检查；尚未完成该版本的训练和
HOME 完整任务成功率验收。教师成功率及混合出生的训练 success 不能代替学生验收结果。

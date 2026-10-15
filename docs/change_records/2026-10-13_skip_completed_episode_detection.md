# 首回合评估跳过已计分环境的重复 DINO 检测

## 问题与证据

语言抓取对照采用每环境首个 HOME 回合计分。原环境自动 reset 后继续运行，
`GroundedRGBMask.__call__` 对每个新 render frame 都调用检测器，没有排除已经计分
的环境。单橘子 seed43 的 GT 评估出现 3 次成功、1 次 90 秒超时；对应 DINO 评估
即使其余环境结束，仍会为它们重置后的回合检测。该开销不产生新的有效评估样本。

## 实际修改与范围

- `pick_anything/grounded_pick/observations.py`：只在 DINO 首回合评估时，根据
  Grounded 的 `_evaluation_seen` 或 OpenVocab 的 `_seen` 排除已完成的行。
  尚未完成的环境仍逐帧检测；GT、普通回放和训练分支保持原计算。
- `scripts/test_open_vocab_pick.py`：检测调用次数验证，覆盖已完成跳过、未完成继续、
  非首回合评估关闭优化。
- 原已完成结果保留；未完成的 single seed43 DINO 和 duck DINO 已由 SIGINT 终止，
  不记为完成或成功。只终止本轮评估父子进程，未停止用户训练。

## checkpoint、命令与进程

仍使用 `models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt`，
未修改或训练权重。需要重新启动评估进程；正在运行的训练无须重启。
原源码快照保留于 `outputs/language_pick/20261013_*_comparison/source/`，
新评估使用新的输出目录及源码快照，避免把不同实现的来源混写。

## 验证

CPU：OpenVocab 7 项、Grounded 10 项、三帧迁移 7 项全部通过；其中混合两环境
检查确认只有未计分行的 RGB 进入检测器，并且掩码写回正确行。普通回放仍检测。
日志位于 `outputs/grounded_pick/language_*tests_skip_completed_20261013.log`。
新 GPU 结果待追加。该修改只节省已计分行的检测，不能修复多物体策略泛化或目标
误检，不改变成功条件。

实际续跑（Conda isaaclab_61，GPU 2）：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --scene single --num-envs 4 --seeds 43 --backends dino \
  --output-dir outputs/language_pick/20261013_single_remaining

CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --scene multi --targets yellow_duck blue_cube red_cylinder --num-envs 1 --seeds 42 \
  --output-dir outputs/language_pick/20261013_multi_remaining
```

剩余三种物体先各做 1 回合功能对照，明确不作为成功率估计；原来已完成的橘子
2 回合对照保留，旧 duck GT 2 回合单独列出，不与新 1 回合混算。

## GPU 续跑完成结果

- 新 single seed43 DINO 已完成4回合：2成功、1超时、1掉落；正常退出。
  `outputs/language_pick/20261013_single_remaining/results.json` 与逐回合 `summary.json` 保存完整结果。
- 新 multi 剩余3种物体各 GT／DINO 1回合，6个条件全部完成，均超时，无实际抬升；正常退出。
  结果位于 `outputs/language_pick/20261013_multi_remaining/results.json`。
- 未完成行继续逐帧检测的CPU检查及实际GPU首回合流程均通过；性能改动没有改变成功
  条件，也没有使多物体迁移通过。没有做独立性能计时对照，不声称确定的倍数加速。
- 两组续跑均已退出，没有留下本轮评估进程，原训练进程未停止，checkpoint hash不变。

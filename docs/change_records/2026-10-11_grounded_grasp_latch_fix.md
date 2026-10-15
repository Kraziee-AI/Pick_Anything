# Grounded：修复 PREGRASP 提前夹持锁存导致阶段无法推进

## 遇到的问题与证据

`model_3400.pt` 在 GT 掩码、HOME 出生的评估中无法完成有效抬起或进盒。
seed 42/43 各 8 回合，16/16 最远停在 PREGRASP 并超时；15/16 在 PREGRASP
就提前锁存 `grasped=True`。具体轨迹和图像见
[详细诊断报告](../../outputs/grounded_pick/diagnostics/REPORT.md)。

共享状态机在任意阶段只要瞬时 GRASP 条件成立，就锁存 `grasped`；
共享 PREGRASP 完成条件却要求 `~progress.grasped`，而接近奖励也在锁存后关闭。
因此提前接触可能同时阻断阶段推进与接近引导。连续 3 步满足原始夹持条件也
不能解决已经卡在 PREGRASP 的情况。CPU 反例和实际回合均复现了这一点。

## 为什么需要修改

继续沿用旧逻辑训练无法解除提前锁存后的阶段阻塞。需要让 Grounded 的夹持
里程碑与 GRASP 阶段的连续确认一致，保留 PREGRASP 接近／就位的可学习路径。
该修复只解除已确认的逻辑障碍，不保证旧权重立即学会正确开爪、抬起或进盒。

## 修改范围与方案

- 新增 `pick_anything/grounded_pick/progress.py`：Grounded 专用状态机，只在
  GRASP 阶段经连续确认并完成阶段时，才新增夹持锁存；PREGRASP 接触不锁存。
- `pick_anything/grounded_pick/env.py`：在加载观测／奖励管理器前，给本环境
  的共享状态对象安装专用状态机。共享函数仍读这个环境自己的状态对象。
- 保留 reset bank 的阶段种子、奖励支付和 partial reset 语义。
- 更新 Grounded 诊断中的里程碑语义标注，避免将新旧 `grasped` 指标混为一谈。
  结果新增 `grasp_semantics`，`diagnostics.py` 与 `GROUNDED_PICK.md` 同步说明。
- 增加回归检查：提前接触后仍能就位、夹持必须连续确认、第一步不付奖、
  中段种子与局部重置、环境级安装及旧任务隔离。
- 更新 `AGENTS.md`：后续每项修改都新增 Markdown 记录，记录问题、原因、
  文件范围、验证结果与训练／评估命令。

不修改旧三种训练方式的源代码，不修改 PPO 网络维度、掩码生成、动作尺度、
物理阈值、奖励权重或课程比例；可以加载原 Grounded checkpoint 续训。

## 验证

### 已完成

- Conda `isaaclab_61` 下运行 `python scripts/test_grounded_pick.py`，10 项检查通过。
  原提前接触反例在 Grounded 中不再锁存；随后真实满足就位条件可以进入 GRASP。
  连续夹持中断会清空确认计数，第三个连续有效步才锁存并进入 LIFT，奖金不重复。
  第一控制步、reset bank 种子、partial reset 和旧状态机隔离均通过检查。
- GPU 1，`model_3400.pt`、seed 42、8 个 HOME 首回合、GT 掩码、90 秒：
  提前锁存 0/8；新 `grasped`、有效抬起与成功均为 0。旧策略仍停在 PREGRASP，
  尚未学会满足准确开爪就位条件，需要续训。原始夹持条件仍有触发，但不再
  冒充确认夹持或关闭接近奖励。
- GPU 0，4 个环境，从 `model_3500.pt` 完成 2 轮 PPO 续训（3500、3501）：
  模型／优化器可加载，value loss 0.0149→0.0007，surrogate loss
  -0.0645→-0.0422，没有 NaN 或维度错误。此为接口和更新检查，不是成功率验收。

验证产物：

- [CPU 日志](../../outputs/grounded_pick/grasp_latch_fix/cpu_tests.log)
- [修复后 HOME 统计](../../outputs/grounded_pick/grasp_latch_fix/model_3400_seed42/summary.json)
- [PPO 续训日志](../../outputs/grounded_pick/grasp_latch_fix/train_smoke.log)
- [冒烟训练 run](../../logs/rsl_rl/pick_orange_so101_grounded/2026-10-11_21-44-34_grasp_latch_fix_smoke/)

本次 HOME 评估的物理 GPU 与旧 seed 42 测试不同，Newton 未启用运行间确定性
保证，不用于逐帧一致性比较。CPU 反例验证规则变化，GPU 检查确认规则在真实
环境中应用、旧 checkpoint 能使用。尚未执行修复后的长训练及 DINO 成功率验收。

## 训练与重启

已启动的训练进程持有旧代码，保存权重不会让该进程自动应用修复。
正式续训前需要在旧 Grounded 训练终端按 Ctrl+C，然后用修复后的入口启动新 run。
已确认 checkpoint 路径存在，并以它完成兼容性检查。正式续训使用独立 run，
先增加 1000 轮，之后对新的 checkpoint 做同口径 HOME 评估。
`--max_iterations 1000` 在续训时表示新增 1000 轮，约从 3500 到 4500。

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything
export ISAACSIM_ASSET_ROOT="$PWD/assets/isaac_cache"

CUDA_VISIBLE_DEVICES=2 python scripts/train_grounded_pick.py \
  --task Pick-Orange-SO101-Grounded-v0 \
  --mask-backend gt \
  --num_envs 512 --device cuda:0 --seed 42 \
  --max_iterations 1000 --viz none \
  --run_name orange_gt_grasp_fix_v1 \
  --checkpoint "$PWD/logs/rsl_rl/pick_orange_so101_grounded/2026-10-11_12-42-26_orange_gt_v1_resume/model_3500.pt"
```

启动日志应出现：

```text
[GROUNDED] grasp milestone: confirmed_GRASP_completion_or_reset_bank_seed
```

评估示例（将 `<new_run>/model_XXXX.pt` 替换为新 run 的权重）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_grounded_pick.py \
  --mask-backend gt --num_envs 8 --device cuda:0 --seed 42 --viz none \
  --checkpoint "$PWD/logs/rsl_rl/pick_orange_so101_grounded/<new_run>/model_XXXX.pt" \
  --diagnostics-dir outputs/grounded_pick/grasp_latch_fix/after_finetune_seed42
```

不能使用 `--train_env_cfg` 代替 HOME 完整任务评估，也不能将更新后的 `grasped`
比例直接与旧瞬时锁存比例比较。本次未终止原训练进程，也未启动正式长训练。

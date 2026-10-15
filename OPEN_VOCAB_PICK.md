# 开放词汇目标定位与多物体 PPO 抓取

## 2026-10-15 归档说明

最新随仓库保存的三帧权重为
`models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`。
2026-10-14 使用真实 DINO、orange 目标、seed42/43/44 各 4/4/2 回合：
单橘子完整任务 8/10，多物体抬升任务 0/10；尚未验证其他语言目标的成功率。
详见[归档逐回合报告](docs/evaluations/2026-10-14_model2599/REPORT.md)和
[汇总](docs/evaluations/2026-10-14_model2599/SUMMARY.json)。
以下 2026-10-13 数字和最初实现限制为历史记录，均保留原适用范围。

复评这两种场景时可用以下命令，每个 scene 分别执行，输出目录须使用未存在的新路径：

```bash
conda activate isaaclab_61
# 将 single 换为 multi 可评估多物体；seed42/43 各4回合。
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt \
  --scene single --targets orange --backends dino --seeds 42 43 --num-envs 4 \
  --output-dir outputs/language_pick/model2599_single_seed42_43_rerun
# 再用 seed44 的2回合，合计10个首回合。
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt \
  --scene single --targets orange --backends dino --seeds 44 --num-envs 2 \
  --output-dir outputs/language_pick/model2599_single_seed44_rerun
```

这些是归档权重的复评命令，本次提交整理没有重新执行 GPU 仿真。
检测器权重仍需按 [GROUNDED_PICK.md](GROUNDED_PICK.md) 准备本地缓存。

## 2026-10-13：直接评估现有三帧抓取 actor

新入口 `scripts/evaluate_language_pick.py` 对同一 checkpoint、目标和种子依次执行
GT／GroundingDINO 首回合评估，保存命令、权重 hash、任务源码快照、RGB／mask、
日志与 JSON 结果。每次使用新的输出目录，避免覆盖证据。`single` 采用原橘子
放盒撤爪标准；`multi` 使用已有四物体场景的稳定抬升标准，两者不能混算成功率。

多物体 play/eval 增加 `--actor-only`：严格加载 actor 和本体归一化参数，自动识别
单帧／三帧，不加载橘子任务的 49 维 critic、优化器或训练迭代。
该选项只用于评估，禁止用于训练恢复。完整新任务 checkpoint 仍可正常恢复。

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --scene multi --targets orange yellow_duck blue_cube red_cylinder \
  --num-envs 2 --seeds 42 --output-dir outputs/language_pick/my_multi_comparison
```

单物体对照将 `--scene multi --targets ...` 换成 `--scene single --targets orange`。
单次中文指令评估：

```bash
export ISAACSIM_ASSET_ROOT="$PWD/assets/isaac_cache"
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 python scripts/eval_open_vocab_pick.py \
  --instruction "抓小黄鸭" --mask-backend dino --actor-only \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --num_envs 2 --seed 42 --device cuda:0 --viz none
```

多物体结果额外记录每回合最大抬升高度和曾经抬起其他物体的比例（阈值 4 cm）。
这些是迁移测试，不代表现有橘子权重已经学会抓取每个新物体。
实际验证结果见 [本次记录](docs/change_records/2026-10-13_language_pick_evaluation.md)。

本次12个对照条件均已完成：单橘子 GT 7/8、DINO 6/8（seed42／43各4回合）；
四物体场景中橘子各0/2，其余三种物体各0/1，GT与DINO均没有实际抬升。
这确认语言掩码闭环已接通，也说明现有单橘子权重尚未通过多目标迁移。
这些小样本结果不是稳定成功率估计。完整命令、图像和失败记录见
`outputs/language_pick/REPORT_20261013.md` 与 `SUMMARY_20261013.json`。

以下保留最初单帧任务的说明；旧文中“不允许直接加载完整 runner”仍成立，
现可通过上述 `--actor-only` 评估当前三帧 actor。

现有橘子模型已经实现 RGB＋目标掩码（4 通道）＋24 维机械臂状态 → 6 维相对关节动作。
`logs/rsl_rl/pick_orange_so101_grounded/2026-10-11_22-00-43_orange_gt_grasp_fix_v1/model_6499.pt`
已检查：actor 第一层卷积权重形状为 `(16,4,5,5)`，输出层为 `(6,128)`。
这证明接口结构；不证明该模型能抓取未训练的物体。

新增任务 `Pick-Anything-SO101-OpenVocab-v0` 使用同样的 actor 结构，
场景包含橘子、小黄鸭、蓝色方块、红色圆柱及原纸盒。成功条件是指定目标被夹持并
稳定抬升 8 cm、连续确认 5 个控制步；纸盒是场景物体，本任务不要求入盒。
它使用独立奖励、HOME 重置和模型目录，不使用橘子 reset bank。

```text
GT 训练：每回合随机目标 → 几何投影矩形掩码 ─┐
语言回放：文本＋腕摄 RGB → GroundingDINO → 矩形掩码 ─┤
                                                    ↓
                         RGB＋mask (4×96×128) ＋ proprioception (24)
                                                    ↓
                                   CNN PPO → 6 维关节动作
```

掩码与完整 RGB 沿通道维拼接，不将 RGB 乘以掩码。掩码是 0/1 矩形区域，
与本地 GCRL_UR10e 的实际 `create_mask()` 一致，不是 SAM 像素轮廓。
非球形物体用有方向的几何包围盒投影，橘子保留球体透视投影。
检测失败使用全零 mask，不读取仿真目标位置来替代检测。

检测器支持任意英文目标描述，保留属性和位置词，如 `striped ceramic mug`、
`blue cube on the left`；常见中文目标有显式映射，复杂中文用 `--caption`
提供英文检测词。模拟训练与评估仍需要场景里的真实资产。
开放词汇定位不等于 PPO 已获得任意物体抓取能力。

## CPU 检查

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything
python scripts/test_open_vocab_pick.py
python scripts/test_grounded_pick.py
```

## 多物体训练入口

下面是短训练命令；本次 GPU 尝试在沙箱中未能访问 CUDA，尚未完成仿真验证。
新任务尚无训练完成的 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_open_vocab_pick.py \
  --num_envs 4 --device cuda:0 --seed 42 --max_iterations 2 --viz none \
  --run_name open_vocab_smoke \
  --dump-dir outputs/open_vocab_pick/smoke_frames --dump-every 1 --dump-max 8
```

GT 训练默认每回合从四种物体随机选目标；指定 `--target-key blue_cube`
可只训练一个目标。模型目录为 `logs/rsl_rl/pick_anything_so101_open_vocab/`。
正式训练须先通过短仿真检查，再将环境数和迭代数调整到实际需要的规模。

## 定位与回放入口

单张图支持开放目标，不启动 Isaac；默认从本仓库已有缓存离线加载：

```bash
python scripts/ground_orange_image.py --device cpu \
  --image /path/to/rgb.png --instruction "抓左边的杯子" \
  --caption "white mug on the left" \
  --output-dir outputs/open_vocab_pick/image_check
```

以下回放和评估需要**新任务**的 checkpoint，不能直接使用旧橘子任务 checkpoint
加载完整 runner，因为 critic 由旧任务的 49 维改为 54 维，奖励目标也不同。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/play_open_vocab_pick.py \
  --instruction "抓小黄鸭" --num_envs 1 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_anything_so101_open_vocab/<run>/model_XXXX.pt \
  --dump-dir outputs/open_vocab_pick/duck_play

CUDA_VISIBLE_DEVICES=2 python scripts/eval_open_vocab_pick.py \
  --instruction "抓蓝色方块" --mask-backend dino \
  --num_envs 1 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_anything_so101_open_vocab/<run>/model_XXXX.pt
```

同一 checkpoint 使用 `--mask-backend gt` 做对照。评估仅统计每个环境首个 HOME
回合，在自动 reset 前读取终止原因，输出 `OPEN_VOCAB_PICK_EVAL_RESULT=`。
`--target-key` 只选择训练／评估真值，DINO 的输入仍仅为 RGB 与文本。
DINO 模式不允许随机真值目标，避免奖励与固定语言指令不一致。

当前验证限于 CPU 契约与真实 PPO 数值更新。未完成多物体 GPU 物理、渲染、
实际 DINO 检测或完整抓取成功率评估；没有启动正式长训练。

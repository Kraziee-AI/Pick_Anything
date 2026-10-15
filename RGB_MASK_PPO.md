# RGB＋mask PPO：三帧历史与 RGB 策略初始化

当前目标是单橘子的 HOME → 抓起 → 放盒 → 撤爪完整任务。
任务 ID：`Pick-Orange-SO101-Grounded-v0`；先用 GT 矩形掩码，之后接语言检测。
使用 Conda `isaaclab_61`，所有输出与缓存均在 Pick_Anything。
独立多物体语言对照与复评命令见 [OPEN_VOCAB_PICK.md](OPEN_VOCAB_PICK.md)。

## 2026-10-14 最新保存权重核对

当前最新保存的三帧掩码checkpoint是：
`logs/rsl_rl/pick_orange_so101_grounded/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`。
已读取内部iter=2599、CNN输入12通道、本体24维、动作6维；actor张量均有限。
本页下方model_148是早期已验证候选，旧成功率不代表model_2599。
后续已在物理GPU 0实测该文件，真实GroundingDINO、语言目标orange，从HOME开始
分别10回合：单橘子完整任务8/10=80%，多物体抓橘子0/10=0%。
单橘子10/10确认抓取和抬升，失败为1次掉落、1次盒外放置后超时；多物体10次
全部超时、0/10确认夹住橘子，1次出现非目标物体抬升。
两套成功条件不同：单橘子要求放盒与撤爪完成，多物体要求正确目标保持夹持并
抬高8cm持续5步。GT独立评估和其他语言目标未在本轮测试。
逐回合结果见[归档报告](docs/evaluations/2026-10-14_model2599/REPORT.md)和
[汇总 JSON](docs/evaluations/2026-10-14_model2599/SUMMARY.json)。原始日志和图像保留在本地 outputs。

GitHub 随代码保存同一权重及训练配置：
`models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`。
SHA256：`f4414a3811906a33b3eba0436140173d076dce08d601eeb87bebdb2bdc8a4386`。
本次归档没有重新训练或改变原文件；下列回放命令使用仓库内的归档副本。

在已配置好的Conda和Kit终端里回放最新权重：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play_grounded_pick.py \
  --instruction "orange" --mask-backend dino --image-history 3 \
  --num_envs 1 --device cuda:0 --seed 42 --viz kit --real-time \
  --checkpoint "$PWD/models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt"
```

核对与限制见 [记录](docs/change_records/2026-10-14_latest_mask_checkpoint_audit.md)。

## 2026-10-13 的训练方案

```text
腕摄 RGB (256×192) → RGB (3×96×128) ─┐
GT 目标矩形框 → binary mask (1×96×128) ─┤
                                       ↓ 每帧按通道拼接
                              [RGB＋mask] × 最近三帧
                              12×96×128 → CNN
关节位置／速度／上一动作／PD目标 (24维) ─→ MLP → PPO actor → 6维动作
```

每帧是 4 通道；CNN 的默认新训练输入为三帧共 12 通道，最旧在前、最新在后。
每环境独立复位历史，首帧重复填充，重复读取不推进历史；指令变更清空旧目标历史。
RGB 双线性缩放、mask 最近邻缩放。Actor 不输入目标世界坐标、阶段号或目标 ID；
critic 仍使用 49 维特权观察。

参考本地 GCRL_UR10e：`mask_7d_v1.py` 将 RGB 与投影矩形 mask 拼接，
`training/Train_reach.py` 使用 CNN＋状态网络、PPO 和三帧堆叠。
其 `done = contact == 2` 是接触目标；本任务继续按物理放盒、撤爪、静置验收。
本项目使用 RSL-RL，不能照搬其 SB3 网络尺寸、2048 步 rollout 和成功率。

旧 mask 策略 `model_13600.pt` 实测 0/32 完整成功、0/32 抬升，31/32 曾确认夹持。
轨迹证明空爪闭到约 -0.175 rad 时蹭桌接触可误判，随后目标留桌而爪离开，
接近奖励因夹持锁存关闭。本次针对这些证据修改：

- Grounded 新夹持确认要求爪角 >0 rad、目标进入3.5cm捕获口袋；共享 RGB 任务不改。
- 接近势能差延续到物体真正抬起，并增加抬起前空爪全闭惩罚。
- 默认不从失败策略延长训练；将已验证的 RGB `model_17596.pt` 初始化到三帧 mask actor。
  最新 RGB 的卷积权重原样复制，历史与 mask 通道初始化为零；其后所有通道均参与 PPO 更新。
- 复制兼容 actor、critic、归一化统计，清空 Adam 动量及训练迭代；源文件保留。
- 新微调配方 lr=2e-5、clip=0.1、entropy=1e-4；加载 checkpoint 后显式恢复该学习率，
  防止 RSL-RL 从旧优化器中带回旧 LR。其余保留 gamma=0.995、lambda=0.95、64 步、5 epochs、8 mini-batches。

初始化时策略仍依赖最新 RGB，尚未学会目标选择。只有 PPO 更新后才能逐步利用新增
mask／历史通道；单橘子成功率也不能证明开放词汇或多目标选择能力。

## 训练命令

当前旧训练进程必须重启才能应用修改。下列默认入口创建新运行，自动激活 Conda，
使用 GPU 0、GT 掩码、无 UI，并从 RGB 成功策略生成三帧初始化：

```bash
cd ~/IsaacLab_6.1/Pick_Anything
PICK_MASK_GPU=0 PICK_MASK_ENVS=128 PICK_MASK_ITERATIONS=1500 \
  bash scripts/train_rgb_mask_ppo.sh
```

`PICK_MASK_ITERATIONS` 是本次追加迭代数，1500 是实验预算，不是收敛保证。
输出在 `logs/rsl_rl/pick_orange_so101_grounded/` 的新时间戳目录。
默认 RGB 源：`models/camera_ppo/2026-10-12_15-10-13_camera_stage_recovery_v2/model_17596.pt`。
初始化产物在 `outputs/grounded_pick/initializations/`，带源权重哈希、历史帧数、学习率；不会覆盖旧文件。

后续续训新模型：

```bash
PICK_MASK_GPU=0 PICK_MASK_ENVS=128 PICK_MASK_ITERATIONS=1500 \
  PICK_MASK_CHECKPOINT="$PWD/models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt" \
  bash scripts/train_rgb_mask_ppo.sh
```

入口会根据第一层卷积自动识别 1 帧或 3 帧；显式 `--image-history` 不匹配时拒绝加载。
`--init-from-rgb` 与 `--checkpoint` 互斥；前者是新初始化，后者恢复训练状态。
`PICK_MASK_RECIPE=legacy` 可保留原超参数，便于历史对照。

从零训练（无 RGB 初始化）：

```bash
conda activate isaaclab_61
export ISAACSIM_ASSET_ROOT="$PWD/assets/isaac_cache"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/train_grounded_pick.py \
  --mask-backend gt --image-history 3 --num_envs 128 --device cuda:0 --seed 42 \
  --max_iterations 12000 --viz none --run_name rgb_mask_temporal_from_scratch
```

从零训练的成功率尚未验证。训练仍保留曝光／白平衡随机化、10% 框抖动及 2% 掩码丢失；回放关闭随机化。

## 完整任务评估与结果

```bash
conda activate isaaclab_61
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/diagnose_grounded_training.py \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/<new_run>/model_XXXX.pt \
  --label my_evaluation --num-envs 16 --seeds 42 43
```

相同种子、从 HOME 出生、首回合、策略均值动作、最长 90 秒；脚本保存权重哈希、
源码快照、轨迹、图像及汇总。label 已存在时拒绝覆盖。新夹持角度阈值写入摘要。
训练中的 `birth_*_success` 是本进程累计比率；平均近期这些日志不是近期回合成功率。
`Episode_Termination/success` 混合了所有出生阶段，也不能代替 HOME 完整任务评估。

2026-10-12 原开爪塑形及 100 轮短训均未得到完整成功，历史结论保留于
`docs/change_records/2026-10-12_grounded_pregrasp_recovery.md`。
2026-10-13 修改、实际 GPU 结果及 checkpoint 记录于
`docs/change_records/2026-10-13_grounded_temporal_transfer.md`。

已完成100＋50轮短训，GPU 0、seed42／43各16、HOME完整首回合结果：

| 条件 | 完整成功 | 有效抬升 |
|---|---:|---:|
| 修改前model_13600 | 0/32 | 0/32 |
| 新流程model_148 | 27/32（84.375%） | 28/32 |

27个成功回合全部出现有效抬升。新夹持代理门收紧，最终物理放盒撤爪标准保持；
改动是整体方案，尚未做mask／历史／初始化的独立消融。小样本结果不代表稳定成功率。
验证权重保存在`models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt`。
mask核从零变为非零；同RGB和状态下mask置零会改变网络输出，但未验证多目标选择。
完整报告见`outputs/grounded_pick/recovery_diagnosis/TEMPORAL_TRANSFER_REPORT_20261013.md`。

## CPU 检查

```bash
python scripts/test_grounded_transfer.py
python scripts/test_grounded_pick.py
python scripts/test_grounded_recovery.py
```

三帧历史、部分复位、动作保持、优化器清空、实际 PPO 更新、新通道可学习、
空爪禁止确认和物理抬升前的恢复信号均有检查。

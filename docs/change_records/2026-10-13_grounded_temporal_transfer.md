# RGB＋mask PPO：空爪误确认、三帧历史与成功策略初始化

## 问题、复现与实际证据

用户要求仔细对照 GCRL_UR10e 修改训练代码。旧 mask 续训长期 HOME 成功率为零。
已读取本地 GCRL_UR10e `training/Train_reach.py` 和 `mask_7d_v1.py`：
RGB＋投影矩形 mask、CNN＋状态网络、PPO、三帧堆叠、确定性独立评估；
其 done 是双指接触，不能用于本项目完整抬升、放盒、撤爪的成功标准。

修改前固定 checkpoint：
`logs/rsl_rl/pick_orange_so101_grounded/2026-10-12_23-32-18_rgb_mask_recovery_v2_resume/model_13600.pt`。
SHA256 `2e021833683cd24efa7f5c29a2c54430f93ad06ce4e2641f8bc846fab68b39ed`。
GPU 0、GT、seed 42/43 各 16、HOME 首回合、策略均值、最长 90 秒：

- 成功 0/32；全部超时；状态机夹持确认 31/32；有效抬升 0/32；开爪就位 0/32。
- 最大阶段 LIFT 31、PREGRASP 1；平均 mask 非空比例 23.18%。
- 采样轨迹有 13,910 条“夹持已锁存、橘子未抬起、爪口距目标 >7.5cm”的记录。
  一个例子：env10 第95步，stage=LIFT、爪距8.78cm、爪角−0.17490rad、橘子z=0.06831m。
  后续爪离开目标，目标留在桌面。低 mask 非空比例伴随目标离开视野，不能据此
  归因于掩码生成器。原接近奖励在 grasped 锁存后关闭，恢复信号消失。

已有成功 RGB 源模型：
`models/camera_ppo/2026-10-12_15-10-13_camera_stage_recovery_v2/model_17596.pt`。
历史独立 RGB 评估 474/512；该成绩不直接用于新 mask 模型。
迁移前后真实 CNN 动作数值等价已通过 CPU 检查。
迁移模型第一组轨迹中，物体高于0.105m且仍在爪口的 LIFT／STABILIZE／TRANSPORT
样本123条，夹爪最小角0.38457rad、中位数0.47791rad；支持拒绝全闭负角范围的误确认。

进一步逐回合检查首次确认：旧权重31次确认的距离中位数5.867cm，30次超过3.5cm；
仅1次角度非正。因此**仅增加角度门不足以解决主要误确认**。
新初始化模型30次首次确认的距离中位数2.665cm，仅1次超过3.5cm。
最终增加既有PREGRASP捕获口袋3.5cm的确认距离门，不用7.5cm宽松门触发新夹持。
3.5cm只用于首次确认，不要求搬运过程持续使用该窄门，后段既有阈值保持。

## 修改目标与范围

恢复能实际抓起的策略起点，引入 GCRL 的时间上下文，并对已经定位的空爪误确认
及奖励断供修复。仅修改 Pick_Anything；外部 GCRL、IsaacLab、旧模型不修改。

- `grounded_pick/checkpoints.py`、`scripts/initialize_rgb_mask_ppo.py`：RGB actor
  初始化为1或3帧 RGB/mask；最新 RGB 卷积核复制，额外通道置零且可训练。
  兼容 critic 和归一化统计复制，Adam 状态清空、iter=0、LR=2e-5。
  校验网络维度，保存源 hash／iteration／历史数，产物带版本且不覆盖已有文件。
- `grounded_pick/observations.py`：可选三帧4通道历史（CNN 共12通道），最旧在前；
  每环境 partial reset、首帧重复填充、同一 render frame 不重复入队、目标变更清历史。
  原单帧4通道保留；不加入 actor 世界坐标或阶段编号。
- `grounded_pick/progress.py`、`env.py`、`config.py`：Grounded 新夹持确认增加
  `gripper_pos > 0.0rad` 的空爪排除门。保留原力门和持续确认，不放宽最终成功。
  新确认近距从7.5cm收紧到已有3.5cm捕获口袋，防目标尚未进入爪口时蹭桌被确认。
  共享 CameraStageProgress、RGB task 和底层物理配置不改。
- `grounded_pick/rewards.py`、`config.py`：接近势能差持续至物理抬升；抬升前全闭
  空爪惩罚，统一缩放后weight=−0.1。已有开爪项保留。奖励均不直接移动机械臂。
- `grounded_pick/diagnostics.py`：raw 夹持条件及阈值报告与新角度门保持一致。
- `scripts/grounded_pick_cli.py`：`--init-from-rgb`、`--image-history`、
  `--training-recipe`；文件 checkpoint 自动识别1／3帧，维度冲突拒绝加载。
  初始化与续训互斥；transfer配方 LR=2e-5、clip=0.1、entropy=1e-4。
  仅在本训练进程内加载后显式设置 optimizer LR，防旧 checkpoint 覆盖新配方。
- `scripts/train_rgb_mask_ppo.sh`：默认GPU 0、GT、无UI，从上述RGB源初始化；
  显式 PICK_MASK_CHECKPOINT 则续训，默认采用 transfer 配方。
- `scripts/diagnose_grounded_training.py`：记录历史帧数、新相关源码快照。
- `scripts/test_grounded_transfer.py`：7项时间历史、真实actor输出保持、真实PPO更新、
  mask核可学习、optimizer重置、路径与输入检查、空爪门及奖励恢复检查。
- `scripts/test_grounded_pick.py`：manager安装 fixture 增加实际配置和爪角字段。
- `RGB_MASK_PPO.md`：改为新训练方案，保留旧失败方案的历史记录链接。

## 已完成验证及限制

- 新7项＋Grounded10项＋recovery7项＋Camera10项＋open-vocabulary6项，共40项CPU检查通过。
  包含实际 PPO 更新，三帧 mask 权重从零变为非零；shell语法通过。
- 修改前旧 mask 权重 GPU32回合诊断已完成（0/32）。
- 迁移初始化、仍使用原夹持角度门：seed42 16/16、seed43 8/16，共24/32=75%。
  抬升26/32；7超时、1掉落。产物见 transfer_init_20261013。
  初始化时新增mask／历史核仍为零，此结果只证明继承了RGB能力。
- 仅角度收紧（确认近距仍7.5cm）的初始化独立评估：seed42 14/16、seed43 11/16，
  共25/32=78.125%；有效抬升30/32；6超时、1掉落。目录transfer_init_strict_20261013。
- 首次GPU微调100轮已完成：64环境、409,600环境步、784.92秒，正常退出。
  该训练在3.5cm收紧前执行；当时源码见上述初始化评估source快照。
  训练HOME累计比率末值76.01%、混合出生末值83.08%，这些不是独立评估结果。
  mask卷积核L2=0.0302143、历史RGB核L2=0.0342759（两者初始化均零），
  实际optimizer LR=2e-5，actor有限。数据见temporal_transfer_pilot_training_summary.json。
- 最终3.5cm规则下续训50轮已完成：64环境、204,800环境步、398.92秒，正常退出。
  HOME累计末值63.94%、混合出生末值81.69%，仅为本进程训练指标，不与上一进程累计比率混算。
  mask核L2=0.0372153、历史RGB核L2=0.0414172；optimizer LR=2e-5，actor／critic均有限。
  源码保存于temporal_transfer_distance_pilot/source_snapshot。
- 最终权重model_148.pt、最终规则的同条件独立评估：seed42 14/16、seed43 13/16，
  共27/32=84.375%。确认夹持28/32、有效抬升28/32；27个成功回合全部出现有效抬升。
  5次超时，无掉落／翻盒／机器人异常终止。目录transfer_final_148_20261013。
  这是小样本GT单橘子仿真结果；不能认定三帧或mask独立贡献，也不承诺稳定84.4%。
- 使用10张实际dump的RGB/mask、重复历史帧和中性归一化状态，只将mask通道置零：
  初始化模型均值动作差0，训练后平均绝对差0.038173、最大差0.197027（未裁剪网络均值）。
  说明mask会影响模型输出，不是多目标选择能力的验收。见temporal_mask_sensitivity.json。
- 首轮100轮＋最终规则50轮，共150轮、614,400环境步。新正式长训练未启动。
- 单橘子本任务不能验证目标选择／开放词汇；三帧历史和mask的独立贡献尚未消融。
  本次没有使用 GroundingDINO、实机、像素轮廓分割或训练多物体。
- `birth_*` 为进程累计成功比率，近期日志均值不能当作近期回合比率；独立HOME评估才是验收依据。

## checkpoint、命令与重启

三帧初始化产物：
`outputs/grounded_pick/initializations/rgb_to_rgb_mask_v1_h3_lr2e-05_a984aeb963b61d5f.pt`。
SHA256 `3ec0427712c441594bd69c00d901b7ee84554ac8683782dc1db3e2f77cf83af4`。

以下全部使用 Conda isaaclab_61、GPU 0 和仓库内缓存。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/diagnose_grounded_training.py \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/2026-10-12_23-32-18_rgb_mask_recovery_v2_resume/model_13600.pt \
  --label baseline_13600_20261013 --num-envs 16 --seeds 42 43

python scripts/initialize_rgb_mask_ppo.py \
  --checkpoint models/camera_ppo/2026-10-12_15-10-13_camera_stage_recovery_v2/model_17596.pt

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/diagnose_grounded_training.py \
  --checkpoint outputs/grounded_pick/initializations/rgb_to_rgb_mask_v1_h3_lr2e-05_a984aeb963b61d5f.pt \
  --label transfer_init_strict_20261013 --num-envs 16 --seeds 42 43
```

原规则初始化评估的label为transfer_init_20261013。
修改前源码保留在baseline目录source；各次运行保存命令和hash。
旧训练进程不会热加载修改，必须重新启动。用户原训练进程未被停止；本轮只执行必要诊断与验证。

首轮短训实际命令：

```bash
PICK_MASK_GPU=0 PICK_MASK_ENVS=64 PICK_MASK_ITERATIONS=100 \
  bash scripts/train_rgb_mask_ppo.sh \
  --experiment_name "$PWD/outputs/grounded_pick/recovery_diagnosis/temporal_transfer_pilot" \
  --run_name temporal_transfer_pilot agent.save_interval=25
```

权重：`outputs/grounded_pick/recovery_diagnosis/temporal_transfer_pilot/2026-10-13_09-35-45_temporal_transfer_pilot/model_99.pt`。
SHA256 `f7cdb04d1ba217ecdb2bfccfbc0e3cfc41a1a48fb3feb6ab5a5dadecadbda448`。
实际续训命令（新距离规则）：

```bash
PICK_MASK_GPU=0 PICK_MASK_ENVS=64 PICK_MASK_ITERATIONS=50 \
  PICK_MASK_CHECKPOINT="$PWD/outputs/grounded_pick/recovery_diagnosis/temporal_transfer_pilot/2026-10-13_09-35-45_temporal_transfer_pilot/model_99.pt" \
  bash scripts/train_rgb_mask_ppo.sh \
  --experiment_name "$PWD/outputs/grounded_pick/recovery_diagnosis/temporal_transfer_distance_pilot" \
  --run_name temporal_transfer_distance_pilot agent.save_interval=25
```

新正式训练命令（尚未执行）：

```bash
cd ~/IsaacLab_6.1/Pick_Anything
PICK_MASK_GPU=0 PICK_MASK_ENVS=128 PICK_MASK_ITERATIONS=1500 \
  PICK_MASK_CHECKPOINT="$PWD/models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt" \
  bash scripts/train_rgb_mask_ppo.sh
```

未暂存、提交或推送Git。

## 最终候选权重与验收命令

原产物：
`outputs/grounded_pick/recovery_diagnosis/temporal_transfer_distance_pilot/2026-10-13_09-55-27_temporal_transfer_distance_pilot/model_148.pt`。
SHA256 `a9e0008609d299bebedbe20e67ac480e5f9ab8f1616b4581d4ebd17f722bae9d`。
已复制到稳定候选目录（源文件保留、hash一致）：
`models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt`。
同目录保存训练env／agent配置和manifest.json；记录样本数与适用范围。

最终实际评估命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/diagnose_grounded_training.py \
  --checkpoint outputs/grounded_pick/recovery_diagnosis/temporal_transfer_distance_pilot/2026-10-13_09-55-27_temporal_transfer_distance_pilot/model_148.pt \
  --label transfer_final_148_20261013 --num-envs 16 --seeds 42 43
```

完整对照报告：`outputs/grounded_pick/recovery_diagnosis/TEMPORAL_TRANSFER_REPORT_20261013.md`。
没有停止用户原训练；本轮两个短训练及所有评估已退出。

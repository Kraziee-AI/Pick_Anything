# RGB＋GT 掩码 PPO：PREGRASP 卡死诊断与续训准备

## 问题、证据与修改目标

用户要求仔细诊断 RGB 图像＋目标掩码输入的 PPO，提供训练代码；GroundingDINO
等定位后续再加。使用已安装 Conda `isaaclab_61`，全部 GPU 测量使用 GPU 0。

参考已读取的本地 GCRL_UR10e `mask_7d_v1.py`、`training/Train_reach.py`：
其训练用投影矩形二值掩码，与 RGB 拼成 4 通道，经图像 CNN 和状态网络送 PPO。
参考训练另有三帧堆叠；本任务保持原来的单帧、4 通道与 24 维本体输入，以便续训。

实际基线 checkpoint：
`logs/rsl_rl/pick_orange_so101_grounded/2026-10-11_22-00-43_orange_gt_grasp_fix_v1/model_6499.pt`。
SHA256：`f4c15e0607c1873468c909662fc921d6583fb2d160eac772820a599d808c530a`。

GPU 0、seed 42/43 各 16 环境、策略均值、90 秒完整 HOME 首回合：

- 0/32 成功，32/32 PREGRASP 超时，确认夹持／有效抬起／进盒均为 0。
- 32/32 从未满足原 PREGRASP 开爪就位条件，5/32 曾连续至少 3 步满足原始夹持条件。
- 平均目标矩形非空比例 90.33%；人工检查 env_0 初始 RGB 与 mask 对齐。
- seed 42 多个回合爪口已到目标 3–10 mm 内，却把夹爪闭至约 -0.175 rad，
  持续输出闭爪动作。橘子仍在桌面，没有有效抬升。
- 现有前段 reward 为距离势能差，原开度相关 pregrasp_distance 权重为 0；
  当前策略缺少直接开爪进度信号，不能据此宣称仅接入阶段交接就足以恢复。
- CPU 反例复现：连续夹持不交接 PREGRASP；HOME 不复位夹爪；同一 GRASP
  控制步重复调用会增加确认次数。新增 5 项测试在修复前有 3 失败、1 错误、1 通过。

基线数据、源码快照与运行参数：
`outputs/grounded_pick/recovery_diagnosis/baseline_6499/`。

## 实际修改与范围

- `grounded_pick/progress.py`：GroundedStageProgress 复用已经过相机任务验证的
  CameraStageProgress，实现持续夹持交接和每环境按控制步去重。
- `grounded_pick/env.py`：HOME 行复位夹爪位置、速度和 PD 目标；bank 行不覆盖。
  终止统计仍在 reset 前，诊断 episode 初始化在复位后。
- 新增 `grounded_pick/rewards.py`：HOME／PREGRASP 的开爪势能差；
  以既有开度阈值 1.3 rad 封顶，0.2 rad 为单位；开闭往复净零、驻留零支付，
  reset 首步零支付，GRASP 起停止支付，不妨碍后续合爪。
- `grounded_pick/config.py`：新增专用奖励配置，统一缩放后该项 weight=0.2。
  修正配置校验钩子为本地 IsaacLab 实际支持的 `validate_config()`。
- `scripts/test_grounded_pick.py`：将瞬时接触测试明确为两步，保持原验证意图；
  终止计数 fixture 补充出生阶段以适配复位逻辑。
- 新增 `scripts/test_grounded_recovery.py`：7 项状态交接、去重、接触中断、HOME
  复位、势能对称／封顶／门控、配置校验测试。
- 新增 `scripts/diagnose_grounded_training.py`：多种子首回合诊断、源码与权重哈希、
  图像和轨迹保存、统计聚合；拒绝覆盖已有结果。
- 新增 `scripts/train_rgb_mask_ppo.sh` 和 `RGB_MASK_PPO.md`：固定 GT 掩码的
  Conda 训练入口、GPU 0 默认、可配置并行数／追加轮数／checkpoint。

保留 RGB＋mask 4×96×128、24 维机械臂状态、49 维 critic、6 维动作、原 PPO
超参数和物理成功阈值。未修改 GCRL_UR10e、IsaacLab、共享 RGB 任务奖励或旧权重。
没有安装依赖、使用 uv、暂存或提交 Git。

## 验证结果

- 修复后新增 7 项、原 Grounded 10 项、Camera 10 项 CPU 测试通过。
  原 Grounded 检查包含真实 CPU PPO 更新和配置隔离；shell 语法检查通过。
- 基线两种子 GPU 诊断已完成，结果如上。
- 修复后同权重、相同两个种子仍为 0/32 成功；31/32 最大阶段为 PREGRASP，
  1/32 到达 LIFT 并确认夹持，但有效抬升仍为 0。32/32 没有开爪就位。
  因此不能宣称修复代码本身提高了成功率；旧策略仍需要学习新的前段行为。
- GPU 0、128 环境完成 100 轮短续训（819,200 个环境步），正常退出；训练耗时
  544.97 秒。新 checkpoint 为下文的 `model_6598.pt`，actor／critic 参数均有限。
  100 次记录的 HOME 出生成功率均为 0；末次混合出生 success=0.2878，不能据此
  报告完整任务成功率。数据见 `outputs/grounded_pick/recovery_diagnosis/pilot_training_summary.json`。
  续训后相同种子 32 回合：完整成功 0/32，状态机确认夹持 10/32，
  有效抬升 0/32，开爪就位 0/32；最大阶段 PREGRASP 19、GRASP 3、LIFT 10。
  全部超时。平均掩码非空比例 85.88%。结果保存在 `pilot_6598/aggregate.json`。
- 结论：已验证兼容性、训练可运行及修复的 CPU 行为；短训后夹持计数增加，
  但未验证成功率提高，也未学会开爪就位。状态机夹持确认不代表物体已经抬起。
  100 轮不足以作为收敛验收，开爪塑形的收益仍未得到完整任务成功率支持。
  后续长训必须继续检查 HOME 成功率和有效抬升；若仍卡死，应重新诊断前段
  探索／奖励量级或比较从零训练，不能仅凭训练总奖励延长训练并宣称有效。
- 未启动正式长训练；没有进行 GroundingDINO 或多物体成功率验收。

## 命令、checkpoint 与重启

基线实际执行：

```bash
source /home/user123/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_61
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/diagnose_grounded_training.py \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/2026-10-11_22-00-43_orange_gt_grasp_fix_v1/model_6499.pt \
  --label baseline_6499 --num-envs 16 --seeds 42 43
```

同权重修复后评估使用相同参数，label 改为 `fixed_6499`。
诊断子进程统一设置本仓库资产根、缓存和 TMPDIR；不加载 GroundingDINO。

实际短续训命令：

```bash
PICK_MASK_GPU=0 PICK_MASK_ENVS=128 PICK_MASK_ITERATIONS=100 \
  bash scripts/train_rgb_mask_ppo.sh \
  --experiment_name "$PWD/outputs/grounded_pick/recovery_diagnosis/pilot" \
  --run_name rgb_mask_recovery_pilot
```

输出权重：
`outputs/grounded_pick/recovery_diagnosis/pilot/2026-10-12_22-29-08_rgb_mask_recovery_pilot/model_6598.pt`。
SHA256：`08f889741c45efb321ea881f70126c0890ecd2954775da2a67535e56cec0f301`。
训练日志：`outputs/grounded_pick/recovery_diagnosis/pilot_train.log`。
完整对照报告：`outputs/grounded_pick/recovery_diagnosis/REPORT.md`。

短训后评估命令（同一 Conda 环境）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/diagnose_grounded_training.py \
  --checkpoint outputs/grounded_pick/recovery_diagnosis/pilot/2026-10-12_22-29-08_rgb_mask_recovery_pilot/model_6598.pt \
  --label pilot_6598 --num-envs 16 --seeds 42 43
```

正式续训入口（尚未执行正式长训练）：

```bash
PICK_MASK_GPU=0 PICK_MASK_ENVS=128 PICK_MASK_ITERATIONS=1500 \
  bash scripts/train_rgb_mask_ppo.sh
```

需要重新启动训练／回放进程才能加载代码修复。旧模型可加载，续训创建新的
时间戳目录；阶段和奖励语义已经改变，应以修复后的完整 HOME 评估验收。

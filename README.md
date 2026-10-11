# Pick_Anything

Isaac Sim 仿真环境：桌子 + 桌上物体 + SO101 机械臂。

基于 [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)（Apache-2.0）的资产与参数，
针对本地 IsaacLab 3.0 / Isaac Sim 6.1 重新实现（leisaac 包本身要求 isaaclab==2.3.0，不可直接安装）。

## 环境

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything
```

## 资产来源（已下载，均在 `assets/`）

| 资产 | 来源 |
|------|------|
| `assets/robots/so101_follower.usd` | [leisaac release v0.1.0](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.0) |
| `assets/scenes/table_with_cube/` | [leisaac release v0.1.2](https://github.com/LightwheelAI/leisaac/releases/tag/v0.1.2)（已从 scene.usd 中移除 cube） |
| `assets/objects/apple/`（GraspNet 012）、`orange/`（GraspNet 016） | 本地 GraspNet 简化物体，经 `scripts/convert_graspnet_object.py` 生成 |
| `assets/objects/cardboard_box/` | 程序化生成的无顶纸盒（16×16×5cm，5 面板），`scripts/make_cardboard_box.py` |

## 运行

```bash
# GUI（推荐）
python scripts/demo_pick_scene.py --visualizer kit

# 无头冒烟测试
python scripts/demo_pick_scene.py --steps 1200

# 固定随机种子（复现水果位置）
python scripts/demo_pick_scene.py --visualizer kit --seed 7
```

demo 中 SO101 的 shoulder_lift/elbow_flex 做缓慢正弦摆动，验证执行器链路。

**随机刷位**：每次启动在机械臂可达环扇形区域（基座前方 0.12–0.30m，
±55°）内随机采样，裁剪到 1.0×0.45m 长方形桌面内；两果间距 ≥10cm，
水果与纸盒间距 ≥13cm（纸盒完整落在桌内），
采样逻辑在 `pick_anything/env_cfg.py: sample_object_positions()`。
苹果、橘子为独立刚体（照片贴图、45mm 夹爪友好尺寸）；
纸盒为 16×16×5cm 无顶纸皮托盘（独立刚体，每面板独立凸包碰撞，
盒内可供机械臂放入物体）。

## 添加更多 GraspNet 物体

```bash
python scripts/convert_graspnet_object.py --object <编号> --name <名字> --mass <kg>
```

编号对应 GraspNet 88 物体（012=苹果、016=橘子、015=梨、013=柠檬…），
产物在 `assets/objects/<名字>/`（自包含 USD + 贴图）。

## 三种训练方式与当前版本

三种方式都完成「抓橘子 → 放入纸盒 → 撤爪确认」，共用八阶段任务逻辑：
HOME → PREGRASP → GRASP → LIFT → STABILIZE → TRANSPORT → PLACE → RELEASE。
任务名里的 Apple 是历史名称，当前 RL 抓取目标是橘子。

| 训练方式 | 当前任务 ID | 策略输入 | 学习方式 |
|---|---|---|---|
| 八阶段坐标 PPO | `Pick-Apple-SO101-Staged-v0` | 111 维状态：物体相对坐标、本体状态、物理里程碑及三帧历史 | actor/critic 使用 PPO 奖励训练，critic 额外读取特权状态 |
| 纯摄像头 PPO | `Pick-Apple-SO101-Staged-Camera-v8` | 128×96 腕部 RGB + 24 维本体感受 | CNN actor 使用 PPO 奖励训练，critic 读取特权状态；不需要教师 |
| 教师到摄像头蒸馏 | `Pick-Apple-SO101-Staged-Camera-Distill-v10` | 学生：256×192 RGB + 24 维本体感受；教师：111 维状态 | 固定已训练的坐标 PPO 教师，在线 DAgger + MSE 动作回归训练学生 |

摄像头两条路线的学生/actor 不读取物体真实坐标；本体感受包括关节位置、
速度、目标位置和上一动作。蒸馏时学生执行动作，教师针对学生到达的状态给出
动作标签；部署时只使用学生。v10 保留教师原始动作标签及学生执行动作，不裁剪至 ±1，
学生从零初始化，没有再做 PPO 奖励优化。

`Pick-Apple-SO101-Staged-Camera-Filtered-v0` 是 Camera-v8 模型的已验证平滑
回放入口：机械臂 EMA 系数 0.7，夹爪保持原动作；它属于摄像头 PPO 路线，
不是第四种训练方法。早期三级坐标任务和历史摄像头 PPO 实验入口已移除。
任务版本 `Camera-v8` 与训练实验目录名 `cam_v12` 是不同编号。

### 模型与验收记录

| 用途 | 本地 checkpoint | 说明 |
|---|---|---|
| 八阶段坐标 PPO 回放 | `logs/rsl_rl/pick_apple_staged_so101/2026-09-26_16-19-15_retreat_shaping_v2/model_12999.pt` | 已有成功记录的坐标模型，动作尺度 0.06/0.10 |
| 摄像头 PPO / Filtered 回放 | `logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt` | 已验证摄像头模型；Filtered 使用同一份权重 |
| v10 蒸馏训练的教师 | `logs/rsl_rl/pick_apple_staged_so101/2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt` | 平滑坐标 PPO 教师，动作尺度 0.033/0.02，与 v10 匹配 |

v10 的配置、动作契约和教师权重严格加载已通过 CPU 检查；本 README 尚无
v10 学生的完整 HOME 确定性成功率验收记录。教师成绩和混合出生的训练 success
不能替代学生的完整任务验收。上述 checkpoint 在 Git 忽略的 `logs/` 中，未上传模型权重。

以下命令都在仓库根目录、Conda `isaaclab_61` 环境下运行。`--device cuda:0`
可按实际 GPU 修改；使用 `CUDA_VISIBLE_DEVICES=1` 时，物理 GPU1 在进程内映射为 `cuda:0`。

### 1. 八阶段坐标 PPO：Staged-v0

从零训练，使用 reset bank 和课程学习：

```bash
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-v0 \
  --num_envs 4096 --device cuda:0 --seed 42 --max_iterations 13000 --viz none \
  --run_name state_staged_v0

python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-v0 \
  --num_envs 4096 --device cuda:0 --seed 42 \
  --max_iterations 13000 --viz none \
  --experiment_name "$PWD/models/state_ppo" \
  --run_name retrain
```

输出目录：`logs/rsl_rl/pick_apple_staged_so101/`。
actor/critic 为 MLP，学习率 3e-4、自适应 KL 调度；动作尺度 0.06/0.10。
训练允许从中间阶段出生，回放和验收强制从 HOME 完成全链任务。

查看已验证模型：

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
python scripts/play_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-v0 \
  --num_envs 4 --device cuda:0 --viz kit \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-09-26_16-19-15_retreat_shaping_v2/model_12999.pt
```

### 2. 纯摄像头 PPO：Camera-v8

从零训练，使用 128×96 腕部相机、固定出生课程和缩放后的奖励：

```bash
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-v8 \
  --num_envs 512 --device cuda:0 --seed 42 --max_iterations 12000 --viz none \
  --run_name camera_v8

python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-v8 \
  --num_envs 512 --device cuda:1 --seed 42 \
  --max_iterations 13000 --viz none \
  --experiment_name "$PWD/models/camera_ppo" \
  --run_name retrain
```

输出目录：`logs/rsl_rl/pick_apple_staged_so101_camera/`。
当前 runner 默认固定学习率 1e-4，动作尺度 0.033/0.02，episode 90 秒。
已验证的 `cam_v12_ft_lowlr` 模型来自低学习率 3e-5 的续训；上面是当前任务的
从零训练入口，新训练结果仍需验收。

原始动作回放：

```bash
python scripts/play_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-v8 \
  --num_envs 1 --device cuda:0 --viz kit --real-time \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt
```

已验证的平滑回放：

```bash
bash scripts/play_camera_smooth.sh
```

该脚本使用 `Camera-Filtered-v0`、同一份 `model_13000.pt` 和实时播放，
保留原有导出模型。滤波发生在动作执行层，不需要修改模型权重。
详细参数和动作平滑评估见 [CAMERA_PPO_SMOOTHING.md](scripts/CAMERA_PPO_SMOOTHING.md)。

### 3. 教师到摄像头蒸馏：Camera-Distill-v10

使用已有平滑坐标 PPO 教师训练摄像头学生。这里的 `--checkpoint` 必须指向
教师模型；不要直接换成尺度 0.06/0.10 的 `model_12999.pt`。

先运行两轮短程检查：

```bash
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Distill-v10 \
  --num_envs 32 --device cuda:0 --seed 42 --max_iterations 2 --viz none \
  --run_name distill_v10_smoke \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt

python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Distill-v10 \
  --num_envs 256 --device cuda:0 --seed 42 \
  --max_iterations 4000 --viz none \
  --experiment_name "$PWD/models/camera_distill_v10" \
  --run_name retrain \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt
```

正式蒸馏训练：

```bash
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Distill-v10 \
  --num_envs 256 --device cuda:0 --seed 42 --max_iterations 4000 --viz none \
  --run_name distill_v10 \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-10-05_17-16-17_teacher_smooth_v1/model_2250.pt
```

输出目录：`logs/rsl_rl/pick_apple_staged_so101_camera_distill/`。
标准 RSL-RL `Distillation`，学习率 5e-4，每次 rollout 32 步、4 epochs，
每 250 轮保存；动作尺度 0.033/0.02、episode 90 秒、八阶段均匀出生。

回放时 `--checkpoint` 改为学生训练输出：

```bash
python scripts/play_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-Distill-v10 \
  --num_envs 1 --device cuda:0 --viz kit --real-time \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera_distill/<run>/model_XXXX.pt
```

更多配置说明见 [CAMERA_DISTILL_V10.md](scripts/CAMERA_DISTILL_V10.md)。

### 完整任务确定性评估

三种方式都使用 `scripts/eval_pick_home.py`：每个环境只统计第一个 HOME
完整 episode，执行策略确定性均值。下面以八阶段坐标 PPO 为例：

```bash
python scripts/eval_pick_home.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-v0 \
  --num_envs 256 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-09-26_16-19-15_retreat_shaping_v2/model_12999.pt \
  --external_callback scripts.eval_pick_home.install_episode_counter
```

| 评估方式 | `--task` | `--checkpoint` |
|---|---|---|
| 坐标 PPO | `Pick-Apple-SO101-Staged-v0` | 坐标 PPO checkpoint |
| 摄像头 PPO | `Pick-Apple-SO101-Staged-Camera-v8` | 摄像头 PPO checkpoint |
| 摄像头蒸馏 | `Pick-Apple-SO101-Staged-Camera-Distill-v10` | 蒸馏学生 checkpoint，评估时不加载教师 checkpoint |

查看输出 `PICK_APPLE_EVAL_RESULT=` 中的 `success_rate`。
Filtered 的成功率和动作变化指标使用专门的
[平滑评估命令](scripts/CAMERA_PPO_SMOOTHING.md#评估与回归检查)。

### 训练监控与 reset bank 查看

```bash
tensorboard --logdir logs/rsl_rl --port 6006 --host 127.0.0.1
python scripts/view_reset_bank.py \
  --device cuda:0 --visualizer newton --per_stage 15 --steps_per_pose 30
```

CPU 配置及动作契约检查：

```bash
python scripts/test_camera_smoothing.py
python scripts/test_distill_v10.py
```

## 结构

```
pick_anything/
├── paths.py      # 资产路径常量
├── robots.py     # SO101_FOLLOWER_CFG（关节名、执行器参数来自 leisaac；自碰撞关闭）
├── env_cfg.py    # PickSceneCfg：场景 USD 引用 + 机械臂 + 水果/纸盒 + 灯光
└── tasks/pick_apple/
    ├── mdp.py            # 共享几何、刷位和状态工具
    ├── pick_env_cfg.py   # 共享场景与仿真基础配置
    ├── staged_env_cfg.py # 八阶段动作、观测、奖励、reset 与终止
    ├── staged_mdp.py     # 八阶段物理状态和奖励实现
    └── agents/staged_ppo_cfg.py # 坐标 PPO 超参
scripts/
├── demo_pick_scene.py
├── train_pick.py              # 训练入口（注册任务 + isaaclab train CLI）
├── play_pick.py               # 策略回放入口
├── convert_graspnet_object.py # GraspNet OBJ → 刚体 USD
└── make_cardboard_box.py      # 生成无顶纸盒 USD
```

## 坐标约定（沿 leisaac）

- 桌面（已改为 1.0×0.45m 长方形）：x ∈ [-0.145, 0.855]，y ∈ [-0.677, -0.227]，顶面 z≈0.041
- 机器人基座 (0.35, -0.64, 0.01)，从桌子后方伸向桌面，零位夹爪可达 y≈-0.34（0.30m）
- 关节：shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper

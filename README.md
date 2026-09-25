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

## 强化学习：抓苹果入纸盒（Pick-Apple-SO101-v0）

manager-based RL 环境，场景资产与 demo 完全共用。任务状态机（`pick_anything/tasks/pick_apple/mdp.py`
的 `PickState`，每控制步刷新一次）：接近苹果 → 夹持举起（苹果离桌 + 夹爪贴近 + 净接触力）→
运送到纸盒上方悬停 → 松手落入盒内（盒局部系内、静止、夹爪撤离）判定成功。

奖励（进度制：每米进展只付一次，防刷分）：

| 项 | 权重 | 说明 |
|----|------|------|
| approach | +12/m | 夹爪→苹果 的距离进展（苹果在桌时） |
| height | +25/m | 举起高度进展（越过 5cm 盒壁） |
| hover | +20/m | 苹果→盒心上方悬停点的距离进展 |
| grasp | +4 | 首次夹起苹果的一次性奖励 |
| place | +30 | 苹果松手落盒并静止（终止） |
| box_disturbed | -1.5/步 | 抓到苹果前纸盒被挪动 >1.2cm |
| box_disturbed_late | -0.5/步 | 抓取后仍大力挪动纸盒 |
| orange_disturbed | -1.0/步 | 橘子被挪动（任何时候，对应"碰到别的东西不好"） |
| action_rate | -0.005 | 平滑正则 |
| apple_dropped / box_ruined / abnormal | -8 / -5 / -15 | 苹果掉离桌面 / 纸盒翻倒 / 关节速度爆表（终止罚） |

终止：成功入盒、苹果离桌、纸盒翻倒/飞出、关节速度 >25 rad/s、超时 12s。
课程：reset 事件按成功率 EMA 扩大刷位范围（难度 0.2→1.0）。

```bash
# 训练（在 Pick_Anything 根目录下运行，日志/模型写入 ./output/）
python scripts/train_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-v0 \
    --num_envs 1024 --max_iterations 13000 --seed 42 --device cuda:0 --viz none \
    agent.experiment_name=/home/user123/IsaacLab_6.1/Pick_Anything/output

# 回放训练好的策略（GUI）
python scripts/play_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-v0 \
    --num_envs 4 --checkpoint output/<run>/model_XXXX.pt --viz kit

# 环境逻辑探针（无头，验证状态机/奖励/终止接线）
python scripts/probe_pick_env.py --num_envs 4
```

**资源约束（实测）**：本机 62GB 内存与他人任务共存时，该场景约 30MB/环境——
1024 envs 稳定（~38GB），2048/4096/8192 会被 OOM killer 杀掉。
若机器空出来，可逐步上调 `--num_envs`（lift 项目 2×4096 双卡曾可行）。

## 结构

```
pick_anything/
├── paths.py      # 资产路径常量
├── robots.py     # SO101_FOLLOWER_CFG（关节名、执行器参数来自 leisaac；自碰撞关闭）
├── env_cfg.py    # PickSceneCfg：场景 USD 引用 + 机械臂 + 水果/纸盒 + 灯光
└── tasks/pick_apple/
    ├── mdp.py            # PickState 状态机 + 自定义奖励/终止/事件/观测
    ├── pick_env_cfg.py   # ManagerBasedRLEnvCfg（动作/观测/事件/奖励/终止）
    └── agents/rsl_rl_ppo_cfg.py  # PPO 超参（沿用 Isaac-Lift-SO101 验证配方）
scripts/
├── demo_pick_scene.py
├── train_pick.py              # 训练入口（注册任务 + isaaclab train CLI）
├── play_pick.py               # 策略回放入口
├── probe_pick_env.py          # 环境逻辑探针
├── convert_graspnet_object.py # GraspNet OBJ → 刚体 USD
└── make_cardboard_box.py      # 生成无顶纸盒 USD
```

## 坐标约定（沿 leisaac）

- 桌面（已改为 1.0×0.45m 长方形）：x ∈ [-0.145, 0.855]，y ∈ [-0.677, -0.227]，顶面 z≈0.041
- 机器人基座 (0.35, -0.64, 0.01)，从桌子后方伸向桌面，零位夹爪可达 y≈-0.34（0.30m）
- 关节：shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper










#八阶段每个阶段reset状态
  cd /home/user123/IsaacLab_6.1/Pick_Anything
  export PYTHONNOUSERSITE=1 PXR_WORK_THREAD_LIMIT=1

  python scripts/view_reset_bank.py \
    --device cuda:2 \
    --visualizer newton \
    --per_stage 15 \
    --steps_per_pose 30


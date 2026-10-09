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

## 坐标 PPO：八阶段抓橘子入纸盒

唯一坐标训练入口：`Pick-Apple-SO101-Staged-v0`。
actor 读取橘子和盒子的相对坐标、本体状态及三帧历史，critic 额外读取特权状态。
任务依次经过 HOME、PREGRASP、GRASP、LIFT、STABILIZE、TRANSPORT、PLACE、RELEASE；
使用物理验证的 reset bank 和课程学习，部署与评估从 HOME 执行完整任务。

成功模型示例：
`logs/rsl_rl/pick_apple_staged_so101/2026-09-26_16-19-15_retreat_shaping_v2/model_12999.pt`。
训练、回放和确定性评估命令见下文。早期三级任务及专属 PPO 配置已移除。

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




#八阶段每个阶段reset状态
  cd /home/user123/IsaacLab_6.1/Pick_Anything
  export PYTHONNOUSERSITE=1 PXR_WORK_THREAD_LIMIT=1

  python scripts/view_reset_bank.py \
    --device cuda:2 \
    --visualizer newton \
    --per_stage 15 \
    --steps_per_pose 30
    
#训练
# CUDA_VISIBLE_DEVICES 让进程只看得见物理 1 卡（映射为 cuda:0）——单进程模式下

CUDA_VISIBLE_DEVICES=1 python scripts/train_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 \
      --num_envs 4096 --device cuda:0 --seed 42 --max_iterations 13000 --viz none \
      --run_name det_align_v1

#tensorboard
tensorboard --logdir ~/IsaacLab_6.1/Pick_Anything/logs/rsl_rl/pick_apple_staged_so101/2026-09-27_11-40-52_det_align_v1 --port 6006 --host 127.0.0.1

#推理（同样用 CUDA_VISIBLE_DEVICES 锁单卡，推理放 0 卡，因为欺骗器在0显卡插着）
  export LD_PRELOAD=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12

  CUDA_VISIBLE_DEVICES=0 python scripts/play_pick.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 \
      --num_envs 4 --device cuda:0 \
      --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-09-26_16-19-15_retreat_shaping_v2/model_12999.pt \
      --viz kit
  #可以任意显卡
  LD_PRELOAD=/home/user123/miniconda3/envs/isaaclab_61/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
  python scripts/play_pick.py --rl_library rsl_rl \
    --task Pick-Apple-SO101-Staged-v0 \
    --num_envs 4 --device cuda:2 --viz kit \
    --checkpoint logs/rsl_rl/pick_apple_staged_so101/2026-09-27_14-57-13_det_align_v1/model_1750.pt


#确定性评估（验收口径：play 只跑均值动作，train 曲线的 sampled success 只当参考）
  export PYTHONNOUSERSITE=1 PXR_WORK_THREAD_LIMIT=1
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_pick_home.py --rl_library rsl_rl --task Pick-Apple-SO101-Staged-v0 \
      --num_envs 256 --device cuda:0 \
      --checkpoint logs/rsl_rl/pick_apple_staged_so101/<run>/model_XXXX.pt \
      --external_callback scripts.eval_pick_home.install_episode_counter \
      --viz none
  # 输出找 PICK_APPLE_EVAL_RESULT= 一行（success_rate 即 home 出生完整任务的确定性成功率）
  # PICK_DEBUG_RELEASE=1 可选：play 时 1Hz 打印每环境 RELEASE 未满足项





## 摄像头 PPO：已验证版本

摄像头 PPO 保留以下两个入口：

- `Pick-Apple-SO101-Staged-Camera-v8`：纯摄像头 PPO，actor 使用腕部 RGB 和本体感受，critic 使用特权状态。
- `Pick-Apple-SO101-Staged-Camera-Filtered-v0`：同一模型加机械臂 EMA 平滑，夹爪保持原动作。

使用的已验证 checkpoint：
`logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt`。
任务版本 v8 和实验目录 cam_v12 是不同编号。

```bash
conda activate isaaclab_61
bash scripts/play_camera_smooth.sh
```

原始动作回放：

```bash
python scripts/play_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-v8 \
  --num_envs 1 --device cuda:0 --viz kit --real-time \
  --checkpoint logs/rsl_rl/pick_apple_staged_so101_camera/2026-10-07_00-53-39_cam_v12_ft_lowlr/model_13000.pt
```

相同任务的 PPO 训练入口：

```bash
python scripts/train_pick.py --rl_library rsl_rl \
  --task Pick-Apple-SO101-Staged-Camera-v8 \
  --num_envs 512 --device cuda:0 --seed 42 --max_iterations 12000 --viz none \
  --run_name camera_v8
```

该命令使用成功版本的任务和 PPO 配方；新训练仍需完整任务验收，不保证复现相同成功率。
模型权重在 Git 忽略的 `logs/` 下，未上传到仓库。
平滑回放和评估见 [CAMERA_PPO_SMOOTHING.md](scripts/CAMERA_PPO_SMOOTHING.md)。
独立教师到摄像头蒸馏入口仍为 `Camera-Distill-v10`，见
[CAMERA_DISTILL_V10.md](scripts/CAMERA_DISTILL_V10.md)。

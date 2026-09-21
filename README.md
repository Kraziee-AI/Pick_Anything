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

## 结构

```
pick_anything/
├── paths.py      # 资产路径常量
├── robots.py     # SO101_FOLLOWER_CFG（关节名、执行器参数来自 leisaac）
└── env_cfg.py    # PickSceneCfg：场景 USD 引用 + 机械臂 + 水果/纸盒 + 灯光
scripts/
├── demo_pick_scene.py
├── convert_graspnet_object.py  # GraspNet OBJ → 刚体 USD
└── make_cardboard_box.py       # 生成无顶纸盒 USD
```

## 坐标约定（沿 leisaac）

- 桌面（已改为 1.0×0.45m 长方形）：x ∈ [-0.145, 0.855]，y ∈ [-0.677, -0.227]，顶面 z≈0.041
- 机器人基座 (0.35, -0.64, 0.01)，从桌子后方伸向桌面，零位夹爪可达 y≈-0.34（0.30m）
- 关节：shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll / gripper

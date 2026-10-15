# 独立多物体桌面场景

以下保留最初场景准备阶段的说明。后续语言定位与独立多物体任务已接入，
当前入口和实测限制见 [OPEN_VOCAB_PICK.md](OPEN_VOCAB_PICK.md)；
最新单橘子权重的多物体抓橘子测试为 0/10，不代表已具备多物体抓取能力。

此场景用于准备后续“语言选目标 → 视觉掩码 → 抓取”。默认生成橘子、小黄鸭、
蓝色方块，并保留原来的纸盒，可选增加红色圆柱。入口是 `scripts/view_tabletop_objects.py`，配置位于
`pick_anything/tabletop/`。现有 RGB PPO 和橘子掩码训练不会加载此场景。

只新增文件，没有改动已有场景、训练入口、任务注册、奖励、reset bank、相机、
模型或训练日志。无需重启正在运行的训练；以后用原训练命令启动仍是单橘子任务。
新场景复用原来的 16 × 16 × 5 cm 纸盒 USD，不复用橘子的八阶段奖励或 reset bank。
纸盒默认显示；仅显式传入 `--no-box` 才隐藏。

| 物体 key | 资产与尺寸 | 语言示例 | 检测提示词 |
| --- | --- | --- | --- |
| `orange` | 原橘子 USD，保留 0.775 缩放 | 抓橘子 | `orange fruit.` |
| `yellow_duck` | 本地程序生成的小黄鸭，约 50 × 77 × 62 mm，25 g | 抓小黄鸭 | `yellow rubber duck.` |
| `blue_cube` | 50 mm 蓝色方块，35 g | 抓蓝色方块 | `blue cube.` |
| `red_cylinder` | 直径 42 mm、高 60 mm 红色圆柱，30 g | 抓红色圆柱 | `red cylinder.` |

新增资产位于 `assets/objects/` 对应的独立目录，自带质量、碰撞体、摩擦材质和
相对路径的颜色贴图。小黄鸭由身体、头、嘴、尾等几何构成，四个凸碰撞体归于同一个
刚体；翅膀和眼睛仅用于外观。这是可供物理仿真的简化玩具模型，抓取姿态与稳定性
仍需后续实测。没有下载外部资产或安装依赖。

放置使用固定种子可复现的随机布局，物体中心限制在机械臂前方可达扇区，按保守
占地圆留出 12 mm 间距，并检查桌边余量。纸盒保持轴对齐，位置仍在原盒子可达
采样区域；物体与纸盒整个矩形占地也留出间距，不会出生在盒内。随机 yaw 使用 IsaacLab 当前版本的
`xyzw` 四元数。出生时底部距桌面 4 mm，再由物理仿真落稳；这些检查只约束出生
布局，不保证每个位置都有可行的抓取姿态。

## 现在可以执行的 CPU 检查

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything

# 不启动 Isaac Sim、不使用 GPU、不加载 PPO 或 GroundingDINO
python scripts/view_tabletop_objects.py --inspect --instruction "抓小黄鸭"
python scripts/test_tabletop_objects.py

# 可选加入圆柱
python scripts/view_tabletop_objects.py --inspect \
  --objects orange yellow_duck blue_cube red_cylinder --instruction "抓红色圆柱"
```

`--inspect` 输出布局、资产路径以及目标的 key、prim 路径和英文检测词。中文和
英文的单目标词汇映射会拒绝多目标指令、未知物体或当前场景里不存在的目标。
它尚未执行视觉定位、自由语言理解或抓取动作。

## 有空闲 GPU 时再手动预览

以下命令会额外启动仿真并占用资源。本次修改期间没有执行它们。选择实际空闲的
GPU；不要为了预览停止现有训练。

```bash
# 下面 GPU 2 只是示例，使用前先确认空闲
CUDA_VISIBLE_DEVICES=2 python scripts/view_tabletop_objects.py \
  --device cuda:0 --viz kit --steps 0 --instruction "抓小黄鸭"

# 有限步数的无窗口物理检查
CUDA_VISIBLE_DEVICES=2 python scripts/view_tabletop_objects.py \
  --device cuda:0 --viz none --steps 600 \
  --objects orange yellow_duck blue_cube red_cylinder --instruction "抓蓝色方块"
```

机械臂保持初始姿态；语言只选择并输出目标，不驱动机械臂。独立预览的缓存位于
`outputs/tabletop_objects/cache/`，不写入两项训练的模型或缓存目录。

资产已随本次修改生成。如需重建，执行
`python scripts/make_tabletop_assets.py --overwrite`；只能在没有程序使用这些新资产时重建。生成器不覆盖原橘子、机器人
或桌面 USD。

## 后续接入多物体语言抓取

新场景为每个物体提供稳定 key、独立 prim 和 `class` 语义标签。后续可把
`resolve_target()` 返回的 caption 传给视觉检测器；真实语言定位应只使用 RGB
和文本，仿真 prim 只用于训练标签或调试。

当前橘子 GroundingDINO 包装器仍只接受橘子，GT mask 也仍使用橘子球体投影。
因此现在的 `train_grounded_pick.py` / `play_grounded_pick.py` 不能直接传“抓小黄鸭”。
接入需建立另一个任务和模型目录，增加按目标切换的观察、适合各形状的训练掩码、
奖励与重置逻辑，并重新验证抓取。旧橘子 reset bank 不适用于鸭子和方块；现有
checkpoint 的多物体抓取能力未经验证。

## 验证范围

CPU 测试检查语言映射、200 个四物体加纸盒的随机布局、USD 刚体／碰撞／尺寸／摩擦／
贴图，以及新配置对现有 RGB 和掩码配置、任务注册的隔离。没有运行 GPU 仿真、
渲染或抓取评估，不宣称这些物体已能被当前策略抓取。

本次 4 项 CPU 测试全部通过；新增模块创建并修改配置后，原训练配置仍一致。
开始修改前记录的 76 个已有源码和物体资产文件，结束时内容校验均未变化。
CPU 绘制的新增资产几何预览位于
`outputs/tabletop_objects/asset_preview.png`，它不是 Isaac Sim 渲染截图。

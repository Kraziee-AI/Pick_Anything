# 橘子语言 Grounding → 掩码 → 抓取（独立实验）

新增任务 ID：`Pick-Orange-SO101-Grounded-v0`。

当前只支持已有橘子，不增加其他物体。任务仍完成抓橘子、抬升、放入纸盒、
撤爪确认。新代码位于 `pick_anything/grounded_pick/`，通过独立入口注册；
原三种方式的源码、任务注册、训练入口、超参数和模型目录均未修改。
本仓库原先已有的未提交修改也保留原样。

## 与本地 GCRL_UR10e 的对应关系

参考文件：

- `~/GCRL_UR10e/README.md`
- `~/GCRL_UR10e/mj_envs/robohive/envs/arms/mask_7d_v1.py`
- `~/GCRL_UR10e/sim2real/src/reachGrasp_env/GdinoReachGraspEnv.py`

参考项目 README 提及 SAM，但上述实际代码的 `create_mask()` 将检测框填成
白色矩形。本实现采用同样的**二值矩形目标区域**，不是 SAM 像素轮廓分割。

```text
训练：橘子仿真位置 + 腕摄姿态/内参 → 投影矩形目标掩码
回放：语言指令 + 腕摄 RGB → GroundingDINO → 检测框 → 矩形目标掩码
                                            ↓
                       RGB(3) + mask(1) + 本体感受(24)
                                            ↓
                                新的 CNN PPO → 6 维关节动作
```

两条路径每个相机新帧更新掩码。DINO 路径只接收 RGB 和语言，不读取橘子位置、
仿真目标 ID、阶段或真实坐标；检测失败返回全零掩码，不回退 GT，也不沿用旧框。
同一个相机帧的重复观测不重复运行检测。

GT 路径用当前资产半径 `ORANGE_RADIUS_M=0.0281` 的球体近似，计算透视包围框。
它包括遮挡部分，不是可见实例分割；近裁剪面附近或镜头后方不生成框。
训练给 GT 框加入 10% 抖动及 2% 掩码丢失，RGB 加逐 episode 曝光／白平衡随机化，
以缓解 GT 与检测框差异。回放关闭这些随机化。

场景物理、八阶段奖励、reset bank、动作尺度 0.033/0.02 从已有稳定基础配置继承。
Grounded 单独使用 `GroundedStageProgress`：新增夹持标志在 GRASP 阶段连续确认
完成后才锁存，防止 PREGRASP 瞬时接触阻断就位条件并提前关闭接近奖励。
reset bank 的已验证阶段种子沿用原语义。共享任务状态机及旧训练方式保持原样。
actor 仅使用图像、掩码和本体感受；critic 和奖励在训练期间仍使用特权状态。
语言只用于选择橘子目标，放盒动作由任务定义；尚未解析任意目的地或多物体指令。
中文“抓橘子／拿橙子”转换为英文检测词 `orange fruit.`，英文目标描述保留给检测器。

## 环境和输出

```bash
conda activate isaaclab_61
cd ~/IsaacLab_6.1/Pick_Anything
```

使用已安装的 Torch、Transformers、Pillow、NumPy 和 RSL-RL；不使用 uv，
不安装或修改 Conda 环境，不修改 `~/GCRL_UR10e` 或 IsaacLab 的代码。
模型下载和运行缓存都定向到 `outputs/grounded_pick/cache/`，
日志及权重写到 `logs/rsl_rl/pick_orange_so101_grounded/`。

新增网络第一层接受 **4 通道**，需要单独训练。之前 RGB PPO 或蒸馏模型的
checkpoint 不能直接作为本任务权重加载；本路线也不使用坐标教师。
训练期间的中间阶段出生成功率不能替代完整 HOME 成功率。

## 1. 检查配置与小规模训练

```bash
python scripts/test_grounded_pick.py

CUDA_VISIBLE_DEVICES=2 python scripts/train_grounded_pick.py \
  --num_envs 4 --device cuda:0 --seed 42 --max_iterations 2 --viz none \
  --run_name grounded_smoke \
  --dump-dir outputs/grounded_pick/smoke_frames --dump-every 1 --dump-max 4
```

默认 `--mask-backend gt`。先用 GT 把橘子完整链条训通，再测 DINO 掩码。
保存的 `*_rgb.png`、`*_mask.png` 和 JSON 可检查目标框、是否检测到及置信度。
相机渲染 256×192，策略输入 128×96；RGB 双线性缩放，掩码最近邻缩放，保持 0/1。

正式训练示例（按可用显存调整环境数量，不自动启动长训练）：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_grounded_pick.py \
  --num_envs 128 --device cuda:0 --seed 42 --max_iterations 12000 --viz none \
  --run_name orange_gt_v1
```

## 2. 准备语言 Grounding 模型并检查单张图

使用已安装 Transformers 中的 GroundingDINO，实现不依赖原项目的 CUDA 扩展。
默认模型为 `IDEA-Research/grounding-dino-tiny`。回放只从缓存加载，不自动联网；
先执行一次明确允许下载的命令：

```bash
python scripts/ground_orange_image.py --download --device cpu \
  --image outputs/grounded_pick/smoke_frames/0001_step_000000_rgb.png \
  --instruction "抓橘子"
```

不传 `--image` 时只下载／检查加载。再次检查可省略 `--download`。
输出默认在 `outputs/grounded_pick/image_check/`，包含 RGB、二值 mask 和检测框 JSON。
支持 `--grounding-model /path/to/local_transformers_model` 读取本地模型目录。
原 GCRL_UR10e 的单个 `.pth` 是不同存储格式，不能直接作为此参数使用。

如环境同时设置了不兼容的 `ALL_PROXY=socks://...` 和有效 HTTP 代理，可只在当前
下载进程中移除 SOCKS 变量，保留 HTTP 代理；无需修改系统网络配置：

```bash
env -u ALL_PROXY -u all_proxy python scripts/ground_orange_image.py --download --device cpu
```

## 3. 语言回放

以下 `<run>/model_XXXX.pt` 替换为本任务正式训练的权重，不要使用两轮冒烟权重验收。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/play_grounded_pick.py \
  --instruction "抓橘子" --num_envs 1 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/<run>/model_XXXX.pt \
  --dump-dir outputs/grounded_pick/dino_play --dump-every 30 --dump-max 100
```

默认 `--mask-backend dino`；可用 `--instruction "pick up the orange"`。
`--grounding-device cpu` 可将检测与 GPU 仿真分开，但可能很慢。
DINO 每帧推理开销较大，回放先使用 1 个环境，不能保证 30 Hz。
`--real-time` 只限制过快的回放，不会加速检测器。

GUI 回放沿用已有 Kit 启动约定：

```bash
export LD_PRELOAD="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
# 将上面的 --viz none 换成 --viz kit，可加 --real-time。
```

GT 对照回放加 `--mask-backend gt`。GT 是调试／对照，不代表完成真实语言定位验收。

## 4. 完整 HOME 确定性评估

同一 checkpoint 分别测 GT 与 DINO，比较目标掩码差异造成的成功率损失：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_grounded_pick.py \
  --mask-backend gt --num_envs 8 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/<run>/model_XXXX.pt

CUDA_VISIBLE_DEVICES=2 python scripts/eval_grounded_pick.py \
  --mask-backend dino --instruction "抓橘子" --num_envs 1 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/<run>/model_XXXX.pt \
  --dump-dir outputs/grounded_pick/dino_eval
```

评估强制从 HOME 出生、使用策略均值，每个环境只统计首个 episode，
**在自动 reset 前**读取终止原因和抓取／抬升／到盒／入盒里程碑。
收满当前 `--num_envs` 个 episode 后输出 `GROUNDED_PICK_EVAL_RESULT=<json>` 并退出。
`rates.success` 为完整任务成功率。1 个 DINO episode 仅供调试，正式验收需更多样本与种子。
先确认橘子完整任务的 GT 和 DINO 成绩，再扩展其他物体。

### 详细 HOME 诊断

在评估入口加 `--diagnostics-dir` 即可记录每个首回合；训练入口不启用诊断，
不会改变奖励、动作、阶段判据或旧训练方式。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_grounded_pick.py \
  --mask-backend gt --num_envs 8 --device cuda:0 --seed 42 --viz none \
  --checkpoint logs/rsl_rl/pick_orange_so101_grounded/<run>/model_XXXX.pt \
  --diagnostics-dir outputs/grounded_pick/diagnostics/seed42 \
  --trace-every 5 \
  --dump-dir outputs/grounded_pick/diagnostics/seed42/frames \
  --dump-every 30 --dump-max 100
```

`summary.json` 包含每回合的阶段停留时间、首次夹持锁存时间／阶段、连续夹持最长
步数、最高橘子高度／升高量、最小爪口距离、开爪就位条件、掩码非空比例及终止快照。
统计每个控制步，`trace.jsonl` 每 5 步记录一次并额外保留状态变化和终止帧。
`--dump-dir` 仍只保存 env_0 的 RGB、mask 和框 JSON。

当前 `grasped` 表示确认完成 GRASP 或恢复 reset bank 的夹持种子；旧诊断中的
`grasped` 只是瞬时物理条件锁存，新旧含义不同。结果中的 `grasp_semantics`
标注当前规则。`grasp_raw` 始终记录原始瞬时条件，连续满足也不等同于已经从
PREGRASP 推进。`lifted` 仍是物理条件锁存；
抬起判断还要求高度、爪口距离和接触力。GT 掩码包含遮挡部分，掩码非空不等同于
橘子在 RGB 中可见。终止快照中的 mask 属于前一个策略观测，已在轨迹中注明。

## 当前验证记录

- CPU 契约检查：投影和坐标约定、二值掩码、语言目标限制、空检测、帧缓存、
  DINO 分支不读取场景真值、终止状态在 reset 前计数、实际 PPO 更新及原配置隔离。
- GPU 2：4 环境、2 轮 GT PPO 已跑通，生成 `model_0.pt`、`model_1.pt`；
  腕摄 RGB 与 GT mask 人工检查对齐。
- 冒烟权重的 GT 回放与评估已跑通。评估临时设 `env.episode_length_s=0.2` 验证
  超时计数，4 个 HOME episode 均计入超时；这不是完整抓取能力验收。
- GroundingDINO 权重已下载到本仓库；真实腕摄图片的“抓橘子”定位成功，
  置信度约 0.802，框为 `[99.63, 8.49, 134.24, 48.24]`，对应 256×192 图像。
  图片与掩码见 `outputs/grounded_pick/image_check/`。
- DINO 模型离线加载，以及 1 环境语言→DINO→mask→策略→动作→评估的 GPU
  短回放已通过；短回放同样使用 0.2 秒超时，仅验证接口。逐帧结果在
  `outputs/grounded_pick/dino_smoke/`。
- 正式长训练已由用户启动。2026-10-11 针对 `model_3400.pt` 做了 seed 42/43
  各 8 个 HOME 首回合详细评估：16/16 超时，最远停在 PREGRASP；15/16 在
  PREGRASP 提前锁存 `grasped`，导致就位条件被阻断，未达成有效抬起或进盒。
  这不是完整任务抓取成功。逐回合统计、轨迹、图像及分析见
  `outputs/grounded_pick/diagnostics/REPORT.md`。
  该报告保留修复前的历史结论；本次修复过程与后续验证见
  `docs/change_records/2026-10-11_grounded_grasp_latch_fix.md`。
- 新的详细诊断及终止记录契约检查通过，CPU 检查总计 6 项。此次只增加
  Grounded 评估诊断，没有改训练阶段、奖励或现有训练进程。
- 随后完成 Grounded 专用夹持锁存修复，当前 CPU 检查总计 10 项；修复后的
  8 回合 HOME 评估没有提前夹持锁存，但旧策略尚未完成准确就位或有效抬起。
  从 `model_3500.pt` 的 2 轮 PPO 续训通过，正式续训命令与验证详情见上述修改记录。

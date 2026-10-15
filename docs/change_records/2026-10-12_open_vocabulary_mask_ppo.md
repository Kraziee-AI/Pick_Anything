# 开放词汇定位与独立多物体掩码 PPO 入口

## 问题、复现与证据

- 用户要求参考本地 `~/GCRL_UR10e`，实现开放词汇输入 → 掩码 → PPO 抓取。
- 原 `orange_caption()` 显式拒绝苹果、鸭子、方块等目标；旧 GT 掩码读取
  `scene["orange"]`，奖励和 reset bank 也绑定橘子，不能直接作为多物体训练。
- 已读取参考项目 README、`mj_envs/robohive/envs/arms/mask_7d_v1.py` 和
  `sim2real/src/reachGrasp_env/GdinoReachGraspEnv.py`。实际 `create_mask()` 填矩形框，
  因此本次沿用二值矩形区域，没有实现 SAM 轮廓分割。
- 修改前原 `scripts/test_grounded_pick.py` 的 10 项检查通过。
- 用户随后询问现成模型；实际读取原 `model_6499.pt`，确认 checkpoint 迭代 6499、
  actor 第一层 `(16,4,5,5)`、本体归一化 `(1,24)`、动作输出 `(6,128)`。

## 修改目标与实际内容

- 新增 `pick_anything/grounded_pick/language.py`：开放英文目标描述、常用中文映射、
  显式英文 caption；不将仿真资产目录当作检测器词汇白名单。
- `grounding.py`：检测器采用通用 caption，增加显式 caption 参数。
- `observations.py`：提取可继承的目标投影／检测方法，保持旧橘子校验；
  同帧语言指令改变时刷新检测结果。原 4 通道拼接和空检测处理保持。
- `scripts/ground_orange_image.py`：支持开放词汇与 `--caption`，保留原入口默认值。
- 新增 `pick_anything/open_vocab_pick/`：独立任务注册、四物体场景、训练真值目标、
  有方向的 GT 包围盒、独立 HOME 重置、目标相对奖励、夹持确认、稳定抬升与误抓惩罚。
  Actor 为 4 通道图像＋24 维状态，critic 为 54 维特权观察，动作仍为 6 维。
  成功条件是抓住指定目标并稳定抬升 8 cm，连续 5 步；没有移植旧橘子入盒任务。
- 新增 `scripts/open_vocab_pick_cli.py` 与 train/play/eval 三个入口。
  新任务使用独立模型目录；DINO 模式拒绝随机奖励目标，评估在自动 reset 前计数。
- 新增 `scripts/test_open_vocab_pick.py` 与 `OPEN_VOCAB_PICK.md`。
- 本地 IsaacLab `configclass` 重写 `validate()`；新任务采用实际支持的
  `validate_config()` 扩展钩子，并通过非法组合检查确认它会被调用。
- 未修改参考项目、IsaacLab 外部文件、现有 RGB PPO 训练配置、旧奖励或旧 checkpoint。
  没有暂存、提交或推送。

## 验证方法与实际结果

环境统一为 Conda `isaaclab_61`，未使用 uv 或安装依赖。

```bash
source /home/user123/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_61
PYTHONDONTWRITEBYTECODE=1 python scripts/test_open_vocab_pick.py
```

- 新增 6 项 CPU 检查通过：自由 caption 与旧校验、检测器开放词汇及空结果、
  有方向投影及近裁剪、DINO 分支不读取场景真值与文本缓存、夹持／抬升连续确认
  和误抓判定、配置隔离及真实 CPU PPO 更新。
- 修改后重新运行 `PYTHONDONTWRITEBYTECODE=1 python scripts/test_grounded_pick.py`，
  原 10 项检查全部通过。
- Actor 测试第一层为 `Conv2d(4,16,...)`，输出为 6 维；critic 测试输入为 54 维，
  PPO 更新损失均为有限数。这不是抓取成功率验证。
- `train_open_vocab_pick.py --help` 正常显示新增参数和训练参数。
- 新增任务的 GPU 命令已尝试，沙箱内报 `No CUDA GPUs are available`，
  在仿真创建前退出，只产生 run 元数据，没有 checkpoint。
- 已请求沙箱外 GPU 2 冒烟验证，用户中断了该工具调用；没有得到完成结果，
  不记为通过，不继续启动 GPU 验证。
- 多物体 GPU 物理、腕摄渲染、实际 DINO 多物体检测、长训练和完整 HOME
  抓取成功率全部未验证。任意目标定位也不证明任意物体抓取能力。

## checkpoint、命令与进程

原模型：
`logs/rsl_rl/pick_orange_so101_grounded/2026-10-11_22-00-43_orange_gt_grasp_fix_v1/model_6499.pt`。
它不是新多物体任务权重；新任务 critic 维度不同，不能直接加载完整旧 runner。
本次没有新任务训练完成的 checkpoint，也没有修改原权重。

已尝试但失败的短训练命令：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 python scripts/train_open_vocab_pick.py \
  --num_envs 4 --device cuda:0 --seed 42 --max_iterations 2 --viz none \
  --run_name open_vocab_smoke \
  --dump-dir outputs/open_vocab_pick/smoke_frames --dump-every 1 --dump-max 8
```

未来评估示例（未执行）：

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/eval_open_vocab_pick.py \
  --instruction "抓蓝色方块" --mask-backend dino --num_envs 1 --device cuda:0 --viz none \
  --checkpoint logs/rsl_rl/pick_anything_so101_open_vocab/<run>/model_XXXX.pt
```

没有启动正式长训练或重启现有进程。新增模块仅由新入口导入；使用新的开放词汇
功能需要启动相应新入口。旧进程不会自动加载这些代码。

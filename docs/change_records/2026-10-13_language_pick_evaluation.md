# 语言掩码抓取与三帧 actor 多物体迁移评估

## 问题、证据与目标

用户确认 RGB＋掩码抓取已可用，要求继续接入语言 grounding，再测试已有多物体环境。
现有 GroundingDINO 已有离线权重和观察入口，但缺少真实语言闭环的本次验收。
多物体 CLI 默认单帧，完整加载原橘子 checkpoint 还会遇到 critic 49／54 维差异。
目标是固定现有 actor 做 GT／DINO 对照，保留来源和实际失败信息。

## 实际修改

- `scripts/open_vocab_pick_cli.py`：自动识别 checkpoint 历史帧数；显式帧数冲突拒绝；
  新增仅回放／评估的 `--actor-only`。
- `pick_anything/open_vocab_pick/checkpoints.py`：调用现有 RSL-RL 严格 actor 加载接口，
  包含归一化缓冲区，不加载 critic、optimizer 或迭代，不转换／覆盖源 checkpoint。
- `open_vocab_pick/config.py`：验证历史帧数；`mdp.py`、`env.py` 记录累计误抬物体和
  最大目标抬升高度，不修改奖励及成功标准。
- `scripts/evaluate_language_pick.py`：顺序执行单／多物体 GT／DINO 对照，保存独立日志、
  命令、hash、源码快照、图像和结果；超时／无结果视为失败，输出目录禁止复用。
- `scripts/test_open_vocab_pick.py`：真实三帧 actor 跨 critic 加载、动作严格一致、critic／
  optimizer／迭代不变、错误通道拒绝及统计复位检查。
- `OPEN_VOCAB_PICK.md`：新增当前三帧迁移评估用法，保留旧接口说明。

## checkpoint 与进程

固定权重：`models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt`。
SHA256：`a9e0008609d299bebedbe20e67ac480e5f9ab8f1616b4581d4ebd17f722bae9d`。
仅使用 Conda `isaaclab_61`，未安装依赖或修改工作区外代码；未暂存、提交或推送。
GPU 0 正忙，评估使用 GPU 2；不停止现有训练。新增代码在新评估进程生效。
本次不训练新权重，已有训练无需因本次评估重启。

## 已完成验证

- CPU open-vocabulary 7 项通过，包含真实 RSL-RL 模型和迁移数值检查。
- 既有 Grounded 10 项、三帧迁移 7 项 CPU 检查通过，共 24 项；修改的 7 个 Python 文件
  AST 解析通过。GPU 物理评估另记，CPU 检查不作为抓取成功证据。
- `evaluate_language_pick.py --help` 通过。
- CLI 实测拒绝训练使用 `--actor-only`、actor-only 缺少本地 checkpoint，以及显式单帧
  加载三帧权重；均在启动仿真之前以 exit=2 退出。
- 首轮真实 DINO 单橘子中文“抓橘子”：seed42、2 环境、HOME 首回合，成功 2/2、
  实际抬升 2/2，无超时／掉落。仅是小样本闭环验证，不是稳定成功率估计。
  结果：`outputs/language_pick/20261013_single_dino_seed42/summary.json`。

实际单橘子启动命令（另将 stdout／stderr 写入 `outputs/grounded_pick/language_dino_eval_20261013.log`）：

```bash
source /home/user123/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_61
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 \
ISAACSIM_ASSET_ROOT="$PWD/assets/isaac_cache" python scripts/eval_grounded_pick.py \
  --mask-backend dino --instruction 抓橘子 --num_envs 2 --device cuda:0 --seed 42 --viz none \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --diagnostics-dir outputs/language_pick/20261013_single_dino_seed42 \
  --dump-dir outputs/language_pick/20261013_single_dino_seed42/frames --dump-every 90 --dump-max 31
```

多物体对照已启动，结果待补充；尚不认定多目标抓取成功：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --scene multi --targets orange yellow_duck blue_cube red_cylinder \
  --num-envs 2 --seeds 42 --output-dir outputs/language_pick/20261013_multi_comparison
```

尚未验证：大样本语言成功率、新物体泛化、实机、像素轮廓分割。

## 对照过程中已确认的证据

- 单橘子 seed42、4 环境：GT 4/4，DINO 4/4；seed43 GT 3/4、1 超时。
  以上与初始中文两回合检查分开，不把重复种子的小样本直接合并为独立估计。
- 四物体场景抓橘子 seed42、2 环境：GT 0/2、DINO 0/2，均超时。
  两组每回合最大 rise 为约 0.004m（出生时落稳余量），没有实际抬升；误抬其他物体为 0。
- env0 的 GT／DINO 初始 RGB 文件逐字节一致，GT 橘子框与 DINO 最高分框 IoU=0。
  人工查看图像后，DINO 框位于红色圆柱附近；不将“返回框”当作选对目标。
  数据：`outputs/language_pick/initial_orange_detection_comparison_20261013.json`。
- 独立 CPU 离线检测同一张多物体初始图：`orange fruit.` 置信度约0.538但定位错误；
  `round orange fruit.` 约0.632、框覆盖橘子。这只是单张图的提示词诊断，未替换
  正在运行的对照配置，也不宣称闭环成功率提升。`blue cube.` 在目标未出现在该
  画面时同样返回了圆柱附近的框。结果：`outputs/language_pick/detection_probe_20261013.json`。
- GT 已失败说明多物体抓取迁移本身尚未通过；检测误选是另一个独立问题，不能用
  单橘子的语言成功率代替多目标选择能力验收。

单物体配对评估实际命令：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
python scripts/evaluate_language_pick.py \
  --checkpoint models/grounded_ppo/2026-10-13_temporal_transfer_v3/model_148.pt \
  --scene single --num-envs 4 --seeds 42 43 \
  --output-dir outputs/language_pick/20261013_single_comparison
```

## 最终完成结果

12/12 个配对条件全部完成。单橘子 seed43 DINO 和剩余三种物体的续跑命令及源码
变更见 `2026-10-13_skip_completed_episode_detection.md`；所有评估进程均已正常退出。
没有启动新的训练。源 checkpoint 的 SHA256 与开头记录一致。

| 场景／目标 | GT 成功／回合 | DINO 成功／回合 |
|---|---:|---:|
| 单橘子，seed42 | 4/4 | 4/4 |
| 单橘子，seed43 | 3/4 | 2/4 |
| 单橘子合计 | 7/8（87.5%） | 6/8（75%） |
| 四物体场景，橘子 | 0/2 | 0/2 |
| 四物体场景，小黄鸭 | 0/1 | 0/1 |
| 四物体场景，蓝色方块 | 0/1 | 0/1 |
| 四物体场景，红色圆柱 | 0/1 | 0/1 |

单物体继续按完整放盒撤爪验收，多物体按指定目标稳定抬升验收，不能将两者合并
为同一任务成功率。样本仅用于功能与迁移检查，未证明稳定 75% 或下降的统计显著性。
单橘子 DINO seed43 的失败分别为 env0 PREGRASP 超时2700步、env3 TRANSPORT
掉落298步；其余 env1／env2 成功。每个回合 mask 非空比例为100%，说明这两次
失败不能简单归因于空掩码。未完成的旧 DINO 尝试始终不计为成功或失败回合。

多物体主矩阵共 GT 5 回合、DINO 5 回合，均超时，最大 rise 约4mm（出生余量），
没有真正抬升，也没有误抬其他物体。旧 duck GT 0/2、初始中文 DINO 2/2 分别作为
补充观察保存，不重复计入主矩阵。共保存30个已完成GPU回合的结果和相应日志。

6 个配对条件的 env0 初始 RGB 文件均逐字节一致。单橘子初始框 IoU 为0.946／
0.942；多物体橘子0、小黄鸭0.072、蓝方块0、红圆柱0.613。这些仅是初始图的
投影矩形对照，不是整体定位准确率；鸭子检测框主要覆盖可见部分，与较完整 GT
几何框有差异。多物体仍需目标选择及抓取训练，grounding 误选需独立改进和验证。

最终报告：`outputs/language_pick/REPORT_20261013.md`。
机器可读汇总：`outputs/language_pick/SUMMARY_20261013.json`，含每组的来源路径。
重建汇总实际执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python outputs/language_pick/summarize_results_20261013.py
```

上述汇总脚本只读取已完成条件，并断言不重复计入同一条件；不包含中断尝试。
本次最终核对12组结果完整、权重hash不变、所有多物体回合最大rise未达抬升门槛。
24项CPU检查通过，`git diff --check`通过。未验证项仍为大样本语言性能、成功的多目标
泛化、实机和像素轮廓分割；当前多物体迁移实际验证未通过。

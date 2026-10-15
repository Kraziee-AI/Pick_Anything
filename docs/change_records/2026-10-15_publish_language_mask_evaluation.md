# 提交多物体语言掩码任务及评估入口

## 问题、证据与目标

用户要求将掩码抓取提交到 Kraziee-AI。单橘子的三帧策略已有独立权重与评估，
但用于真实语言对照的多物体场景、资产、actor-only 加载和批量评估脚本仍未跟踪。
只发布单橘子入口不能复现最新报告中的多物体对照。

## 本次整理范围

- 提交 `pick_anything/open_vocab_pick/`、`pick_anything/tabletop/`，以及蓝方块、
  红圆柱、小黄鸭的 USD 和相对路径贴图。
- 提交独立多物体训练／回放／评估、场景查看／生成和 CPU 检查脚本；
  `scripts/evaluate_language_pick.py` 保存命令、权重哈希、源码快照、逐回合结果和采样帧。
- 提交现有开放词汇、语言对照、跳过已完成环境检测的历史修改记录。
- OPEN_VOCAB_PICK.md 新增 model_2599 的归档路径、最新结果和用现有通用脚本
  进行 4/4/2 回合复评的命令；旧结果保留日期和适用范围。
  TABLETOP_OBJECTS.md 和 RGB_MASK_PPO.md 增加任务导航。
- 依赖前一单橘子提交提供的 Grounded 公共代码和权重；不提交其他诊断实验、
  摄像头任务注册、蒸馏、MuJoCo 或缓存／原始训练日志。

## 权重、评估与进程

最新 checkpoint：
`models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`。
SHA256：`f4414a3811906a33b3eba0436140173d076dce08d601eeb87bebdb2bdc8a4386`。
真实 DINO 历史结果：单橘子 8/10、多物体抓橘子 0/10；两任务成功标准不同。
逐回合报告／JSON 已随前一提交保存到 `docs/evaluations/2026-10-14_model2599/`。
本次不训练、不更新 checkpoint，也不需要重启已有进程；复评命令见 OPEN_VOCAB_PICK.md。

## 验证

已从暂存区导出到 `outputs/publish_validation/language/`，在隔离副本中使用 Conda isaaclab_61：

- OpenVocab 7 项、Tabletop 4 项 CPU 检查均通过，包含实际 PPO 更新、actor-only 加载
  保留归一化／不改 critic 和优化器、检测器不读场景真值及四物体布局／资产检查。
- 场景 `--inspect --instruction "抓小黄鸭"`、批量评估入口 `--help` 通过。
- 19 个待提交 Python 文件均通过无字节码编译检查；没有安装依赖或修改环境。
- 归档最新权重的 SHA256 与历史 SUMMARY 一致，两场景各 10 个唯一首回合，
  从逐回合结果重算单橘子 8、多物体 0，与原汇总一致。
- 三个生成的 USDA 文件仅清理末尾多余空行，没有改变任何资产属性；
  隔离资产检查使用清理后的文件。日志在隔离目录 `outputs/validation_logs/`。
- 暂存区范围已核对，排除了其他本地实验。推送后的远端状态另外通过 Git 核对。

本次整理未重新进行 GPU 物理、DINO 闭环或新的成功率测试，不将历史成绩算作本次新实测。

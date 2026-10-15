# 归档并提交单橘子 RGB＋mask 抓取路线

## 问题、证据与目标

用户要求将已验证的掩码抓取提交到 Kraziee-AI 的 GitHub 仓库。
核对时远端 master 为 `286d08a`；Grounded 包、入口、文档和候选权重均未被 Git 跟踪。
最新 model_2599 位于被忽略的 logs 中，仅推送代码无法直接加载这个已验证版本。
目标是保存完整的训练、语言回放和 HOME 首回合评估入口及必要依赖。

## 本次整理范围

- 提交 `pick_anything/grounded_pick/`、对应训练／回放／评估／初始化／诊断脚本及已有 CPU 检查。
- 提交 `stages.py` 中被 Grounded 使用的连续确认基类和按环境更新能力，以及
  `camera_progress.py` 中的持续夹持交接、同一步去重实现；不包含摄像头任务注册、
  教师 DR、蒸馏、MuJoCo 或其他已有工作区改动。
- 保存初始化实际依赖的 RGB model_17596、早期候选 model_148 和最新 model_2599。
  最新权重复制到 `models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/`，
  同目录保存原 env／agent 配置和新增 manifest。原 logs 文件保留。
- 复制历史 20 回合 REPORT.md／SUMMARY.json 到 `docs/evaluations/2026-10-14_model2599/`，
  保留其原始内容和原始本地证据路径；RGB_MASK_PPO.md 补充仓库内权重和报告链接。
- README 仅提交新增的掩码路线导航；其余尚未提交的摄像头内容保留在工作区。
- 保留该路线此前的独立修改记录，不改写历史结论。

## 权重、命令与结果范围

归档最新权重 SHA256：
`f4414a3811906a33b3eba0436140173d076dce08d601eeb87bebdb2bdc8a4386`。
历史真实 DINO 评估：单橘子完整任务 8/10，抓起并抬升 10/10；多物体抓橘子 0/10。
这两项成功标准不同，不能据此声称多目标泛化或稳定总体成功率。

```bash
conda activate isaaclab_61
export ISAACSIM_ASSET_ROOT="$PWD/assets/isaac_cache"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python scripts/eval_grounded_pick.py \
  --mask-backend dino --instruction "orange" --image-history 3 \
  --num_envs 4 --device cuda:0 --seed 42 --viz none \
  --checkpoint models/grounded_ppo/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt
```

上面是归档副本的复评命令，本次发布整理未重新执行 GPU 评估或训练。
已有训练／回放进程无需为此次文件归档重启；新启动进程读取所指定的 checkpoint。
完整日志、检测器缓存和采样帧继续保留在本地被忽略的 outputs 中。

## 本次验证

- 已复制最新权重并断言其 SHA256 与历史评估完全一致。
- 已从暂存区导出到 `outputs/publish_validation/grounded/`，使用 Conda isaaclab_61
  在隔离目录运行已有测试，避免本地未提交依赖掩盖问题。
- `test_grounded_pick.py` 10 项、`test_grounded_recovery.py` 7 项、
  `test_grounded_transfer.py` 7 项均通过，共 24 项；包括真实 actor 输出迁移等价、
  实际 PPO 更新、新增 mask 通道可学习、首回合终止计数和逐环境确认去重。
- `bash -n scripts/train_rgb_mask_ppo.sh` 通过。日志保存在隔离目录 `outputs/validation_logs/`。
- 暂存区清单和 README 差异已检查，仅包含上述路线及必要依赖。
  空行格式检查发现 `progress.py` 文件末尾多余空行，已清理，不改变运行逻辑。
- 本记录仅记本地整理与验证；远端提交结果由完成推送后的 Git 状态核对。

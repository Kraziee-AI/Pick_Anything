# model_2599 单橘子与多物体各10回合实测

固定最新RGB+mask三帧actor，物理GPU 0。语言目标均为orange；catalog送入真实
GroundingDINO的文本为`orange fruit.`，检测框生成二值矩形掩码，再与RGB和本体状态输入策略。

Checkpoint：`/home/user123/IsaacLab_6.1/Pick_Anything/logs/rsl_rl/pick_orange_so101_grounded/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`。
SHA256：`f4414a3811906a33b3eba0436140173d076dce08d601eeb87bebdb2bdc8a4386`。

| 场景 | 成功/回合 | 实测成功率 | 既有成功判定 |
|---|---:|---:|---|
| 单橘子 | 8/10 | 80% | HOME开始，抓取、放盒与回撤全部完成 |
| 多物体，目标橘子 | 0/10 | 0% | 正确目标夹持并抬高8cm，连续保持5步 |

两套任务成功口径和reset分布不同，以上分别报告原环境指标。多物体其他物体作为
干扰物存在，本轮没有轮换语言目标，因此结果不能代表其他目标的成功率。

seed42/43/44分别4/4/2个环境；每个环境只计首次HOME回合。程序异常不计为失败，
无自动重置后回合混入，无GT替代DINO，无修改权重、阈值、场景reset或成功条件。

## 每回合结果

| 场景 | seed | env_id | 成功 | 终止原因 |
|---|---:|---:|---|---|
| single | 42 | 0 | 是 | success |
| single | 42 | 1 | 否 | staged_orange_dropped |
| single | 42 | 2 | 是 | success |
| single | 42 | 3 | 是 | success |
| single | 43 | 0 | 是 | success |
| single | 43 | 1 | 是 | success |
| single | 43 | 2 | 是 | success |
| single | 43 | 3 | 否 | time_out |
| single | 44 | 0 | 是 | success |
| single | 44 | 1 | 是 | success |
| multi | 42 | 0 | 否 | time_out |
| multi | 42 | 1 | 否 | time_out |
| multi | 42 | 2 | 否 | time_out |
| multi | 42 | 3 | 否 | time_out |
| multi | 43 | 0 | 否 | time_out |
| multi | 43 | 1 | 否 | time_out |
| multi | 43 | 2 | 否 | time_out |
| multi | 43 | 3 | 否 | time_out |
| multi | 44 | 0 | 否 | time_out |
| multi | 44 | 1 | 否 | time_out |

## 抓取与终止诊断

单橘子grasped 10/10，lifted 10/10。
单橘子终止原因：`{"success": 8, "staged_orange_dropped": 1, "time_out": 1}`。
多物体正确橘子grasped 0/10；出现错误物体抬升 1/10。
多物体终止原因：`{"time_out": 10}`。
错误物体抬升为至少一个非目标物体中心升高超过4cm的物理标记，不等于确认夹持了该物体。
单橘子抓取／抬升里程碑与最终放盒成功分别统计，不能把前者当作完整任务成功。

## 可复现命令与证据

在Pick_Anything目录：

```bash
/home/user123/miniconda3/envs/isaaclab_61/bin/python outputs/language_pick/20261014_model2599_eval10/run_evaluation.py single
/home/user123/miniconda3/envs/isaaclab_61/bin/python outputs/language_pick/20261014_model2599_eval10/run_evaluation.py multi
```

已有输出目录不可覆盖；复跑应创建新的实验目录并复制启动脚本。
各批次`run.json`保存实际子进程命令、checkpoint与源码哈希，`source/`保存源快照；
每组`orange_dino_seed*/`内有run.log、result.json及采样RGB、mask、metadata。
单橘子额外保存summary.json逐回合诊断和trace.jsonl；完整汇总见同目录SUMMARY.json。

本轮只有各10回合，这是当前测试样本结果，不等同于已证明的稳定总体成功率。

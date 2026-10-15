# 最新RGB+mask权重：单橘子和多物体各10回合评估

## 问题、证据与目标

用户要求分别实测单橘子和多物体成功率，各10次。已有6/8与0/5结果来自早期
model_148，不能代表最新model_2599。此次固定最新权重、真实GroundingDINO和物理
GPU 0，记录10个从HOME出生的完整首回合，不计自动重置后的第二回合。

## 本次完成的准备与范围

- 新增独立实验启动脚本
  `outputs/language_pick/20261014_model2599_eval10/run_evaluation.py`。
- 新增同目录`summarize_evaluation.py`，仅在两种场景全部完成后核对每批计数、
  不重复的seed/env_id、权重哈希、源码一致性和终止原因，生成SUMMARY.json与REPORT.md。
- 两种场景按seed42/43/44分别4/4/2个环境，总计各10回合。语言目标orange，实际
  送入检测器的catalog文本为`orange fruit.`。多物体默认保持相同目标用于比较。
- 调用已有evaluate_language_pick.py，真实DINO矩形掩码进入RGB+mask三帧策略。
  多物体使用actor-only加载，保留actor与归一化参数。
- 每次运行保存权重SHA、源码快照、实际命令、日志、采样帧和评估结果；单橘子
  同时保存每回合诊断。没有修改策略、场景reset、成功条件或检测阈值。
- 使用小批次避免此前32环境原生崩溃问题。程序异常不能算作抓取失败。

## Checkpoint与执行命令

`logs/rsl_rl/pick_orange_so101_grounded/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`

SHA256：`f4414a3811906a33b3eba0436140173d076dce08d601eeb87bebdb2bdc8a4386`。

在Pick_Anything中使用Conda isaaclab_61对应的解释器：

```bash
/home/user123/miniconda3/envs/isaaclab_61/bin/python outputs/language_pick/20261014_model2599_eval10/run_evaluation.py single
/home/user123/miniconda3/envs/isaaclab_61/bin/python outputs/language_pick/20261014_model2599_eval10/run_evaluation.py multi
```

脚本固定CUDA_VISIBLE_DEVICES=0。内部完整评估命令保存至各批次run.json。
无需重启训练，评估单独启动新进程，不能替换旧UI进程正在加载的权重。

## 验证方法、实际结果与限制

准备阶段已核对权重和输入接口。闭环评估尚待完成，未填写成功率。
两种环境既有成功口径不同：单橘子要求抓取、放盒和回撤完成；多物体要求正确
目标保持夹持并抬升8cm持续5步。因此分别报告，不把两种指标直接视作同一任务。

## 运行中的已完成结果

单橘子seed42/43/44三批正常退出，分别3/4、3/4、2/2，累计8/10=80%。
10/10有grasped与lifted里程碑；失败包括1次staged_orange_dropped、1次time_out。
超时回合的诊断显示橘子已经离爪且停在盒外，按90秒原上限结束。
多物体seed42首批正常退出，0/4，4次time_out，1/4记录wrong_object_lifted。
剩余多物体6回合仍在运行，不能把0/4作为最终10回合结果。

## 最终完成与核验

上述“运行中”段落为当时的阶段记录；现已完成全部20个首回合。

| 场景 | seed42 | seed43 | seed44 | 总成功率 |
|---|---:|---:|---:|---:|
| 单橘子 | 3/4 | 3/4 | 2/2 | 8/10，80% |
| 多物体抓橘子 | 0/4 | 0/4 | 0/2 | 0/10，0% |

- 全部六个评估子进程正常退出；没有把程序异常计为抓取失败。
- 单橘子10/10确认抓取并抬升，1次掉落、1次盒外放置后90秒超时，其余8次
  达到原环境物理放盒、撤爪及静置成功门。
- 多物体10/10超时、0/10确认夹持正确橘子；1/10出现非目标物体升高超过4cm。
  橘子最大升高为约1.01cm，因此本轮失败不是仅差8cm成功门。
- 汇总脚本已实际执行并通过断言：两组各10、每组seed/env_id唯一、所有批次
  权重SHA符合预期、固定GPU 0、三帧输入、DINO、HOME首回合；批次源码哈希
  一致，逐回合成功计数与环境打印结果一致。结束后权重SHA仍相同。
- 输出`outputs/language_pick/20261014_model2599_eval10/SUMMARY.json`与同目录
  `REPORT.md`；原始日志、帧、源码与诊断均保留。
- 更新`RGB_MASK_PPO.md`最新权重结果和旧核对记录的后续评估链接，保留旧
  model_148历史数据。没有更改训练/任务代码，没有暂存、提交或推送。

汇总实际命令：

```bash
/home/user123/miniconda3/envs/isaaclab_61/bin/python -B outputs/language_pick/20261014_model2599_eval10/summarize_evaluation.py
```

只评估语言orange，没有执行最新模型的GT消融、轮换其他目标或重新训练。
各10回合是本轮小样本实测，尚未验证稳定总体成功率及最新模型的失败原因消融。
单橘子完整任务与多物体抬升任务口径差异仍然存在，不能将两项直接当作同一任务对照。

# 核对最新掩码checkpoint并纠正回放权重说明

## 问题与证据

用户指出给出的UI回放命令使用 `model_148.pt`，它不是后续掩码训练的最后保存权重。
此前语言闭环、随机多物体和控制诊断全部明确固定为早期候选model_148；但回复中
没有核对后续训练目录，容易让人把旧候选结果误解为最新模型能力。

## 实际完成的核对与修正

- 按保存时间检查 `logs/rsl_rl/pick_orange_so101_grounded/` 和 `models/grounded_ppo/`。
- 当前最新保存文件：
  `logs/rsl_rl/pick_orange_so101_grounded/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt`。
  保存时间2026-10-13 16:41:40，本地文件17MB；已用Conda isaaclab_61实际读取。
- checkpoint内部iter=2599；actor卷积第一层(16,12,5,5)，即3帧×4通道；本体归一化
  (1,24)，动作输出(6,128)，critic归一化(1,49)；actor全部张量有限。
- SHA256：`f4414a3811906a33b3eba0436140173d076dce08d601eeb87bebdb2bdc8a4386`。
- 该训练run配置记录从同路线model_600加载，追加2000轮；image_history=3、夹持
  确认距离0.035m、最小爪角0rad，与当前Grounded入口接口一致。
- `RGB_MASK_PPO.md` 增加最新保存文件及其回放命令，并明确旧model_148结果的范围。
  没有修改训练超参数、现有权重或旧实验记录，没有暂存、提交或推送。

## 验证与限制

完成文件时间／目录核对、真实checkpoint读取、网络维度和有限数检查。
model_2599的真实GroundingDINO UI、独立HOME成功率和多物体迁移尚未在本次核对中
执行；文件较新不证明更高成功率。之前DINO6/8、多物体GT及DINO各0/5和36回合
原因诊断都只适用于model_148，不能直接外推至model_2599。

不需要重启训练；用户要查看新权重时，需停止旧UI回放后用新checkpoint启动。
已提供的纠正命令（尚未由本次核对执行）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/play_grounded_pick.py \
  --instruction "orange" --mask-backend dino --image-history 3 \
  --num_envs 1 --device cuda:0 --seed 42 --viz kit --real-time \
  --checkpoint "$PWD/logs/rsl_rl/pick_orange_so101_grounded/2026-10-13_14-03-59_rgb_mask_temporal_transfer_v3/model_2599.pt"
```

使用前仍须激活Conda isaaclab_61、进入Pick_Anything并配置本地Isaac资产和Kit的
LD_PRELOAD；沿用前面UI命令中的环境设置即可。

## 后续独立评估（2026-10-14）

核对阶段之后，用户明确要求单橘子、多物体分别测试10次。该评估已用同一
model_2599和SHA在GPU 0实际完成，真实DINO、目标orange：单橘子8/10=80%，
多物体0/10=0%。旧model_148结果继续单独保留。
详细命令、成功口径差异、逐回合结果和限制见
[评估记录](2026-10-14_model2599_ten_episode_evaluation.md)与
[实验报告](../../outputs/language_pick/20261014_model2599_eval10/REPORT.md)。

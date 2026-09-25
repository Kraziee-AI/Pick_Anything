"""home-only 完整任务评估：强制全部环境从 HOME 出生，精确计 N 个 episode。

用法（Pick_Anything 根目录，conda env isaaclab_61）：

    python scripts/eval_pick_home.py --rl_library rsl_rl \
        --task Pick-Apple-SO101-Staged-v0 --num_envs 256 --device cuda:1 \
        --checkpoint logs/rsl_rl/pick_apple_staged_so101/<run>/model_XXXX.pt \
        --external_callback scripts.eval_pick_home:install_episode_counter \
        --viz none

机制（移植自参考项目 evaluation.py 的 wrapper patch 模式）：
* 任务的 play_mode 已把 reset 的 forced_stage 钉在 0（HOME）——所有 episode
  从桌面完整任务开始，不使用任何中间阶段出生；
* patch RslRlVecEnvWrapper.step：每个 env 只计第一个 episode，读取终止原
  因分类（success / timeout / dropped / box / abnormal）与阶段到达统计，
  收满 N 个 episode 后打印 ``PICK_APPLE_EVAL_RESULT=<json>`` 并退出。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

EVALUATION_EPISODES = 256

_TERMINATION_KEYS = ("success", "time_out", "staged_orange_dropped", "staged_box_tipped", "staged_abnormal_robot")


def install_episode_counter() -> list[str]:
    """patch play 循环：精确计 N 个 home 出生 episode 并打印汇总。"""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    target = EVALUATION_EPISODES
    counts = dict.fromkeys(
        ("episodes", "successes", "grasp", "lift", "over_box", "in_box") + _TERMINATION_KEYS, 0
    )
    counted: torch.Tensor | None = None

    original_step = RslRlVecEnvWrapper.step

    def counted_step(self, actions):
        nonlocal counted
        if self.num_envs != target:
            raise RuntimeError(f"home-only 评估要求 --num_envs {target}（每环境恰好一局）")
        result = original_step(self, actions)
        dones = result[2].bool()
        if counted is None:
            counted = torch.zeros_like(dones)
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        done_ids = done_ids[~counted[done_ids]]
        if done_ids.numel() == 0:
            return result
        counted[done_ids] = True
        env = self.unwrapped

        counts["episodes"] += int(done_ids.numel())
        for key in _TERMINATION_KEYS:
            try:
                term = env.termination_manager.get_term(key)[done_ids]
            except Exception:
                term = torch.zeros(int(done_ids.numel()), dtype=torch.bool, device=done_ids.device)
            counts[key] += int(term.sum())
        counts["successes"] = counts["success"]
        state = getattr(env, "staged_pick_state", None)
        if state is not None:
            counts["grasp"] += int(state.progress.grasped[done_ids].sum())
            counts["lift"] += int(state.progress.lifted[done_ids].sum())
            counts["over_box"] += int(state.progress.over_box_latched[done_ids].sum())
            counts["in_box"] += int(state.progress.in_box_latched[done_ids].sum())

        if counts["episodes"] >= target:
            episodes = counts["episodes"]
            summary = {
                "episodes": episodes,
                "success_rate": counts["successes"] / episodes,
                "grasp_rate": counts["grasp"] / episodes,
                "lift_rate": counts["lift"] / episodes,
                "over_box_rate": counts["over_box"] / episodes,
                "in_box_rate": counts["in_box"] / episodes,
                "timeout_rate": counts["time_out"] / episodes,
                "orange_dropped_rate": counts["staged_orange_dropped"] / episodes,
                "box_tipped_rate": counts["staged_box_tipped"] / episodes,
                "abnormal_rate": counts["staged_abnormal_robot"] / episodes,
                "birth_policy": "forced HOME (complete task)",
            }
            print("PICK_APPLE_EVAL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = counted_step
    return sys.argv[1:]


if __name__ == "__main__":
    import pick_anything.tasks  # noqa: F401,E402  -- 注册 gym 任务

    from isaaclab_rl.entrypoints.dispatch import run_play_cli  # noqa: E402

    raise SystemExit(run_play_cli())

"""CLI for the independent multi-object PPO task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from grounded_pick_cli import configure_caches, prelaunch_kit


def run(mode):
    configure_caches()
    from pick_anything.open_vocab_pick import TASK_ID, register_task
    from pick_anything.grounded_pick.grounding import DEFAULT_CACHE, DEFAULT_MODEL, workspace_output
    from pick_anything.grounded_pick.language import goal_caption
    from pick_anything.tabletop.catalog import OBJECTS, resolve_target

    parser = argparse.ArgumentParser(add_help=False, description="开放词汇目标→矩形掩码→PPO 抓取／抬升")
    parser.add_argument("--instruction", default="抓橘子", help="单目标描述；支持自由英文与常用中文目标")
    parser.add_argument("--caption", default="", help="显式英文检测词，例如 'yellow rubber duck on the left'")
    parser.add_argument("--target-key", choices=["random", *OBJECTS], default=None,
                        help="训练／评估真值目标；GT 训练默认 random，其余模式从文本解析")
    parser.add_argument("--mask-backend", choices=["gt", "dino"], default="gt" if mode == "train" else "dino")
    parser.add_argument("--grounding-model", default=DEFAULT_MODEL)
    parser.add_argument("--grounding-device", default="")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--dump-every", type=int, default=30)
    parser.add_argument("--dump-max", type=int, default=100)
    parser.add_argument("--image-history", type=int, choices=[1, 3], default=None)
    parser.add_argument("--actor-only", action="store_true",
                        help="仅回放／评估：严格加载 actor（含归一化），跳过旧任务 critic 和优化器")
    args, remaining = parser.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        print(parser.format_help())
    try:
        if args.actor_only and mode == "train":
            raise ValueError("--actor-only 仅用于回放／评估，不是完整训练恢复。")
        checkpoint_parser = argparse.ArgumentParser(add_help=False)
        checkpoint_parser.add_argument("--checkpoint", default="")
        checkpoint = checkpoint_parser.parse_known_args(remaining)[0].checkpoint
        history = args.image_history
        if args.actor_only and not Path(checkpoint).expanduser().is_file():
            raise ValueError("--actor-only 必须指定本地 --checkpoint 文件。")
        if checkpoint and Path(checkpoint).expanduser().is_file():
            from pick_anything.grounded_pick.checkpoints import image_history_from_checkpoint
            expected = image_history_from_checkpoint(Path(checkpoint).expanduser())
            if history is not None and history != expected:
                raise ValueError(f"checkpoint 需要 {expected} 帧，与 --image-history 不一致。")
            history = expected
        history = history or 1
        caption = goal_caption(args.instruction, args.caption)
        key = args.target_key
        if key is None:
            key = "random" if mode == "train" and args.mask_backend == "gt" else resolve_target(caption, tuple(OBJECTS)).key
        if key == "random" and (mode != "train" or args.mask_backend != "gt"):
            raise ValueError("随机目标只适用于 GT 训练；回放和评估需要固定目标。")
        if args.dump_dir:
            args.dump_dir = str(workspace_output(args.dump_dir))
        if args.dump_every < 1 or args.dump_max < 0:
            raise ValueError("dump_every 必须为正，dump_max 必须非负。")
        if not (0 < args.box_threshold < 1 and 0 < args.text_threshold < 1):
            raise ValueError("检测阈值必须在 (0,1) 内。")
    except ValueError as exc:
        parser.error(str(exc))
    print(f"[OPEN-VOCAB] target_key={key}, detector_caption={caption}, backend={args.mask_backend}", flush=True)
    values = {
        "instruction": args.instruction, "backend": args.mask_backend,
        "model_name": args.grounding_model, "grounding_device": args.grounding_device,
        "cache_dir": str(DEFAULT_CACHE), "local_files_only": True,
        "box_threshold": args.box_threshold, "text_threshold": args.text_threshold,
        "dump_dir": args.dump_dir, "dump_every": args.dump_every, "dump_max": args.dump_max,
        "image_history": history,
    }
    overrides = [f"env.observations.rgb_mask.image.params.{name}={json.dumps(value, ensure_ascii=False)}"
                 for name, value in values.items()]
    overrides += [f"env.target_key={json.dumps(key)}", f"env.grounding_caption={json.dumps(caption)}"]
    if mode == "eval":
        overrides.append("env.evaluate_first_episode=true")
    if any(a == "--train_env_cfg" or a.startswith("--train_env_cfg=") for a in remaining):
        parser.error("开放词汇任务统一从 HOME 重置，不使用旧任务的中间阶段配置。")
    if any(a == "--task" or a.startswith("--task=") for a in remaining):
        parser.error("此入口固定使用开放词汇任务；旧橘子任务请用原入口。")
    remaining += ["--task", TASK_ID]
    if not any(a == "--rl_library" or a.startswith("--rl_library=") for a in remaining):
        remaining += ["--rl_library", "rsl_rl"]
    if mode != "train" and not any(a == "--num_envs" or a.startswith("--num_envs=") for a in remaining):
        remaining += ["--num_envs", "1"]
    prelaunch_kit(remaining)
    if mode != "train":
        # Avoid exporting over files in a source checkpoint directory.
        from rsl_rl.runners import OnPolicyRunner
        OnPolicyRunner.export_policy_to_jit = lambda *args, **kwargs: None
        OnPolicyRunner.export_policy_to_onnx = lambda *args, **kwargs: None
        if args.actor_only:
            from pick_anything.open_vocab_pick.checkpoints import load_evaluation_actor
            OnPolicyRunner.load = load_evaluation_actor
    register_task()
    sys.argv = [sys.argv[0], *remaining, *overrides]
    from isaaclab_rl.entrypoints.dispatch import run_train_cli, run_play_cli
    return (run_train_cli if mode == "train" else run_play_cli)()

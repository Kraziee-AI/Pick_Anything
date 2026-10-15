"""Shared launcher for the new task only; no changes to existing entry points."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.dont_write_bytecode = True


def configure_caches() -> None:
    # Keep downloads, compilation caches and bytecode out of sibling projects.
    cache = PROJECT_ROOT / "outputs/grounded_pick/cache"
    for key, subdir in {
        "XDG_CACHE_HOME": "xdg", "WARP_CACHE_PATH": "warp", "HF_HOME": "huggingface",
        "HF_HUB_CACHE": "huggingface", "TORCH_HOME": "torch", "TORCH_EXTENSIONS_DIR": "torch_extensions",
        "MPLCONFIGDIR": "matplotlib", "CUDA_CACHE_PATH": "cuda",
    }.items():
        os.environ[key] = str(cache / subdir)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # HTTP download uses the configured proxy; Xet's separate transport may not.
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def prelaunch_kit(argv):
    if "-h" in argv or "--help" in argv:
        return
    if "--viz=kit" in argv or any(a == "--viz" and b == "kit" for a, b in zip(argv, argv[1:])):
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args(argv)
        AppLauncher(args)


def run(mode: str) -> int:
    configure_caches()
    from pick_anything.grounded_pick import TASK_ID, register_task
    from pick_anything.grounded_pick.grounding import DEFAULT_CACHE, DEFAULT_MODEL, workspace_output
    from pick_anything.grounded_pick.masks import orange_caption

    parser = argparse.ArgumentParser(add_help=False, description="新增橘子语言／掩码抓取参数")
    parser.add_argument("--instruction", default="抓橘子", help="只支持橘子；中英文均可")
    parser.add_argument("--mask-backend", choices=["gt", "dino"], default="gt" if mode == "train" else "dino")
    parser.add_argument("--grounding-model", default=DEFAULT_MODEL, help="HF 模型 ID 或本地 Transformers 模型目录")
    parser.add_argument("--grounding-device", default="", help="默认与仿真 device 相同，可单独设 cpu/cuda:N")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--dump-dir", default="", help="保存 env_0 RGB／mask／检测元数据，路径须位于本仓库")
    parser.add_argument("--dump-every", type=int, default=30)
    parser.add_argument("--dump-max", type=int, default=100)
    parser.add_argument("--diagnostics-dir", default="", help="HOME 评估逐步诊断与每回合统计（仅 eval）")
    parser.add_argument("--trace-every", type=int, default=5, help="轨迹记录间隔；事件帧仍逐次记录")
    parser.add_argument("--image-history", type=int, choices=[1, 3], default=None,
                        help="RGB/mask 历史帧数；文件 checkpoint 自动识别，旧模型为1帧")
    parser.add_argument("--init-from-rgb", default="", help="仅训练：从兼容 RGB 抓取权重初始化三帧 mask PPO")
    parser.add_argument("--training-recipe", choices=["legacy", "transfer"], default=None,
                        help="transfer: lr=2e-5、clip=0.1、entropy=1e-4；legacy 保留原配置")
    args, remaining = parser.parse_known_args()
    if "-h" in remaining or "--help" in remaining:
        print(parser.format_help())
    orange_caption(args.instruction)
    checkpoint_parser = argparse.ArgumentParser(add_help=False)
    checkpoint_parser.add_argument("--checkpoint", default="")
    checkpoint_args, _ = checkpoint_parser.parse_known_args(remaining)
    if args.init_from_rgb and (mode != "train" or checkpoint_args.checkpoint):
        parser.error("--init-from-rgb 仅用于训练初始化，不能同时指定 --checkpoint。")
    if args.training_recipe is not None and mode != "train":
        parser.error("--training-recipe 仅用于训练。")
    recipe = args.training_recipe or ("transfer" if args.init_from_rgb else "legacy")
    history = args.image_history
    if args.init_from_rgb:
        from pick_anything.grounded_pick.checkpoints import initialize_from_rgb

        history = history or 3
        learning_rate = 2e-5 if recipe == "transfer" else 1e-4
        initialized = initialize_from_rgb(args.init_from_rgb, "outputs/grounded_pick/initializations",
                                          history, learning_rate)
        remaining.extend(["--checkpoint", str(initialized)])
        print(f"[GROUNDED] RGB initialization: {initialized}; image_history={history}", flush=True)
    elif checkpoint_args.checkpoint and Path(checkpoint_args.checkpoint).expanduser().is_file():
        from pick_anything.grounded_pick.checkpoints import image_history_from_checkpoint

        expected = image_history_from_checkpoint(Path(checkpoint_args.checkpoint).expanduser())
        if history is not None and history != expected:
            parser.error(f"checkpoint 需要 {expected} 帧 RGB/mask，与 --image-history 不一致。")
        history = expected
    history = history or 1
    if args.dump_dir:
        args.dump_dir = str(workspace_output(args.dump_dir))
    if args.diagnostics_dir:
        if mode != "eval":
            parser.error("--diagnostics-dir 仅用于 eval_grounded_pick.py，不改变训练。")
        args.diagnostics_dir = str(workspace_output(args.diagnostics_dir))
    if args.trace_every < 1:
        parser.error("--trace-every 必须为正数。")
    # Values enter Hydra as quoted strings, never as shell command fragments.
    values = {
        "backend": args.mask_backend, "instruction": args.instruction,
        "model_name": args.grounding_model, "grounding_device": args.grounding_device,
        "cache_dir": str(DEFAULT_CACHE), "local_files_only": True,
        "box_threshold": args.box_threshold, "text_threshold": args.text_threshold,
        "dump_dir": args.dump_dir, "dump_every": args.dump_every, "dump_max": args.dump_max,
        "image_history": history,
    }
    overrides = [f"env.observations.rgb_mask.image.params.{key}={json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]
    if mode == "train" and recipe == "transfer":
        overrides.extend(["agent.algorithm.learning_rate=0.00002", "agent.algorithm.clip_param=0.1",
                          "agent.algorithm.entropy_coef=0.0001", 'agent.algorithm.schedule="fixed"'])
        # RSL-RL restores the optimizer LR from checkpoints. Reapply this explicit
        # recipe AFTER load so a legacy optimizer cannot silently override it.
        from rsl_rl.runners import OnPolicyRunner

        original_load = OnPolicyRunner.load

        def transfer_load(runner, *positional, **keywords):
            result = original_load(runner, *positional, **keywords)
            runner.alg.learning_rate = 2e-5
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = 2e-5
            return result

        OnPolicyRunner.load = transfer_load
    if mode == "eval":
        if "--train_env_cfg" in remaining:
            parser.error("完整任务评估不能使用 --train_env_cfg。")
        overrides.append("env.evaluate_first_episode=true")
        overrides.append(f"env.evaluation_diagnostics_dir={json.dumps(args.diagnostics_dir)}")
        overrides.append(f"env.evaluation_trace_every={args.trace_every}")
    if not any(a == "--task" or a.startswith("--task=") for a in remaining):
        remaining.extend(["--task", TASK_ID])
    if not any(a == "--rl_library" or a.startswith("--rl_library=") for a in remaining):
        remaining.extend(["--rl_library", "rsl_rl"])
    if mode != "train" and not any(a == "--num_envs" or a.startswith("--num_envs=") for a in remaining):
        remaining.extend(["--num_envs", "1"])
    prelaunch_kit(remaining)
    if mode == "eval" and args.diagnostics_dir:
        # The ordinary player exports into the checkpoint run. Measurements
        # should preserve those files, especially when testing seeds in parallel.
        from rsl_rl.runners import OnPolicyRunner

        OnPolicyRunner.export_policy_to_jit = lambda *args, **kwargs: None
        OnPolicyRunner.export_policy_to_onnx = lambda *args, **kwargs: None
    register_task()
    sys.argv = [sys.argv[0], *remaining, *overrides]
    from isaaclab_rl.entrypoints.dispatch import run_play_cli, run_train_cli

    return (run_train_cli if mode == "train" else run_play_cli)()

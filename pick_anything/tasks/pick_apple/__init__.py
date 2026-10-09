"""Gym registration for the apple-picking-into-box task."""

import gymnasium as gym

# Isolated pure-PPO smoothing tasks; historical Camera-v8 remains reproducible.
for _task_suffix, _env_class in (
    ("Smooth", "CameraSmoothEnvCfg"),
    ("Filtered", "CameraFilteredEnvCfg"),
):
    gym.register(
        id=f"Pick-Apple-SO101-Staged-Camera-{_task_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.smooth_camera_cfg:{_env_class}",
            "rsl_rl_cfg_entry_point": f"{__name__}.smooth_camera_cfg:CameraSmoothPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )

from . import agents
from .pick_env_cfg import PickAppleEnvCfg

gym.register(
    id="Pick-Apple-SO101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_env_cfg:PickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickApplePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 八阶段版（物理验证 reset 状态库 + 课程学习 + 非对称 AC）；v0 保持不变
gym.register(
    id="Pick-Apple-SO101-Staged-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.staged_env_cfg:StagedPickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.staged_ppo_cfg:StagedPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 腕部相机版（纯视觉 actor：腕部 RGB + 本体感受，CNN 编码；阶段状态机/
# reset 课程/奖励/物理全部继承 staged；critic 保留特权观测）
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraStagedPickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraStagedPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v1：HOME 出生橘子刷进腕摄可见窗口（cam_v2 复盘：HOME 出生腕摄看不到
# 目标 → 盲搜，33.5 万次出生 0 成功；出生即入镜把 HOME→PREGRASP 变成可学的
# 视觉伺服）。PPO 算法修复（entropy 0.001 / desired_kl 0.02）在
# camera_ppo_cfg 的 CAMERA_ALGO_CFG，v0/v1 共用；v0 的 env cfg 与注册零改动
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraVisibleHomeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraStagedPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v2：前段教程式势能差塑形（cam_v4 平台期复盘：tanh 存在包远场零梯度
# + 驻留津贴 → home_born 全程 0；换 ApproachProgressReward 同款势能差，
# active 直到 grasped）+ home 0.40 加权课程。v1 保留可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraFrontShapedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraStagedPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v3：奖励全表 ×1/30 到教程量级 + PPO 回教程配方（cam_v5 复盘：std
# 钉 0.3 与有效 LR ~1e-4 均与教程日志一致、非病根；病根是 Loss/value
# 500-1000 vs 教程 0.1——每局回报 ~200 vs 教程 ~7，critic 拟合不动 →
# advantage 噪声 → 峰值 87.6% 后回落。v2 语义/比例全保留仅缩尺度）。
# v2 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraTutorialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v4：v3 缩放奖励（cam_v6 实证 Loss/value 329→0.06）+ cam_v5 固定 LR
# 优化器 CAMERA_ALGO_CFG（1e-4/entropy 0.001）。cam_v6 复盘：adaptive 在
# CNN 相机 actor 上第三次钉死 1e-5 饿死（teacher_fine_v4 同配方状态 MLP
# 却能自恢复起飞 → 触底为 CNN 特有）。v3 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraTutorialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialFixedLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v5：v4 配方不动（缩放奖励+固定 LR），HOME 出生几何换 O 姿态+双窗口
# （与蒸馏 v4 同款：scan_home_robot_cfg/SCAN_ORANGE_WINDOW/HOME_BOX_WINDOW，
# 关节噪声 ±0.05）——直训线与蒸馏线学生同出生分布，可互相对照；旧教师
# model_4250 在该几何下实测 96.1%。v3/v4 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraScanHomeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialFixedLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v6：v5 + episode 90s→20s（与蒸馏 v5 同单变量，两线同日对照启动）。
# PPO 配方沿用 v4/v5 的 CameraTutorialFixedLRPPORunnerCfg；v5 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraScanHomeShortEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialFixedLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v7：v5 + 合爪盲区修复相机（与蒸馏 v6 同款）+ 128×96（FOV 放大必须
# 补像素）+ episode 回 90s（v6 的 20s 实测勒死学习者：time_out 88%、LIFT
# 出生 0）。与蒸馏 v6 学生同出生分布+同相机，直接对照；v6 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraClosureScanHomeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialFixedLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v8：v7 + 课程三档钉死为固定表（home 0.50）——精修/续训专用口径。
# cam_v9_cont 复盘：续训课程归零重爬 + HOME 链占梯度 → bank 中段出生遗忘
# 性退化（LIFT 51→22%/STAB 43→7%）。教程全程固定任务分布（无课程），固定
# 出生分布消除续训瞬态。PPO 配方沿用 v7；v7 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v8",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraClosureFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialFixedLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v9：v8 + 腕摄 256×192——蒸馏学生 PPO 收尾（2026-10-07 路线 A）。
# actor 从 v9 蒸馏学生热启动（环境变量 PICK_PPO_ACTOR_FROM_DISTILL，
# agents/hybrid_ppo.HybridPPO），critic 从 --checkpoint 热启动，LR 3e-5。
# v8 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v9",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraFineTuneHdEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraFineTuneLowLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 相机 v10（Plan B 物理域随机化）：v8 唯一变量 = events 换 PhysicsDREventsCfg
# （执行器带逐局随机）。训练配方照抄 cam_v12 夜巡：强 checkpoint 续训 +
# LR 3e-5 注入 + 512 envs + guard-eval。v8 注册不动可复现
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:CameraClosureDRFinetuneEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.camera_ppo_cfg:CameraTutorialFixedLRPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# Latest distillation: standalone raw-label/action configuration.
gym.register(
    id="Pick-Apple-SO101-Staged-Camera-Distill-v10",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_distill_v10_cfg:CameraDistillV10PickAppleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.camera_distill_v10_cfg:CameraDistillV10RunnerCfg",
        "default_agent": "rsl_rl",
    },
)

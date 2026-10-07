"""腕部相机观测项（纯视觉 actor 专用）。

IsaacLabTutorial ``DomainRandomizedCameraImage`` 的移植：读取腕部相机
RGB，转 CHW float[0,1]，叠加 episode 一致的光度域随机化（曝光/对比度/
白平衡/亮度——每个环境每条 episode 重采样一次、episode 内保持不变，
实机相机漂移的近似）。play 模式下 IsaacLab 会关闭观测 corruption，
此时返回未随机化的原图（确定性评估口径）。

光度范围与噪声幅度均取教程在同款 SO101 腕部相机上验证过的值。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.managers import ManagerTermBase

from .staged_mdp import StagedPickState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import Camera


def _tensor(value):
    """Isaac Lab 代理数组转 torch 视图（非代理值原样返回）。"""
    return value.torch if hasattr(value, "torch") else value


class CameraImage(ManagerTermBase):
    """episode 一致光度域随机化的腕部 RGB 观测。

    输出 (N, 3, H, W) float[0,1]——rsl_rl ``CNNModel`` 按维度自动把该
    2D 观测组路由进卷积编码器。随机化参数每 reset 对被重置环境重采样。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._exposure_range = self._validate_range("exposure_range", cfg.params["exposure_range"], positive=True)
        self._contrast_range = self._validate_range("contrast_range", cfg.params["contrast_range"], positive=True)
        self._white_balance_range = self._validate_range(
            "white_balance_range", cfg.params["white_balance_range"], positive=True
        )
        self._brightness_range = self._validate_range("brightness_range", cfg.params["brightness_range"])
        shape = (self.num_envs, 1, 1, 1)
        self._exposure = torch.ones(shape, device=self.device)
        self._contrast = torch.ones(shape, device=self.device)
        self._white_balance = torch.ones((self.num_envs, 3, 1, 1), device=self.device)
        self._brightness = torch.zeros(shape, device=self.device)
        self.reset()

    @staticmethod
    def _validate_range(name: str, values: tuple[float, float], *, positive: bool = False) -> tuple[float, float]:
        lower, upper = values
        if not lower <= upper or (positive and lower <= 0.0):
            qualifier = "positive and ordered" if positive else "ordered"
            raise ValueError(f"{name} must be {qualifier}, got {values}.")
        return float(lower), float(upper)

    @staticmethod
    def _resample(tensor: torch.Tensor, env_ids: Sequence[int] | None, value_range: tuple[float, float]) -> None:
        if env_ids is None:
            tensor.uniform_(*value_range)
        else:
            tensor[env_ids] = torch.empty_like(tensor[env_ids]).uniform_(*value_range)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._resample(self._exposure, env_ids, self._exposure_range)
        self._resample(self._contrast, env_ids, self._contrast_range)
        self._resample(self._white_balance, env_ids, self._white_balance_range)
        self._resample(self._brightness, env_ids, self._brightness_range)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        exposure_range: tuple[float, float],
        contrast_range: tuple[float, float],
        white_balance_range: tuple[float, float],
        brightness_range: tuple[float, float],
    ) -> torch.Tensor:
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        image = _tensor(camera.data.output["rgb"])[..., :3]
        image = image.permute(0, 3, 1, 2).contiguous().float().div(255.0)
        # play 模式（enable_corruption=False）返回原图；训练模式叠加光度 DR。
        # 组名硬编码为 wrist_rgb 与本任务图像观测组一致（教程同款）。
        group_cfg = getattr(env.cfg.observations, "wrist_rgb", None)
        if group_cfg is not None and group_cfg.enable_corruption:
            image = (image - 0.5) * self._contrast + 0.5
            image = (image * self._exposure * self._white_balance + self._brightness).clamp(0.0, 1.0)
        return image


class CameraApproachProgress(ManagerTermBase):
    """前段势能差塑形：口袋-橘子距离每减小 ``scale`` 米付一单位，直到夹持锁存。

    IsaacLabTutorial ``ApproachProgressReward`` 的移植（教程纯摄像头 PPO
    以此达 99%）：势能差付的是「接近动作」而非「在场状态」——25cm 和 2cm
    处给同样的梯度，纯像素策略才能从 home 姿态发现接近路径；tanh 存在包
    （staged 的 home_approach/pregrasp_distance）在 2 倍 std 外是平的，
    橘子刷位距爪 15-35cm 正好落在盲区，这是 cam_v4 home_born_success
    全程 0 的奖励侧根因。

    门控用 ``~progress.grasped``（锁存，stages.py:240）而非阶段号：
    * 全未夹持状态空间对称计账（C1）——撞飞橘子付负、再逼近只是回本，
      往复净收益恒为零，无套利环；
    * GRASP 段把橘子撞开后仍有恢复梯度（阶段门控项在 GRASP 段断供）；
    * 锁存不可逆——一旦夹持过本项永久关闭，掉橘有 premature_release /
      orange_dropped 兜底，无法开合刷分。
    势能 ``min(d, d_max)`` 封顶（军规：每局总支付有界）；无 per-step
    clamp——臂 scale 0.033 rad/步，单步位移远小于 scale=1cm，clamp 在
    物理行程外（C2 的教训是 clamp 落在物理行程内才破坏 telescoping）。
    reset 后第一步 ``_has_prev=False`` 付 0（C5 出生白拿）。
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._has_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_prev[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01, d_max: float = 0.45) -> torch.Tensor:
        state = StagedPickState.get(env)
        state.update(env)
        dist = (state.orange_pos - state.pocket_pos).norm(dim=-1).clamp(max=d_max)
        pay = torch.where(
            self._has_prev & ~state.progress.grasped,
            (self._prev_dist - dist) / scale,
            torch.zeros_like(dist),
        )
        self._prev_dist.copy_(dist)
        self._has_prev.fill_(True)
        return pay


class SafeCameraApproachProgress(CameraApproachProgress):
    """CameraApproachProgress 的 NaN 防护版（与 SafeGraspClose 同模式）。"""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.nan_to_num(super().__call__(env), nan=0.0, posinf=1.0, neginf=-1.0)

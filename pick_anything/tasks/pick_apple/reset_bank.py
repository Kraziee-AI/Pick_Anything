"""物理验证 reset 状态数据集的保存 / 加载 / 校验（reset state bank）。

设计移植自 IsaacLabTutorial 的 ``reset/dataset.py``（Apache-2.0，本仓库自
建适配版）。每行是一条**经过物理验证**的完整状态：

    joint_position (6,)   5 个手臂关节 + 夹爪（rad，USD 关节序）
    joint_target    (6,)  恢复时写入控制器的关节目标（软舵机不瞬移）
    orange_pose     (7,)  env 局部系位置 + xyzw 四元数（唯一抓取目标）
    box_pose        (7,)  env 局部系位姿（纸盒可被推动/旋转，必须整行恢复）
    stage           ()    出生阶段 0..7（PickStage 枚举值）
    difficulty      ()    阶段进度 (stage + 段内 fraction) / 7，范围 [0, 1]
    grasped         ()    生成器验证时的物理夹持事实（bool；阶段 >=3 出生
                          即视为已夹住，与表示口径一致）

2026-09-25 schema v1（橘子单目标版）：删除 apple_pose 与干扰物 orange 的
双果字段，apple_grasped 更名 grasped；旧（苹果/立方体版）数据集与本
schema 不兼容，加载直接报错，绝不静默混用。

不变量：

* 所有位姿均在 env 局部系（世界系减 env 原点）——恢复时由加载方加回
  ``env_origins``，兼容任意 env_spacing。
* 四元数归一化、IsaacLab 3.0 的 (x, y, z, w) 顺序。
* 文件带 SHA-256 内容摘要，加载时强校验 + schema 版本检查，损坏或版本
  不符的数据集直接报错，绝不静默使用。
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pick_anything.tasks.pick_apple.stages import NUM_STAGES, STAGE_NAMES

FORMAT = "pick-orange-reset-dataset"
SCHEMA_VERSION = 1
STATE_SHAPES: dict[str, tuple[int, ...]] = {
    "joint_position": (6,),
    "joint_target": (6,),
    "orange_pose": (7,),
    "box_pose": (7,),
    "stage": (),
    "difficulty": (),
    "grasped": (),
}
_FLOAT_FIELDS = ("joint_position", "joint_target", "orange_pose", "box_pose", "difficulty")
_BOOL_FIELDS = ("grasped",)
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}

# 各阶段最少有效样本数：低于该数训练会打 WARNING 并把该阶段采样权重
# 收敛到 0（回退到已有有效阶段），绝不静默使用不足的数据
MIN_SAMPLES_PER_STAGE = 16


def _validate_state_fields(states: Mapping[str, Any]) -> None:
    """字段集合与 dtype 严格校验（多字段/缺字段/错类型立即报错）。"""
    if set(states) != set(STATE_SHAPES):
        missing = sorted(set(STATE_SHAPES) - set(states), key=repr)
        extra = sorted(set(states) - set(STATE_SHAPES), key=repr)
        raise ValueError(f"Reset state fields do not match the schema (missing={missing}, extra={extra}).")
    for name, value in states.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Reset field {name!r} must be a tensor.")
    for name in _FLOAT_FIELDS:
        if not states[name].is_floating_point():
            raise ValueError(f"Reset field {name!r} must have a floating-point dtype.")
    if states["stage"].dtype not in _INTEGER_DTYPES:
        raise ValueError("Reset field 'stage' must have an integer dtype.")
    for name in _BOOL_FIELDS:
        if states[name].dtype != torch.bool:
            raise ValueError(f"Reset field {name!r} must have a boolean dtype.")


def _canonical_states(states: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """统一为 CPU float32/int64/bool 连续张量（稳定存储 dtype）。"""
    if not isinstance(states, Mapping):
        raise ValueError("Reset states must be a mapping.")
    _validate_state_fields(states)
    canonical = {
        name: states[name].detach().cpu().to(torch.float32).contiguous()
        for name in _FLOAT_FIELDS
    }
    canonical["stage"] = states["stage"].detach().cpu().to(torch.int64).contiguous()
    canonical["grasped"] = states["grasped"].detach().cpu().to(torch.bool).contiguous()
    return canonical


def validate_reset_states(states: Mapping[str, Any]) -> int:
    """完整校验并返回行数（行数必须 >= 1）。"""
    if not isinstance(states, Mapping):
        raise ValueError("Reset states must be a mapping.")
    _validate_state_fields(states)
    row_count = int(states["stage"].shape[0]) if states["stage"].ndim == 1 else 0
    if row_count == 0:
        raise ValueError("Reset dataset must contain at least one row.")
    for name, trailing_shape in STATE_SHAPES.items():
        expected = (row_count, *trailing_shape)
        if tuple(states[name].shape) != expected:
            raise ValueError(f"Reset field {name!r} must have shape {expected}, got {tuple(states[name].shape)}.")
    for name in _FLOAT_FIELDS:
        if not bool(torch.isfinite(states[name]).all()):
            raise ValueError(f"Reset field {name!r} contains non-finite values.")
    if bool(((states["stage"] < 0) | (states["stage"] >= NUM_STAGES)).any()):
        raise ValueError("Reset stage IDs are outside the declared stage table.")
    if bool(((states["difficulty"] < 0.0) | (states["difficulty"] > 1.0)).any()):
        raise ValueError("Reset difficulty must lie in [0, 1].")
    for name in ("orange_pose", "box_pose"):
        quat_norm = torch.linalg.vector_norm(states[name][:, 3:7], dim=-1)
        if not bool(torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=2.0e-3, rtol=0.0)):
            raise ValueError(f"Reset {name} quaternions must be normalized XYZW quaternions.")
    return row_count


def stage_row_counts(stage: torch.Tensor) -> torch.Tensor:
    """每个阶段的行数（长度 NUM_STAGES 的 long 张量）。"""
    return torch.bincount(stage.cpu().to(torch.int64), minlength=NUM_STAGES)


_DIGEST_FIELDS = ("joint_position", "joint_target", "orange_pose", "box_pose", "stage", "difficulty", "grasped")


def _content_digest(states: Mapping[str, torch.Tensor]) -> str:
    """按固定字段顺序计算 SHA-256（dtype + shape + 原始字节）。"""
    digest = hashlib.sha256()
    for name in _DIGEST_FIELDS:
        tensor = states[name]
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def content_digest(states: Mapping[str, Any]) -> str:
    return _content_digest(_canonical_states(states))


def save_reset_dataset(
    path: str | Path,
    states: Mapping[str, Any],
    *,
    generator: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """校验并原子写入一个 reset 数据集工件。

    ``generator`` / ``validation`` 元数据会原样存入文件（生成参数、各阶段
    样本数、验证通过率等），便于追溯数据 provenance。
    """
    if not isinstance(generator, dict) or not isinstance(validation, dict):
        raise ValueError("Reset artifact generator and validation metadata must be dictionaries.")
    canonical = _canonical_states(states)
    row_count = validate_reset_states(canonical)
    artifact = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "stage_names": STAGE_NAMES,
        "states": canonical,
        "generator": generator,
        "validation": validation,
        "content_sha256": _content_digest(canonical),
        "row_count": row_count,
    }
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # 原子写：临时文件 + os.replace，中断不会留下半个数据集
    mode = output.stat().st_mode & 0o777 if output.exists() else 0o644
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as file:
            temporary = Path(file.name)
        torch.save(artifact, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return artifact


def load_reset_dataset(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """加载并强校验一个 reset 数据集工件（损坏即抛错，绝不静默使用）。"""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Reset dataset not found at {source}. Generate it with: "
            f"python scripts/generate_reset_bank.py --output {source}"
        )
    artifact = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError(f"{source} is not a {FORMAT!r} artifact.")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reset schema {artifact.get('schema_version')}; expected {SCHEMA_VERSION}.")
    if tuple(artifact.get("stage_names", ())) != STAGE_NAMES:
        raise ValueError("Reset stage table does not match this task version.")
    for name in ("generator", "validation"):
        if not isinstance(artifact.get(name), dict):
            raise ValueError(f"Reset artifact {name!r} metadata must be a dictionary.")
    states = _canonical_states(artifact.get("states"))
    row_count = validate_reset_states(states)
    stored_row_count = artifact.get("row_count")
    if not isinstance(stored_row_count, int) or isinstance(stored_row_count, bool) or stored_row_count != row_count:
        raise ValueError("Reset artifact row_count does not match its state tensors.")
    if artifact.get("content_sha256") != _content_digest(states):
        raise ValueError("Reset artifact content digest does not match its state tensors.")
    artifact["states"] = {name: tensor.to(device) for name, tensor in states.items()}
    return artifact

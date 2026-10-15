"""Opt-in multi-object tabletop scene; importing this package registers no RL tasks."""

from .catalog import OBJECTS, resolve_target, sample_layout

__all__ = ["OBJECTS", "resolve_target", "sample_layout"]

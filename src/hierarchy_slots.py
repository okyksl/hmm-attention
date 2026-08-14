"""Shared hierarchical position layout for probes and teacher evaluation."""

from typing import Any, List, Mapping

import torch


VALID_SLOT_MODES = ("surface", "coarse")


def validate_slot_mode(mode: str) -> None:
    if mode not in VALID_SLOT_MODES:
        raise ValueError(
            f"hierarchy slot mode must be one of {VALID_SLOT_MODES}; got {mode!r}"
        )


def slot_mode_from_config(
    config: Mapping[str, Any],
    default: str = "coarse",
) -> str:
    """Read the shared mode, falling back to the legacy probe-only field."""
    misc = config.get("misc", {})
    evaluation = misc.get("evaluation", {})
    probe = misc.get("probe", {})
    mode = evaluation.get("slot_mode", probe.get("slot_mode", default))
    validate_slot_mode(mode)
    return mode


def hierarchy_slot_counts(teacher, mode: str) -> List[int]:
    """Number of reported slots at every level, including the surface level."""
    validate_slot_mode(mode)
    if mode == "surface":
        return list(teacher._span)
    return [level.size for level in teacher.levels] + [1]


def hierarchy_slot_ids(
    positions: torch.Tensor,
    teacher,
    level: int,
    mode: str,
) -> torch.Tensor:
    """Map absolute surface positions to reported slots at one hierarchy level."""
    validate_slot_mode(mode)
    span = teacher._span[level]
    if mode == "surface":
        return positions % span
    child_span = teacher._span[level + 1] if level < teacher.num_levels else 1
    return (positions % span) // child_span

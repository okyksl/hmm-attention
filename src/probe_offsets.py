"""Resolve probe offsets on a level-aware surface-token horizon.

Offsets remain expressed in units of the probed teacher level.  The automatic
grid merely chooses how many such units to include, so ``k0`` and ``k-1`` keep
their local meaning at every depth.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Union


OffsetConfig = Optional[Union[Sequence[int], Sequence[Sequence[int]]]]
VALID_OFFSET_MODES = ("auto", "legacy")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def normalize_offsets_by_level(
    offsets: Union[Sequence[int], Sequence[Sequence[int]]],
    num_levels: int,
) -> List[List[int]]:
    """Normalize a broadcast flat list or an explicit per-level nested list."""
    if num_levels < 1:
        raise ValueError("num_levels must be positive")
    raw = list(offsets)
    if not raw:
        raise ValueError("probe offsets must not be empty")

    is_nested = isinstance(raw[0], Sequence) and not isinstance(
        raw[0], (str, bytes)
    )
    if is_nested:
        if len(raw) != num_levels:
            raise ValueError(
                f"per-level probe offsets have {len(raw)} levels; "
                f"expected {num_levels}"
            )
        resolved = [
            [int(value) for value in level_offsets] for level_offsets in raw
        ]
    else:
        shared = [int(value) for value in raw]
        resolved = [list(shared) for _ in range(num_levels)]

    for level, values in enumerate(resolved):
        if not values:
            raise ValueError(f"probe offsets for level {level} must not be empty")
        if len(values) != len(set(values)):
            raise ValueError(f"probe offsets for level {level} contain duplicates")
    return resolved


def resolve_probe_offsets(
    level_spans: Sequence[int],
    surface_burn_in: int,
    configured: OffsetConfig = None,
    mode: str = "auto",
) -> List[List[int]]:
    """Return offsets for every probed level.

    ``auto`` compares levels over two common surface-token horizons:

    * retention covers the teacher's full burn-in/context horizon;
    * planning covers one top-level latent unit.

    Dividing those horizons by each level's surface span naturally gives lower
    levels a denser/wider range.  ``legacy`` broadcasts the old
    ``[-base_context, ..., 0, +1]`` range to every level.  Any explicit flat or
    nested ``configured`` value overrides the mode.
    """
    spans = [int(span) for span in level_spans]
    if not spans or any(span < 1 for span in spans):
        raise ValueError("level_spans must contain positive integers")
    if surface_burn_in < 1:
        raise ValueError("surface_burn_in must be positive")
    if configured is not None:
        return normalize_offsets_by_level(configured, len(spans))
    if mode not in VALID_OFFSET_MODES:
        raise ValueError(
            f"probe offset mode must be one of {VALID_OFFSET_MODES}; got {mode!r}"
        )

    top_span = spans[0]
    if mode == "legacy":
        base_context = _ceil_div(surface_burn_in, top_span)
        shared = list(range(-base_context, 2))
        return [list(shared) for _ in spans]

    return [
        list(
            range(
                -_ceil_div(surface_burn_in, span),
                _ceil_div(top_span, span) + 1,
            )
        )
        for span in spans
    ]


def all_probe_offsets(offsets_by_level: Sequence[Sequence[int]]) -> List[int]:
    """Sorted union used only by axes that compare several teacher levels."""
    return sorted(
        {int(offset) for offsets in offsets_by_level for offset in offsets}
    )

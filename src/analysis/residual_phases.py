"""Learning-order analysis for residual-stream probes.

The attention-head analysis in :mod:`src.analysis.head_phases` is useful for a
flat, one-block student, but head identities become difficult to interpret once
either the teacher hierarchy or the student is deep.  The probe logger already
provides a more stable coordinate system:

``(residual layer, teacher level, within-unit slot, relative-unit offset)``.
Teacher levels include the terminal surface-token level.

This module turns those scalar histories into comparable trajectories and
stable acquisition events.  Its primary score normalizes the logged excess NLL
between a uniform predictor and the teacher's Bayes optimum.  This is important
because different teacher levels can have different alphabet sizes and
intrinsic uncertainty, so raw accuracies do not live on the same scale.

The resulting event table supports two complementary questions:

* for a fixed latent target, which residual layer becomes linearly decodable
  first?; and
* within a fixed residual layer, which latent targets become decodable first?

No causal claim is made: a probe measures linear decodability at a checkpoint,
not where a representation was computed or whether the model uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, prod
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.hierarchy_slots import slot_mode_from_config
from src.probe_offsets import (
    all_probe_offsets,
    normalize_offsets_by_level,
    resolve_probe_offsets,
)


PROBE_METRICS = (
    "acc",
    "nll",
    "n",
    "bayes_acc",
    "bayes_nll",
    "excess_nll",
)
BAYES_METRICS = ("bayes_acc", "bayes_nll", "excess_nll")


def offset_name(offset: int) -> str:
    """The unambiguous offset component used by :class:`ProbeLogger`."""
    return f"k+{offset}" if offset > 0 else f"k{offset}"


def offset_regime(offset: int) -> str:
    """Semantic name for a probe target's relative-unit offset."""
    if offset < 0:
        return "retention"
    if offset == 0:
        return "current/refinement"
    return "planning"


def target_label(level: int, slot: int, offset: int) -> str:
    """Compact, sortable name for one latent target."""
    return f"level{level}/slot{slot}/{offset_name(offset)}"


@dataclass(frozen=True)
class ProbeSpec:
    """Resolved dimensions of the probe grid for one training run.

    ``num_layers`` includes ``L0`` (post input/position encoder), so a student
    with ``N`` transformer blocks has ``N + 1`` probed residual streams.
    ``slots_per_level`` and ``class_counts`` are ordered top-to-bottom in the
    teacher hierarchy and include the terminal surface-token level.
    """

    num_layers: int
    slots_per_level: List[int]
    offsets: Union[List[int], List[List[int]]]
    class_counts: List[int]
    slot_mode: str = "coarse"

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must include at least L0")
        if len(self.slots_per_level) != len(self.class_counts):
            raise ValueError("slots_per_level and class_counts must have equal length")
        if any(n < 1 for n in self.slots_per_level):
            raise ValueError("every teacher level must have at least one slot")
        if any(n < 2 for n in self.class_counts):
            raise ValueError("every probed alphabet must contain at least two classes")
        if self.slot_mode not in {"surface", "coarse"}:
            raise ValueError("slot_mode must be 'surface' or 'coarse'")
        normalize_offsets_by_level(self.offsets, self.num_levels)

    @property
    def num_levels(self) -> int:
        return len(self.slots_per_level)

    @property
    def level_spans(self) -> List[int]:
        """Number of surface tokens covered by one unit at each level."""
        if self.slot_mode == "surface":
            return list(self.slots_per_level)
        return [prod(self.slots_per_level[level:]) for level in range(self.num_levels)]

    @property
    def offsets_by_level(self) -> List[List[int]]:
        """Configured offsets normalized to one list per teacher level."""
        return normalize_offsets_by_level(self.offsets, self.num_levels)

    @property
    def all_offsets(self) -> List[int]:
        """Sorted union for plots whose level is itself an axis."""
        return all_probe_offsets(self.offsets_by_level)


def _base_burn_in(config: Mapping[str, Any]) -> int:
    teacher = config["teacher"]
    base = teacher.get("base_teacher", teacher)
    burn_in = base.get("burn_in")
    if burn_in is None and base.get("span_lengths"):
        span_lengths = list(base["span_lengths"])
        stride = base.get("stride")
        burn_in = (
            (len(span_lengths) - 1) * int(stride) + span_lengths[-1]
            if stride is not None
            else sum(span_lengths)
        )
    if burn_in is None:
        burn_in = base.get("window", config.get("dataset", {}).get("window"))
    if burn_in is None:
        raise ValueError("cannot infer default probe offsets from run config")
    return int(burn_in)


def probe_spec_from_config(config: Mapping[str, Any]) -> ProbeSpec:
    """Infer probe dimensions and class counts from a resolved run config."""
    teacher = config["teacher"]
    student = config["student"]
    base = teacher.get("base_teacher", teacher)
    dataset_dim = config.get("dataset", {}).get("dim")
    probe = config.get("misc", {}).get("probe", {})

    def resolved_dim(value: Any) -> int:
        if value is None:
            if dataset_dim is None:
                raise ValueError("cannot infer the terminal surface alphabet size")
            return int(dataset_dim)
        dimension = int(value)
        if dimension == -1:
            if dataset_dim is None:
                raise ValueError("chunk_dim=-1 requires dataset.dim")
            return int(dataset_dim)
        return dimension

    if "levels" in teacher:
        levels = list(teacher["levels"])
        chunk_sizes = [
            int(level.get("chunk_size", level.get("size"))) for level in levels
        ]
        base_dim = int(base["dim"])
        class_counts = [base_dim] + [
            resolved_dim(level.get("chunk_dim", level.get("out_dim")))
            for level in levels
        ]
    elif "chunk_sizes" in teacher:
        chunk_sizes = [int(size) for size in teacher["chunk_sizes"]]
        dimensions = list(teacher.get("chunk_dims", []))
        if len(dimensions) != len(chunk_sizes):
            raise ValueError("teacher.chunk_dims and chunk_sizes must have equal length")
        class_counts = [int(base["dim"])] + [
            resolved_dim(dimension) for dimension in dimensions
        ]
    elif "chunk_size" in teacher:
        chunk_sizes = [int(teacher["chunk_size"])]
        class_counts = [
            int(base["dim"]),
            resolved_dim(teacher.get("chunk_dim", teacher.get("dim"))),
        ]
    else:
        raise ValueError(
            "probe analysis requires a hierarchical teacher config with "
            "teacher.levels, teacher.chunk_sizes, or teacher.chunk_size"
        )

    slot_mode = slot_mode_from_config(config)
    if slot_mode == "surface":
        slots = [prod(chunk_sizes[level:]) for level in range(len(chunk_sizes))]
    elif slot_mode == "coarse":
        slots = list(chunk_sizes)
    else:
        raise ValueError("hierarchy slot_mode must be 'surface' or 'coarse'")
    # The terminal surface level has one token per unit and therefore one slot.
    slots.append(1)

    level_spans = [prod(chunk_sizes[level:]) for level in range(len(chunk_sizes))]
    level_spans.append(1)
    configured_offsets = probe.get("offsets")
    # Runs created before level-aware offsets had no mode field and used the
    # same base-unit range at every level. Preserve their readable layout.
    offset_mode = probe.get("offset_mode", "legacy")
    offsets_by_level = resolve_probe_offsets(
        level_spans,
        surface_burn_in=_base_burn_in(config) * level_spans[0],
        configured=configured_offsets,
        mode=offset_mode,
    )

    return ProbeSpec(
        num_layers=int(student["num_blocks"]) + 1,
        slots_per_level=slots,
        offsets=offsets_by_level,
        class_counts=class_counts,
        slot_mode=slot_mode,
    )


def probe_metric_key(
    layer: int,
    level: int,
    slot: int,
    offset: int,
    metric: str = "acc",
    split: str = "val",
) -> str:
    """Return one scalar W&B key emitted by :class:`ProbeLogger`."""
    return (
        f"probe/L{layer}/level{level}/slot{slot}/{offset_name(offset)}/"
        f"{metric}/{split}"
    )


def probe_metric_keys(
    spec: ProbeSpec,
    metrics: Sequence[str] = PROBE_METRICS,
    split: str = "val",
) -> List[str]:
    """All requested *logged* probe keys for a resolved run, in grid order.

    The current logger has no Bayes calculation for planning targets (positive
    offsets), so Bayes-derived keys are omitted there even when requested.
    This distinction matters for W&B: ``scan_history(keys=...)`` only returns
    rows on which every requested key exists.
    """
    unknown = sorted(set(metrics) - set(PROBE_METRICS))
    if unknown:
        raise ValueError(f"unknown probe metrics: {unknown}")
    return [
        probe_metric_key(layer, level, slot, offset, metric, split)
        for level, num_slots in enumerate(spec.slots_per_level)
        for slot in range(num_slots)
        for offset in spec.offsets_by_level[level]
        for layer in range(spec.num_layers)
        for metric in metrics
        if offset <= 0 or metric not in BAYES_METRICS
    ]


def _metric_series(
    history: pd.DataFrame, step_key: str, key: str
) -> pd.DataFrame:
    """One metric series, merging duplicate sparse W&B rows by last value."""
    if key not in history:
        return pd.DataFrame(columns=[step_key, key])
    series = history[[step_key, key]].dropna(subset=[key]).sort_values(step_key)
    return series.drop_duplicates(step_key, keep="last")


def probe_history_long(
    history: pd.DataFrame,
    spec: ProbeSpec,
    split: str = "val",
    step_key: str = "_step",
    strict: bool = False,
) -> pd.DataFrame:
    """Convert sparse probe histories to one tidy row per grid cell and step.

    The output includes raw metrics plus:

    ``chance_acc``
        Uniform-chance accuracy, ``1 / number_of_classes``.
    ``attainable_acc``
        The logged Bayes accuracy when present.  Planning targets do not yet
        have a logged Bayes ceiling, so their fallback is 1.0 and is explicitly
        labelled ``unit_fallback`` in ``ceiling_source``.
    ``acc_progress``
        ``(acc - chance) / (attainable - chance)``.  Values are deliberately not
        clipped, preserving below-chance results and ceiling overshoots.
    ``nll_progress``
        ``1 - excess_nll / (log(number_of_classes) - bayes_nll)``.  This is the
        preferred cross-level score because it accounts for each level's
        alphabet size and intrinsic Bayes uncertainty.

    Missing optional metrics remain NaN.  With ``strict=False`` (the default),
    missing accuracy cells are skipped; ``missing_probe_cells`` can be used to
    report them.  At least one accuracy series must exist.
    """
    if step_key not in history:
        raise KeyError(f"history is missing step column {step_key!r}")

    rows: List[pd.DataFrame] = []
    missing: List[str] = []
    for level, num_slots in enumerate(spec.slots_per_level):
        chance = 1.0 / spec.class_counts[level]
        for slot in range(num_slots):
            for offset in spec.offsets_by_level[level]:
                for layer in range(spec.num_layers):
                    keys = {
                        metric: probe_metric_key(
                            layer, level, slot, offset, metric, split
                        )
                        for metric in PROBE_METRICS
                    }
                    acc_key = keys["acc"]
                    if acc_key not in history or history[acc_key].notna().sum() == 0:
                        missing.append(acc_key)
                        continue

                    cell = _metric_series(history, step_key, acc_key).rename(
                        columns={acc_key: "acc"}
                    )
                    for metric in PROBE_METRICS:
                        if metric == "acc":
                            continue
                        series = _metric_series(history, step_key, keys[metric])
                        if not series.empty:
                            series = series.rename(columns={keys[metric]: metric})
                            cell = cell.merge(series, on=step_key, how="left")

                    for metric in PROBE_METRICS:
                        if metric not in cell:
                            cell[metric] = np.nan
                    cell["layer"] = layer
                    cell["level"] = level
                    cell["slot"] = slot
                    cell["offset"] = offset
                    cell["regime"] = offset_regime(offset)
                    cell["target"] = target_label(level, slot, offset)
                    cell["chance_acc"] = chance

                    has_bayes = cell["bayes_acc"].notna()
                    cell["attainable_acc"] = cell["bayes_acc"].where(has_bayes, 1.0)
                    cell["ceiling_source"] = np.where(
                        has_bayes, "teacher_bayes", "unit_fallback"
                    )
                    denom = cell["attainable_acc"] - chance
                    cell["acc_progress"] = np.where(
                        denom > 1e-12, (cell["acc"] - chance) / denom, np.nan
                    )
                    cell["acc_progress_clipped"] = cell["acc_progress"].clip(0.0, 1.0)

                    uniform_nll = log(spec.class_counts[level])
                    nll_denom = uniform_nll - cell["bayes_nll"]
                    cell["nll_progress"] = np.where(
                        nll_denom > 1e-12,
                        1.0 - cell["excess_nll"] / nll_denom,
                        np.nan,
                    )
                    rows.append(cell)

    if strict and missing:
        raise KeyError(
            f"missing {len(missing)} probe accuracy series, e.g. {missing[0]!r}"
        )
    if not rows:
        example = probe_metric_key(
            0, 0, 0, spec.offsets_by_level[0][0], "acc", split
        )
        raise KeyError(
            "no probe accuracy histories matched the resolved layout; "
            f"for example expected {example!r}"
        )

    out = pd.concat(rows, ignore_index=True, sort=False)
    ordered = [
        step_key,
        "layer",
        "level",
        "slot",
        "offset",
        "regime",
        "target",
        *PROBE_METRICS,
        "chance_acc",
        "attainable_acc",
        "ceiling_source",
        "acc_progress",
        "acc_progress_clipped",
        "nll_progress",
    ]
    return out[ordered].sort_values(
        ["level", "slot", "offset", "layer", step_key]
    ).reset_index(drop=True)


def missing_probe_cells(
    history: pd.DataFrame,
    spec: ProbeSpec,
    split: str = "val",
) -> pd.DataFrame:
    """Expected probe cells whose accuracy series is absent or entirely NaN."""
    rows = []
    for level, num_slots in enumerate(spec.slots_per_level):
        for slot in range(num_slots):
            for offset in spec.offsets_by_level[level]:
                for layer in range(spec.num_layers):
                    key = probe_metric_key(layer, level, slot, offset, "acc", split)
                    if key not in history or history[key].notna().sum() == 0:
                        rows.append(
                            {
                                "layer": layer,
                                "level": level,
                                "slot": slot,
                                "offset": offset,
                                "key": key,
                            }
                        )
    return pd.DataFrame(rows)


def sustained_acquisition_step(
    steps: Sequence[float],
    scores: Sequence[float],
    threshold: float = 0.8,
    min_dwell: int = 3,
    require_final: bool = True,
) -> Optional[float]:
    """Return a stable threshold-crossing step from one score trajectory.

    With ``require_final=True``, acquisition is the start of the *final* run of
    above-threshold evaluations, and that run must contain at least
    ``min_dwell`` samples.  This ignores transient early successes and gives a
    precise "learned and retained" interpretation.  With ``require_final=False``
    the first above-threshold run of sufficient length is used instead.
    """
    if min_dwell < 1:
        raise ValueError("min_dwell must be at least 1")
    steps_arr = np.asarray(steps, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)
    if steps_arr.shape != scores_arr.shape:
        raise ValueError("steps and scores must have the same shape")
    finite = np.isfinite(steps_arr) & np.isfinite(scores_arr)
    if not finite.any():
        return None
    frame = (
        pd.DataFrame({"step": steps_arr[finite], "score": scores_arr[finite]})
        .sort_values("step")
        .drop_duplicates("step", keep="last")
    )
    steps_arr = frame["step"].to_numpy()
    above = frame["score"].to_numpy() >= threshold

    if require_final:
        if not above[-1]:
            return None
        below = np.flatnonzero(~above)
        start = int(below[-1]) + 1 if len(below) else 0
        if len(above) - start < min_dwell:
            return None
        return float(steps_arr[start])

    run_start: Optional[int] = None
    for i, is_above in enumerate(above):
        if is_above and run_start is None:
            run_start = i
        elif not is_above:
            run_start = None
        if run_start is not None and i - run_start + 1 >= min_dwell:
            return float(steps_arr[run_start])
    return None


def acquisition_table(
    long_history: pd.DataFrame,
    score: str = "nll_progress",
    threshold: float = 0.8,
    min_dwell: int = 3,
    require_final: bool = True,
    offsets: Optional[Sequence[int]] = None,
    step_key: str = "_step",
) -> pd.DataFrame:
    """One row per probe cell with its stable acquisition step."""
    required = {step_key, "layer", "level", "slot", "offset", score}
    missing = sorted(required - set(long_history.columns))
    if missing:
        raise KeyError(f"long probe history is missing columns: {missing}")
    selected = long_history
    if offsets is not None:
        selected = selected[selected["offset"].isin(offsets)]

    rows = []
    group_cols = ["level", "slot", "offset", "layer"]
    for (level, slot, offset, layer), cell in selected.groupby(
        group_cols, sort=True, dropna=False
    ):
        cell = cell.sort_values(step_key)
        acquired = sustained_acquisition_step(
            cell[step_key],
            cell[score],
            threshold=threshold,
            min_dwell=min_dwell,
            require_final=require_final,
        )
        finite = cell[score].dropna()
        rows.append(
            {
                "level": int(level),
                "slot": int(slot),
                "offset": int(offset),
                "regime": offset_regime(int(offset)),
                "target": target_label(int(level), int(slot), int(offset)),
                "layer": int(layer),
                "acquisition_step": acquired,
                "acquired": acquired is not None,
                "final_score": float(finite.iloc[-1]) if len(finite) else np.nan,
                "max_score": float(finite.max()) if len(finite) else np.nan,
                "n_evals": int(len(finite)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["level", "slot", "offset", "layer"]
    ).reset_index(drop=True)


def _step_text(value: Any) -> str:
    return "never" if pd.isna(value) else f"{float(value):g}"


def _event_story(group: pd.DataFrame, item_col: str) -> str:
    acquired = group[group["acquisition_step"].notna()].sort_values(
        ["acquisition_step", item_col]
    )
    parts: List[str] = []
    for step, tied in acquired.groupby("acquisition_step", sort=True):
        items = "=".join(str(item) for item in tied[item_col])
        parts.append(f"{items}@{_step_text(step)}")
    never = group[group["acquisition_step"].isna()].sort_values(item_col)
    if len(never):
        parts.append("=".join(str(item) for item in never[item_col]) + "@never")
    return " -> ".join(parts)


def layer_order_by_target(events: pd.DataFrame) -> pd.DataFrame:
    """For each latent target, order residual layers by acquisition time."""
    rows = []
    target_cols = ["level", "slot", "offset", "regime", "target"]
    for values, group in events.groupby(target_cols, sort=True, dropna=False):
        group = group.copy()
        group["layer_name"] = group["layer"].map(lambda layer: f"L{layer}")
        acquired = group[group["acquisition_step"].notna()]
        if acquired.empty:
            first = "never"
            first_step = np.nan
        else:
            first_step = float(acquired["acquisition_step"].min())
            tied = acquired[acquired["acquisition_step"] == first_step]
            first = "=".join(tied.sort_values("layer")["layer_name"])
        rows.append(
            dict(
                zip(target_cols, values),
                first_layer=first,
                first_step=first_step,
                layer_order=_event_story(group, "layer_name"),
                n_acquired=int(group["acquired"].sum()),
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["level", "slot", "offset"]
    ).reset_index(drop=True)


def target_order_by_layer(events: pd.DataFrame) -> pd.DataFrame:
    """For each residual layer, order latent targets by acquisition time."""
    rows = []
    for layer, group in events.groupby("layer", sort=True):
        acquired = group[group["acquisition_step"].notna()]
        first_step = (
            float(acquired["acquisition_step"].min()) if len(acquired) else np.nan
        )
        first_targets = (
            "=".join(
                acquired[acquired["acquisition_step"] == first_step]
                .sort_values("target")["target"]
            )
            if len(acquired)
            else "never"
        )
        rows.append(
            {
                "layer": int(layer),
                "residual": f"L{int(layer)}",
                "first_target": first_targets,
                "first_step": first_step,
                "target_order": _event_story(group, "target"),
                "n_acquired": int(group["acquired"].sum()),
            }
        )
    return pd.DataFrame(rows)


def adjacent_layer_deltas(events: pd.DataFrame) -> pd.DataFrame:
    """Acquisition-time differences between adjacent residual streams.

    A negative ``L{j}-L{i}`` value means the deeper residual ``Lj`` became
    decodable at an earlier training checkpoint than ``Li`` for that target.
    Deltas are NaN unless both layers acquired the target.
    """
    index = ["level", "slot", "offset", "regime", "target"]
    if events.duplicated(index + ["layer"]).any():
        raise ValueError("events contain duplicate target/layer rows")
    # ``pivot`` (rather than ``pivot_table``) retains targets that were never
    # acquired by any layer, which is diagnostically important.
    wide = events.pivot(
        index=index, columns="layer", values="acquisition_step"
    ).reset_index()
    layer_cols = sorted(col for col in wide.columns if isinstance(col, (int, np.integer)))
    for left, right in zip(layer_cols, layer_cols[1:]):
        wide[f"L{right}-L{left}"] = wide[right] - wide[left]
    return wide

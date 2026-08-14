"""Pure post-training plots for numerical attention and probe snapshots."""

from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.residual_phases import probe_metric_key
from src.probe_offsets import (
    all_probe_offsets,
    normalize_offsets_by_level,
    resolve_probe_offsets,
)
from src.visualizer import compute_attention_alignment


@dataclass(frozen=True)
class ProbeLayout:
    """Probe grid dimensions and configured relative offsets."""

    num_layers: int
    slots_per_level: List[int]
    offsets: Union[List[int], List[List[int]]]

    @property
    def offsets_by_level(self) -> List[List[int]]:
        return normalize_offsets_by_level(self.offsets, len(self.slots_per_level))

    @property
    def all_offsets(self) -> List[int]:
        return all_probe_offsets(self.offsets_by_level)


def probe_layout_from_config(config: Mapping[str, Any]) -> ProbeLayout:
    """Infer the logger's probe layout from a resolved run configuration."""
    teacher = config["teacher"]
    student = config["student"]
    if "chunk_sizes" in teacher:
        chunk_sizes = [int(size) for size in teacher["chunk_sizes"]]
    elif "levels" in teacher:
        chunk_sizes = [int(level["chunk_size"]) for level in teacher["levels"]]
    elif "chunk_size" in teacher:
        chunk_sizes = [int(teacher["chunk_size"])]
    else:
        chunk_sizes = []
    probe = config.get("misc", {}).get("probe", {})
    slot_mode = probe.get("slot_mode", "coarse")
    if slot_mode == "surface":
        slots = [prod(chunk_sizes[level:]) for level in range(len(chunk_sizes))]
    elif slot_mode == "coarse":
        slots = list(chunk_sizes)
    else:
        raise ValueError("probe slot_mode must be 'surface' or 'coarse'")
    # Chunk-code levels describe transformations between node alphabets.  Probe
    # their input alphabets plus the terminal surface alphabet, whose units are
    # individual tokens and therefore have one slot.
    if not slots:
        raise ValueError("probe layout requires a hierarchical teacher")
    slots.append(1)

    base = teacher.get("base_teacher", teacher)
    base_burn_in = base.get("burn_in")
    if base_burn_in is None and base.get("span_lengths"):
        span_lengths = list(base["span_lengths"])
        stride = base.get("stride")
        base_burn_in = (
            (len(span_lengths) - 1) * int(stride) + span_lengths[-1]
            if stride is not None
            else sum(span_lengths)
        )
    if base_burn_in is None:
        base_burn_in = base.get("window", config.get("dataset", {}).get("window"))
    if base_burn_in is None:
        raise ValueError("cannot infer default probe offsets from run config")
    level_spans = [prod(chunk_sizes[level:]) for level in range(len(chunk_sizes))]
    level_spans.append(1)
    offsets = resolve_probe_offsets(
        level_spans,
        surface_burn_in=int(base_burn_in) * level_spans[0],
        configured=probe.get("offsets"),
        # Configs from older runs lack this marker and used the legacy grid.
        mode=probe.get("offset_mode", "legacy"),
    )

    return ProbeLayout(
        num_layers=int(student["num_blocks"]) + 1,
        slots_per_level=slots,
        offsets=offsets,
    )


def attention_table_to_array(table: pd.DataFrame) -> np.ndarray:
    """Reconstruct ``(head, query, key)`` attention from a W&B table."""
    required = {"head", "query_idx", "key_idx", "weight"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"attention table is missing columns: {missing}")
    if table.duplicated(["head", "query_idx", "key_idx"]).any():
        raise ValueError("attention table contains duplicate head/query/key rows")

    heads = sorted(table["head"].unique())
    queries = sorted(table["query_idx"].unique())
    keys = sorted(table["key_idx"].unique())
    expected_rows = len(heads) * len(queries) * len(keys)
    if len(table) != expected_rows:
        raise ValueError(
            f"attention table is incomplete: expected {expected_rows} rows, "
            f"found {len(table)}"
        )

    matrices = []
    for head in heads:
        matrix = (
            table.loc[table["head"] == head]
            .pivot(index="query_idx", columns="key_idx", values="weight")
            .reindex(index=queries, columns=keys)
        )
        if matrix.isna().any().any():
            raise ValueError(f"attention table has missing cells for head {head}")
        matrices.append(matrix.to_numpy(dtype=np.float32))
    return np.stack(matrices)


def _matrix_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    xlabel: str = "",
    ylabel: str = "Student head",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=(max(4, len(col_labels)), max(3, len(row_labels)))
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=matrix.size <= 100,
        fmt=".2f",
        xticklabels=False,
        yticklabels=False,
        linewidths=0.5,
    )
    _set_readable_heatmap_ticks(ax, row_labels, col_labels)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def _tick_indices(size: int, max_ticks: int) -> np.ndarray:
    """Return evenly spaced indices, preserving both endpoints."""
    if size <= max_ticks:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, max_ticks, dtype=int))


def _set_readable_heatmap_ticks(
    ax: plt.Axes,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    max_ticks: int = 10,
) -> None:
    """Thin dense heatmap labels while retaining their cell-center alignment."""
    row_indices = _tick_indices(len(row_labels), max_ticks)
    col_indices = _tick_indices(len(col_labels), max_ticks)
    ax.set_xticks(
        col_indices + 0.5,
        [col_labels[index] for index in col_indices],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        row_indices + 0.5,
        [row_labels[index] for index in row_indices],
        rotation=0,
    )


def plot_attention_heatmaps(
    attn: np.ndarray,
    step: int,
    split: str = "val",
    token_seq: Optional[Sequence[str]] = None,
) -> Dict[str, plt.Figure]:
    """Recreate the former average and per-head attention heatmaps."""
    num_heads, seq_len, _ = attn.shape
    labels = list(token_seq) if token_seq is not None else [str(i) for i in range(seq_len)]
    matrices = [("average", "Average heads", attn.mean(axis=0))]
    matrices.extend(
        (f"head{h + 1}", f"Head {h + 1}", attn[h]) for h in range(num_heads)
    )

    figures: Dict[str, plt.Figure] = {}
    for key, label, matrix in matrices:
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            matrix,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            xticklabels=False,
            yticklabels=False,
            cbar=True,
        )
        _set_readable_heatmap_ticks(ax, labels, labels)
        ax.set_title(f"{label} ({split}, step {step})")
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        fig.tight_layout()
        figures[key] = fig
    return figures


def plot_attention_alignment(
    attn: np.ndarray,
    span_lengths: List[int],
    context_length: int,
    step: int,
    split: str = "val",
    stride: Optional[int] = None,
) -> Dict[str, plt.Figure]:
    """Recreate alignment heatmaps and per-head offset charts."""
    alignment = compute_attention_alignment(
        attn, span_lengths, context_length, stride=stride
    )
    num_heads = attn.shape[0]
    row_labels = [f"Head {h + 1}" for h in range(num_heads)]
    col_labels = [f"Span {k + 1}" for k in range(len(span_lengths))]
    figures: Dict[str, plt.Figure] = {}
    figures["align_cos_sim"] = _matrix_heatmap(
        alignment.cosine_similarity,
        row_labels,
        col_labels,
        title=f"Attention cosine similarity ({split}, step {step})",
        xlabel="GT span",
        vmin=-1.0,
        vmax=1.0,
    )
    abs_max = float(np.abs(alignment.projected_norm).max()) or 1.0
    figures["align_proj_norm"] = _matrix_heatmap(
        alignment.projected_norm,
        row_labels,
        col_labels,
        title=f"Attention projected norm ({split}, step {step})",
        xlabel="GT span",
        vmin=-abs_max,
        vmax=abs_max,
    )

    seq_len = attn.shape[-1]
    context_start = max(0, seq_len - context_length)
    positions = np.arange(context_start, seq_len)
    for h in range(num_heads):
        n_cols = len(span_lengths) + 1
        fig, axes = plt.subplots(
            1, n_cols, figsize=(3 * n_cols, 3), sharey=True, squeeze=False
        )
        axes = axes[0]
        axes[0].bar(positions, alignment.last_rows[h, context_start:], color="steelblue")
        axes[0].set_title(f"Head {h + 1}\n(student)", fontsize=9)
        axes[0].set_xlabel("Key pos")
        axes[0].set_ylabel("Weight")
        for k in range(len(span_lengths)):
            axes[k + 1].bar(
                positions,
                alignment.ground_truth[k, context_start:],
                color="coral",
                alpha=0.8,
            )
            axes[k + 1].set_title(
                f"GT span {k + 1}\ncos={alignment.cosine_similarity[h, k]:.2f}",
                fontsize=9,
            )
            axes[k + 1].set_xlabel("Key pos")
        fig.suptitle(
            f"Head {h + 1} — last-query attention ({split}, step {step})",
            fontsize=10,
        )
        fig.tight_layout()
        figures[f"offset_head{h + 1}"] = fig
    return figures


def plot_value_alignment(
    student_norms: np.ndarray,
    cosine_similarity: np.ndarray,
    step: int,
    split: str = "val",
) -> Dict[str, plt.Figure]:
    """Recreate value cosine/projected-norm heatmaps from logged scalars."""
    row_labels = [f"Head {h + 1}" for h in range(cosine_similarity.shape[0])]
    col_labels = [
        f"Teacher {k + 1}" for k in range(cosine_similarity.shape[1])
    ]
    projected = student_norms[:, None] * cosine_similarity
    abs_max = float(np.abs(projected).max()) or 1.0
    return {
        "value_cos_sim": _matrix_heatmap(
            cosine_similarity,
            row_labels,
            col_labels,
            title=f"Value matrix cosine similarity ({split}, step {step})",
            xlabel="Teacher matrix",
            vmin=-1.0,
            vmax=1.0,
        ),
        "value_proj_norm": _matrix_heatmap(
            projected,
            row_labels,
            col_labels,
            title=f"Value matrix projected norm ({split}, step {step})",
            xlabel="Teacher matrix",
            vmin=-abs_max,
            vmax=abs_max,
        ),
    }


def probe_matrix_from_metrics(
    metrics: Mapping[str, Any],
    level: int,
    offset: int,
    metric: str,
    split: str,
    num_layers: int,
    num_slots: int,
) -> np.ndarray:
    """Build the former layer×slot probe summary matrix from scalar history."""
    offset_name = f"k+{offset}" if offset > 0 else f"k{offset}"
    matrix = np.full((num_layers, num_slots), np.nan, dtype=np.float32)
    for layer in range(num_layers):
        for slot in range(num_slots):
            key = (
                f"probe/L{layer}/level{level}/slot{slot}/{offset_name}/"
                f"{metric}/{split}"
            )
            value = metrics.get(key)
            if value is not None and not pd.isna(value):
                matrix[layer, slot] = float(value)
    return matrix


def _probe_offset_matrix_from_metrics(
    metrics: Mapping[str, Any],
    level: int,
    slot: int,
    offsets: Sequence[int],
    metric: str,
    split: str,
    num_layers: int,
) -> np.ndarray:
    """Build a layer×offset matrix for one level/slot overview facet."""
    matrix = np.full((num_layers, len(offsets)), np.nan, dtype=np.float32)
    for layer in range(num_layers):
        for column, offset in enumerate(offsets):
            key = probe_metric_key(layer, level, slot, int(offset), metric, split)
            value = metrics.get(key)
            if value is not None and not pd.isna(value):
                matrix[layer, column] = float(value)
    return matrix


def plot_probe_heatmap(
    matrix: np.ndarray,
    level: int,
    offset: int,
    metric: str,
    step: int,
    split: str = "val",
) -> plt.Figure:
    """Recreate a former probe accuracy or excess-NLL summary heatmap."""
    if np.isnan(matrix).all():
        raise ValueError("probe matrix has no logged values")
    if metric == "acc":
        vmin, vmax, cmap, label = 0.0, 1.0, "viridis", "probe acc"
    elif metric == "excess_nll":
        vmin = 0.0
        vmax = max(float(np.nanmax(matrix)), 1e-6)
        cmap, label = "magma", "excess nll"
    else:
        raise ValueError("metric must be 'acc' or 'excess_nll'")

    n_layers, n_slots = matrix.shape
    fig, ax = plt.subplots(
        figsize=(1.5 + 0.6 * n_slots, 1.0 + 0.5 * n_layers)
    )
    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        annot=matrix.size <= 100,
        fmt=".2f",
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        ax=ax,
    )
    _set_readable_heatmap_ticks(
        ax,
        [f"L{layer}" for layer in range(n_layers)],
        [f"s{s}" for s in range(n_slots)],
    )
    ax.set_title(
        f"{label} — level{level}, {split}, k={offset:+d}, step {step}"
    )
    ax.set_xlabel("within-unit slot")
    ax.set_ylabel("layer")
    fig.tight_layout()
    return fig


def plot_probe_trajectory(
    rows: Mapping[int, Mapping[str, Any]],
    layer: int,
    level: int,
    slot: int,
    offset: int,
    metric: str = "acc",
    split: str = "val",
) -> plt.Figure:
    """Plot one metric across all steps for a selected residual-stream probe."""
    if metric not in {"acc", "excess_nll"}:
        raise ValueError("metric must be 'acc' or 'excess_nll'")
    key = probe_metric_key(layer, level, slot, offset, metric, split)
    steps: List[int] = []
    values: List[float] = []
    for step, row in sorted(rows.items()):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        steps.append(int(step))
        values.append(float(value))

    if not steps:
        raise ValueError(
            f"no probe {metric} values for "
            f"L{layer}/level{level}/slot{slot}/k={offset:+d}"
        )

    color = "tab:blue" if metric == "acc" else "tab:orange"
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        steps, values, marker="o", markersize=4, linewidth=2, color=color
    )
    if metric == "acc":
        axis.set_ylabel("probe accuracy")
        axis.set_ylim(-0.02, 1.02)
        metric_label = "Accuracy"
    else:
        axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
        axis.set_ylabel("excess NLL (lower is better)")
        metric_label = "Excess NLL"
    axis.set_xlabel("training step")
    axis.grid(alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    offset_label = f"k={offset:+d}" if offset else "k=0"
    axis.set_title(
        f"{metric_label} — L{layer}, level{level}, slot{slot}, {offset_label}"
    )
    fig.tight_layout()
    return fig


_PROBE_DIMENSIONS = ("layer", "level", "slot", "offset")


def _probe_dimension_values(
    dimension: str,
    num_layers: int,
    slots_per_level: Sequence[int],
    offsets: Sequence[int],
) -> List[int]:
    if dimension == "layer":
        return list(range(num_layers))
    if dimension == "level":
        return list(range(len(slots_per_level)))
    if dimension == "slot":
        return list(range(max(slots_per_level)))
    if dimension == "offset":
        return [int(offset) for offset in offsets]
    raise ValueError(f"unknown probe dimension: {dimension!r}")


def _probe_dimension_label(dimension: str, value: int) -> str:
    if dimension == "layer":
        return f"L{value}"
    if dimension == "level":
        return f"level{value}"
    if dimension == "slot":
        return f"slot{value}"
    return f"k={value:+d}" if value else "k=0"


def probe_slice_from_metrics(
    metrics: Mapping[str, Any],
    x_dimension: str,
    y_dimension: str,
    coordinates: Mapping[str, int],
    metric: str,
    split: str,
    num_layers: int,
    slots_per_level: Sequence[int],
    offsets: Union[Sequence[int], Sequence[Sequence[int]]],
) -> tuple[np.ndarray, List[str], List[str]]:
    """Build a 2-D probe slice while fixing the two remaining coordinates."""
    if x_dimension == y_dimension:
        raise ValueError("probe slice axes must be different")
    if metric not in {"acc", "excess_nll"}:
        raise ValueError("metric must be 'acc' or 'excess_nll'")
    if not slots_per_level:
        raise ValueError("probe slice requires at least one teacher level")

    offsets_by_level = normalize_offsets_by_level(offsets, len(slots_per_level))
    level_is_axis = "level" in {x_dimension, y_dimension}
    offset_values = (
        all_probe_offsets(offsets_by_level)
        if level_is_axis
        else offsets_by_level[int(coordinates["level"])]
    )
    x_values = _probe_dimension_values(
        x_dimension, num_layers, slots_per_level, offset_values
    )
    y_values = _probe_dimension_values(
        y_dimension, num_layers, slots_per_level, offset_values
    )
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float32)

    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            point = {dimension: int(coordinates[dimension])
                     for dimension in _PROBE_DIMENSIONS}
            point[x_dimension] = x_value
            point[y_dimension] = y_value
            level = point["level"]
            slot = point["slot"]
            if level < 0 or level >= len(slots_per_level):
                continue
            if slot < 0 or slot >= int(slots_per_level[level]):
                continue
            if point["offset"] not in offsets_by_level[level]:
                continue
            key = probe_metric_key(
                point["layer"], level, slot, point["offset"], metric, split
            )
            value = metrics.get(key)
            if value is not None and not pd.isna(value):
                matrix[y_index, x_index] = float(value)

    x_labels = [_probe_dimension_label(x_dimension, value) for value in x_values]
    y_labels = [_probe_dimension_label(y_dimension, value) for value in y_values]
    return matrix, x_labels, y_labels


def _probe_color_scale(
    metric: str,
    values: Sequence[np.ndarray],
) -> tuple[float, float, str, str]:
    if metric == "acc":
        return 0.0, 1.0, "viridis", "Accuracy"
    if metric != "excess_nll":
        raise ValueError("metric must be 'acc' or 'excess_nll'")
    finite = [array[np.isfinite(array)] for array in values]
    finite = [array for array in finite if array.size]
    vmax = max((float(array.max()) for array in finite), default=1e-6)
    return 0.0, max(vmax, 1e-6), "magma", "Excess NLL"


def _draw_probe_matrix(
    axis: plt.Axes,
    matrix: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    vmin: float,
    vmax: float,
    cmap: str,
) -> Any:
    image = axis.imshow(
        np.ma.masked_invalid(matrix), aspect="auto", interpolation="none",
        vmin=vmin, vmax=vmax, cmap=cmap,
    )
    x_indices = _tick_indices(len(x_labels), 10)
    y_indices = _tick_indices(len(y_labels), 10)
    axis.set_xticks(
        x_indices, [x_labels[index] for index in x_indices],
        rotation=45, ha="right",
    )
    axis.set_yticks(y_indices, [y_labels[index] for index in y_indices])
    if matrix.size <= 100:
        for row, column in zip(*np.where(np.isfinite(matrix))):
            axis.text(
                column, row, f"{matrix[row, column]:.2f}",
                ha="center", va="center", fontsize=8,
            )
    return image


def plot_probe_slice(
    metrics: Mapping[str, Any],
    step: int,
    x_dimension: str,
    y_dimension: str,
    coordinates: Mapping[str, int],
    metric: str,
    split: str,
    num_layers: int,
    slots_per_level: Sequence[int],
    offsets: Union[Sequence[int], Sequence[Sequence[int]]],
) -> plt.Figure:
    """Plot a selectable two-dimensional slice of one probe snapshot."""
    matrix, x_labels, y_labels = probe_slice_from_metrics(
        metrics, x_dimension, y_dimension, coordinates, metric, split,
        num_layers, slots_per_level, offsets,
    )
    if np.isnan(matrix).all():
        raise ValueError("this probe slice has no logged values")
    vmin, vmax, cmap, metric_label = _probe_color_scale(metric, [matrix])
    figure, axis = plt.subplots(
        figsize=(max(5.0, 0.8 * len(x_labels) + 2.0),
                 max(4.0, 0.6 * len(y_labels) + 1.8))
    )
    image = _draw_probe_matrix(
        axis, matrix, x_labels, y_labels, vmin, vmax, cmap
    )
    fixed = [
        _probe_dimension_label(dimension, int(coordinates[dimension]))
        for dimension in _PROBE_DIMENSIONS
        if dimension not in {x_dimension, y_dimension}
    ]
    axis.set_title(
        f"{metric_label} at step {step} — " + ", ".join(fixed)
    )
    axis.set_xlabel(x_dimension)
    axis.set_ylabel(y_dimension)
    figure.colorbar(image, ax=axis, label=metric_label)
    figure.tight_layout()
    return figure


def plot_probe_overview(
    metrics: Mapping[str, Any],
    step: int,
    metric: str,
    split: str,
    num_layers: int,
    slots_per_level: Sequence[int],
    offsets: Union[Sequence[int], Sequence[Sequence[int]]],
) -> plt.Figure:
    """Plot all probes as level×slot facets with offsets on each x-axis.

    Putting offsets inside each heatmap keeps automatic fine-level ranges from
    creating dozens of subplot columns. Tick thinning remains local to each
    level's actual range.
    """
    offsets_by_level = normalize_offsets_by_level(offsets, len(slots_per_level))
    matrices = {
        (level, slot): _probe_offset_matrix_from_metrics(
            metrics, level, slot, offsets_by_level[level], metric, split,
            num_layers,
        )
        for level in range(len(slots_per_level))
        for slot in range(int(slots_per_level[level]))
    }
    if not matrices or all(np.isnan(matrix).all() for matrix in matrices.values()):
        raise ValueError("this probe overview has no logged values")
    vmin, vmax, cmap, metric_label = _probe_color_scale(
        metric, list(matrices.values())
    )
    num_rows = len(slots_per_level)
    num_columns = max(int(count) for count in slots_per_level)
    figure, axes = plt.subplots(
        num_rows, num_columns,
        figsize=(max(5.0, 4.0 * num_columns), max(3.5, 2.5 * num_rows)),
        squeeze=False, constrained_layout=True,
    )
    image = None
    for level in range(num_rows):
        for slot in range(num_columns):
            axis = axes[level, slot]
            if slot >= int(slots_per_level[level]):
                axis.set_visible(False)
                continue
            matrix = matrices[(level, slot)]
            x_labels = [
                _probe_dimension_label("offset", int(offset))
                for offset in offsets_by_level[level]
            ]
            y_labels = [f"L{layer}" for layer in range(matrix.shape[0])]
            image = _draw_probe_matrix(
                axis, matrix, x_labels, y_labels, vmin, vmax, cmap
            )
            axis.set_title(f"level{level}, slot{slot}", fontsize=10)
            if level == num_rows - 1:
                axis.set_xlabel("offset")
            if slot == 0:
                axis.set_ylabel("layer")
            if np.isnan(matrix).all():
                axis.text(
                    0.5, 0.5, "No values", transform=axis.transAxes,
                    ha="center", va="center", color="0.4",
                )
    figure.suptitle(f"All probe {metric_label.lower()} at step {step}")
    figure.colorbar(
        image, ax=axes.ravel().tolist(), shrink=0.8, label=metric_label
    )
    return figure


def save_figures(
    figures: Mapping[str, plt.Figure],
    output_dir: Path,
    prefix: str,
    extension: str = "png",
) -> None:
    """Save a named figure collection to the gitignored figures directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        figure.savefig(
            output_dir / f"{prefix}_{name}.{extension}", bbox_inches="tight", dpi=200
        )

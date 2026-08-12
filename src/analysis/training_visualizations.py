"""Pure post-training plots for numerical attention and probe snapshots."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.visualizer import compute_attention_alignment


@dataclass(frozen=True)
class ProbeLayout:
    """Probe grid dimensions and configured relative offsets."""

    num_layers: int
    slots_per_level: List[int]
    offsets: List[int]


def probe_layout_from_config(config: Mapping[str, Any]) -> ProbeLayout:
    """Infer the logger's probe layout from a resolved run configuration."""
    teacher = config["teacher"]
    student = config["student"]
    if "levels" in teacher:
        slots = [int(level["chunk_size"]) for level in teacher["levels"]]
    elif "chunk_size" in teacher:
        slots = [int(teacher["chunk_size"])]
    else:
        slots = []

    probe = config.get("misc", {}).get("probe", {})
    configured_offsets = probe.get("offsets")
    if configured_offsets is not None:
        offsets = [int(offset) for offset in configured_offsets]
    else:
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
        offsets = list(range(-int(burn_in), 2))

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
        annot=True,
        fmt=".2f",
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


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
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
        )
        ax.set_title(f"{label} ({split}, step {step})")
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        ax.tick_params(axis="x", rotation=45)
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
        annot=True,
        fmt=".2f",
        xticklabels=[f"s{s}" for s in range(n_slots)],
        yticklabels=[f"L{layer}" for layer in range(n_layers)],
        cbar=True,
        ax=ax,
    )
    ax.set_title(
        f"{label} — level{level}, {split}, k={offset:+d}, step {step}"
    )
    ax.set_xlabel("within-unit slot")
    ax.set_ylabel("layer")
    fig.tight_layout()
    return fig


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

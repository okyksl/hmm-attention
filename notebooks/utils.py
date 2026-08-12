"""Unified utilities for W&B data fetching and plotting."""

import json
import warnings

warnings.filterwarnings(
    "ignore",
    category=Warning,
    module="pydantic.*",
)

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Sequence, Optional, Union
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# Must match conf/misc/default.yaml — a wrong target returns an empty frame with
# no error, which reads as "no runs matched" rather than "wrong project".
DEFAULT_ENTITY = "okyksl"
DEFAULT_PROJECT = "hmm-attention"

REPORT_BLUE = "#1f77b4"
REPORT_YELLOW = "#ff7f0e"
REPORT_GREEN = "#2ca02c"
REPORT_RED = "#d62728"
REPORT_PURPLE = "#9467bd"

REPORT_HEAD_COLORS = [REPORT_BLUE, REPORT_YELLOW, REPORT_GREEN]
REPORT_HEAD_COLORS_WITH_EXTRAS = REPORT_HEAD_COLORS + [REPORT_PURPLE]
REPORT_LINEWIDTH = 3
REPORT_AXIS_LABEL_SIZE = 24
REPORT_TICK_LABEL_SIZE = 18
REPORT_LEGEND_SIZE = 18
REPORT_TITLE_SIZE = 20


def report_head_color_map(head_order: Sequence[int]) -> Dict[int, str]:
    """Map raw W&B head indices into the report's blue, yellow, green order."""
    if len(head_order) > len(REPORT_HEAD_COLORS_WITH_EXTRAS):
        raise ValueError(
            f"Only {len(REPORT_HEAD_COLORS_WITH_EXTRAS)} report colors are defined, "
            f"got {len(head_order)} heads."
        )
    return {
        head_idx: REPORT_HEAD_COLORS_WITH_EXTRAS[color_idx]
        for color_idx, head_idx in enumerate(head_order)
    }


def style_report_axis(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = False,
    legend: bool = False,
    legend_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Apply the report's clean Matplotlib style to a single axis."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=REPORT_AXIS_LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=REPORT_AXIS_LABEL_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=REPORT_TITLE_SIZE)
    ax.grid(grid)
    ax.tick_params(labelsize=REPORT_TICK_LABEL_SIZE)
    if legend:
        kwargs = {"fontsize": REPORT_LEGEND_SIZE, "frameon": True}
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)


def _attention_artifact_path(artifact_path_or_run_id: str) -> str:
    """Accept either a W&B attention artifact path or a bare run id."""
    if artifact_path_or_run_id.startswith("run-"):
        return artifact_path_or_run_id
    return f"run-{artifact_path_or_run_id}-val_attention_weights"


def _step_shift(shift_steps: Union[bool, int]) -> int:
    """Keep old boolean behavior while supporting explicit integer shifts."""
    if isinstance(shift_steps, bool):
        return 2000 if shift_steps else 0
    return int(shift_steps)


def _save_or_show(fig: plt.Figure, save_name: Optional[str]) -> None:
    if not save_name:
        plt.show()
        return

    save_path = Path(save_name)
    if save_path.suffix != ".pdf":
        save_path = save_path.with_suffix(".pdf")
    if save_path.parent == Path("."):
        save_path = Path("figures") / save_path

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to {save_path}")


def _unwrap_wandb_value(node: Any) -> Any:
    if isinstance(node, dict):
        if set(node) == {"value"}:
            return _unwrap_wandb_value(node["value"])
        return {
            key: _unwrap_wandb_value(value)
            for key, value in node.items()
            if key != "_wandb"
        }
    if isinstance(node, list):
        return [_unwrap_wandb_value(value) for value in node]
    return node


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Normalize W&B config shapes from api.run and api.runs."""
    if isinstance(config, str):
        config = json.loads(config)
    elif not isinstance(config, dict):
        config = dict(config)
    return _unwrap_wandb_value(config)


# ============================================================================
# W&B Data Fetching
# ============================================================================

def fetch_runs(
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    tags_any: Optional[Sequence[str]] = None,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> List[wandb.apis.public.Run]:
    """Fetch W&B runs matching filters."""
    api = wandb.Api()
    filters: Dict[str, Any] = {}
    if tags_any:
        filters["tags"] = {"$in": list(tags_any)}
    if extra_filters:
        filters.update(extra_filters)
    return list(api.runs(f"{entity}/{project}", filters=filters))


def fetch_run_data(
    run_id: str,
    metrics: List[str],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> Dict[str, Any]:
    """Fetch data for a specific run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    all_metrics = list(set(metrics + ["_step"]))
    history_df = pd.DataFrame(run.scan_history(keys=all_metrics))
    return {
        "df": history_df,
        "config": _config_to_dict(run.config),
        "name": run.name,
        "id": run.id,
    }


def get_runs_data(
    runs: Sequence[wandb.apis.public.Run],
    metrics: Sequence[str],
    step_key: str = "_step",
    include_config: bool = True,
    config_sep: str = ".",
    config_prefix: str = "cfg",
) -> pd.DataFrame:
    """Aggregate history and config from multiple runs."""
    dfs: List[pd.DataFrame] = []

    for r in runs:
        h = pd.DataFrame(r.scan_history(keys=list(metrics) + [step_key]))
        if h.empty:
            continue

        meta = {"_run_id": r.id, "_run_name": r.name}

        if include_config:
            flat_cfg = pd.json_normalize(_config_to_dict(r.config), sep=config_sep)
            flat_cfg = flat_cfg.add_prefix(f"{config_prefix}{config_sep}")
            meta.update(flat_cfg.to_dict(orient="records")[0])

        meta_block = pd.DataFrame([meta] * len(h)).reset_index(drop=True)
        h = pd.concat([h.reset_index(drop=True), meta_block], axis=1)
        dfs.append(h)

    if not dfs:
        return pd.DataFrame()

    cleaned_dfs = [df.dropna(axis=1, how="all") for df in dfs]
    return pd.concat(cleaned_dfs, ignore_index=True, sort=False)


def differing_config(
    df: pd.DataFrame,
    run_id_col: str = "_run_id",
    run_name_col: str = "_run_name",
) -> pd.DataFrame:
    """Return one row per run with only config columns that differ across runs."""
    if df.empty:
        return pd.DataFrame()

    cfg_cols = [c for c in df.columns if c.startswith("cfg.")]
    if not cfg_cols:
        return pd.DataFrame()

    id_cols = [run_id_col, run_name_col]
    per_run = df.groupby(id_cols, dropna=False)[cfg_cols].first().reset_index()

    comparable_cfg = per_run[cfg_cols].apply(
        lambda col: col.map(
            lambda value: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
    )
    varying_cols = comparable_cfg.nunique(dropna=False)
    varying_cols = list(varying_cols.index[varying_cols > 1])

    return per_run[id_cols + varying_cols] if varying_cols else per_run[id_cols]


def get_table(
    artifact_path: str,
    step: int,
    split: str = "val",
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> pd.DataFrame:
    """Get attention table from W&B artifact."""
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_path}:v{step}")
    table_key = f"{split}_attention_weights"
    table = artifact.get(table_key)

    if table is None:
        raise ValueError(f"No table '{table_key}' in artifact {artifact_path}:v{step}")

    return pd.DataFrame(data=table.data, columns=table.columns)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_combined_heads(
    artifact_path: str,
    steps: Union[int, Sequence[int]],
    frequency: int = 1,
    split: str = "val",
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    head_colors: Optional[Dict[int, str]] = None,
    staircases: Optional[Dict[int, List[int]]] = None,
) -> None:
    """Plot combined attention heads with color overlays."""
    if isinstance(steps, int):
        steps = [steps]
    steps = list(steps)
    artifact_path = _attention_artifact_path(artifact_path)

    n_cols = len(steps)
    fig, axes = plt.subplots(1, n_cols, figsize=(12 * n_cols, 12), sharey=False)
    if n_cols == 1:
        axes = [axes]

    colors = [REPORT_GREEN, REPORT_YELLOW, REPORT_BLUE]

    for col_idx, display_step in enumerate(steps):
        artifact_step = display_step // frequency
        ax = axes[col_idx]
        df = get_table(artifact_path, artifact_step, split=split, entity=entity, project=project)

        head_indices = sorted(df["head"].unique())
        colors_map = head_colors or {h: colors[i % len(colors)] for i, h in enumerate(head_indices)}

        first_head_df = df[df["head"] == head_indices[0]]
        query_indices = sorted(first_head_df["query_idx"].unique())
        key_indices = sorted(first_head_df["key_idx"].unique())

        combined_rgb = np.ones((len(query_indices), len(key_indices), 3))

        for head_idx in head_indices:
            df_head = df[df["head"] == head_idx]
            attn = df_head.pivot(index="query_idx", columns="key_idx", values="weight")
            color_rgb = np.array(plt.cm.colors.to_rgb(colors_map[head_idx]))

            for i, query_idx in enumerate(query_indices):
                for j, key_idx in enumerate(key_indices):
                    weight = attn.loc[query_idx, key_idx]
                    combined_rgb[i, j] = combined_rgb[i, j] * (1 - weight) + color_rgb * weight

        ax.imshow(combined_rgb, aspect="equal", interpolation="nearest", origin="upper")

        if staircases is not None:
            offsets_for_step = staircases.get(col_idx)
            if offsets_for_step is not None:
                nq = len(query_indices)
                nk = len(key_indices)
                x_edges = np.arange(-0.5, nk + 0.5, 1)
                for off in offsets_for_step:
                    y_centers = np.clip(np.arange(nk) - off, 0, nq)
                    y_post = np.r_[y_centers, y_centers[-1]] - 0.5
                    ax.step(x_edges, y_post, where="post", color="black", linewidth=3)

        ax.set_xticks(range(len(key_indices)))
        ax.set_xticklabels(key_indices, fontsize=32, rotation=90)
        ax.set_yticks(range(len(query_indices)))
        ax.set_yticklabels(query_indices, fontsize=32)

        if col_idx == 0:
            ax.set_ylabel("Query Positions", fontsize=55)
        ax.set_xlabel("Key Positions", fontsize=55)
        ax.set_title(f"$\\mathbf{{Step~{display_step}}}$", fontsize=70, pad=20)

    plt.tight_layout()
    _save_or_show(fig, save_name)


def plot_separated_heads(
    artifact_path: str,
    steps: Union[int, Sequence[int]],
    frequency: int = 1,
    split: str = "val",
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    head_colors: Optional[Dict[int, str]] = None,
    staircases: Optional[Dict[tuple, List[int]]] = None,
) -> None:
    """Plot one attention heatmap per head and training step."""
    if isinstance(steps, int):
        steps = [steps]
    steps = list(steps)
    artifact_path = _attention_artifact_path(artifact_path)

    first_step = steps[0] // frequency
    first_df = get_table(artifact_path, first_step, split=split, entity=entity, project=project)
    head_indices = sorted(first_df["head"].unique())

    n_rows = len(steps)
    n_cols = len(head_indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 8 * n_rows),
        sharey=False,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    default_colors = [REPORT_GREEN, REPORT_YELLOW, REPORT_BLUE, REPORT_PURPLE]
    colors_map = head_colors or {
        head: default_colors[i % len(default_colors)] for i, head in enumerate(head_indices)
    }

    for row_idx, display_step in enumerate(steps):
        artifact_step = display_step // frequency
        df = get_table(artifact_path, artifact_step, split=split, entity=entity, project=project)

        first_head_df = df[df["head"] == head_indices[0]]
        query_indices = sorted(first_head_df["query_idx"].unique())
        key_indices = sorted(first_head_df["key_idx"].unique())

        for col_idx, head_idx in enumerate(head_indices):
            ax = axes[row_idx][col_idx]
            df_head = df[df["head"] == head_idx]
            attn = df_head.pivot(index="query_idx", columns="key_idx", values="weight")

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "custom_head_cmap",
                ["white", colors_map[head_idx]],
            )
            ax.imshow(
                attn.values,
                cmap=cmap,
                aspect="equal",
                interpolation="nearest",
                origin="upper",
                vmin=0.0,
                vmax=1.0,
            )

            if staircases is not None:
                offsets_for_subplot = staircases.get((row_idx, col_idx))
                if offsets_for_subplot is not None:
                    nq = len(query_indices)
                    nk = len(key_indices)
                    x_edges = np.arange(-0.5, nk + 0.5, 1)
                    for off in offsets_for_subplot:
                        y_centers = np.clip(np.arange(nk) - off, 0, nq)
                        y_post = np.r_[y_centers, y_centers[-1]] - 0.5
                        ax.step(x_edges, y_post, where="post", color="black", linewidth=3)

            ax.set_xticks(range(len(key_indices)))
            ax.set_xticklabels(key_indices, fontsize=24, rotation=90)
            ax.set_yticks(range(len(query_indices)))
            ax.set_yticklabels(query_indices, fontsize=24)

            if col_idx == 0:
                ax.set_ylabel("Query Positions", fontsize=28, labelpad=15)
                ax.text(
                    -0.25,
                    0.5,
                    f"$\\mathbf{{Step~{display_step}}}$",
                    transform=ax.transAxes,
                    fontsize=32,
                    va="center",
                    ha="right",
                    rotation=90,
                )
            else:
                ax.set_ylabel("")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Key Positions", fontsize=28, labelpad=10)
            else:
                ax.set_xlabel("")

            if row_idx == 0:
                ax.set_title(f"$\\mathbf{{Head~{head_idx + 1}}}$", fontsize=32, pad=15)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.08, wspace=0.2, hspace=0.25)
    _save_or_show(fig, save_name)


def plot_combined_heads_individual(
    artifact_path: str,
    step: int,
    split: str = "val",
    gamma: float = 1.0,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot combined attention heads with screen blending (individual step)."""
    artifact_path = _attention_artifact_path(artifact_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    df = get_table(artifact_path, step, split=split, entity=entity, project=project)
    colors = (REPORT_GREEN, REPORT_BLUE, REPORT_YELLOW)

    head_indices = sorted(df["head"].unique())
    head_colors = {
        h: np.array(mcolors.to_rgb(colors[i % len(colors)]), dtype=float)
        for i, h in enumerate(head_indices)
    }

    query_indices = sorted(df["query_idx"].unique())
    key_indices = sorted(df["key_idx"].unique())
    nq, nk = len(query_indices), len(key_indices)

    prod_color = np.ones((nq, nk, 3), dtype=float)
    prod_alpha = np.ones((nq, nk), dtype=float)

    for h in head_indices:
        df_h = df[df["head"] == h]
        A = (
            df_h.pivot(index="query_idx", columns="key_idx", values="weight")
            .reindex(index=query_indices, columns=key_indices)
            .fillna(0.0)
            .to_numpy()
            .astype(float)
        )

        if gamma != 1.0:
            A = np.power(A, gamma)

        W = np.clip(A, 0.0, 1.0)
        C = head_colors[h]

        prod_color *= 1.0 - W[..., None] * C[None, None, :]
        prod_alpha *= 1.0 - W

    S = 1.0 - prod_color
    t = 1.0 - prod_alpha
    combined_rgb = (1.0 - t)[..., None] + S
    combined_rgb = np.clip(combined_rgb, 0.0, 1.0)

    ax.imshow(combined_rgb, aspect="equal", interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    _save_or_show(fig, save_name)


def plot_kl_divergence_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    learnable: bool = False,
    shift_steps: Union[bool, int] = True,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot KL divergence over training steps."""
    if learnable:
        metrics = [
            "kl_div_unigram_learned_val",
            "kl_div_bigram_learned_val",
            "kl_div_teacher_val",
        ]
        kl_metrics = {
            "kl_div_unigram_learned_val": "4-gram",
            "kl_div_bigram_learned_val": "8-gram",
            "kl_div_teacher_val": "12-gram",
        }
    else:
        metrics = [
            "kl_div_prefix_1_teacher_val",
            "kl_div_prefix_2_teacher_val",
            "kl_div_prefix_3_teacher_val",
        ]
        kl_metrics = {
            "kl_div_prefix_1_teacher_val": r"${A^{*}_{1}}$",
            "kl_div_prefix_2_teacher_val": r"${A^{*}_{1:2}}$",
            "kl_div_prefix_3_teacher_val": r"${A^{*}_{1:3}}$",
        }

    data = fetch_run_data(run_id, metrics, entity=entity, project=project)
    df = data["df"]

    if df.empty:
        print(f"No data found for run {run_id}")
        return

    plot_df = df if max_steps is None else df[df["_step"] <= max_steps]
    plot_df = plot_df.copy()
    step_shift = _step_shift(shift_steps)
    if step_shift:
        plot_df["_step"] = plot_df["_step"] - step_shift

    plt.figure(figsize=figsize)
    x_min, x_max = plot_df["_step"].min(), plot_df["_step"].max()
    plt.margins(x=0)
    plt.xlim(x_min, x_max)

    if divergence_steps and len(divergence_steps) >= 2:
        strategy_colors = REPORT_HEAD_COLORS
        adjusted_steps = [s - step_shift for s in divergence_steps] if step_shift else divergence_steps

        x_range = x_max - x_min
        x_min_ext = x_min - 0.01 * x_range
        x_max_ext = x_max + 0.01 * x_range

        plt.axvspan(x_min_ext, adjusted_steps[0], alpha=0.2, color=strategy_colors[0])
        plt.axvspan(adjusted_steps[0], adjusted_steps[1], alpha=0.2, color=strategy_colors[1])
        plt.axvspan(adjusted_steps[1], x_max_ext, alpha=0.2, color=strategy_colors[2])

    for metric, label in kl_metrics.items():
        if metric in df.columns:
            plt.plot(plot_df["_step"], plot_df[metric], label=label, linewidth=4)

    ax = plt.gca()
    style_report_axis(ax, xlabel="Training Step", ylabel="KL Divergence")
    ax.legend(fontsize=REPORT_LEGEND_SIZE, loc="upper right", framealpha=1)
    plt.tight_layout()

    _save_or_show(plt.gcf(), save_name)


def plot_val_loss_simple(
    run_id: str,
    divergence_steps: Optional[List[int]] = None,
    max_steps: Optional[int] = None,
    figsize: tuple = (12, 8),
    shift_steps: Union[bool, int] = True,
    save_name: Optional[str] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
) -> None:
    """Plot validation loss over training steps."""
    data = fetch_run_data(run_id, ["val_loss"], entity=entity, project=project)
    df = data["df"]
    plot_df = df if max_steps is None else df[df["_step"] <= max_steps]
    plot_df = plot_df.copy()
    step_shift = _step_shift(shift_steps)
    if step_shift:
        plot_df["_step"] = plot_df["_step"] - step_shift

    plt.figure(figsize=figsize)
    plt.margins(x=0)

    if divergence_steps and len(divergence_steps) >= 2:
        strategy_colors = REPORT_HEAD_COLORS
        adjusted_steps = [s - step_shift for s in divergence_steps] if step_shift else divergence_steps

        x_min, x_max = plot_df["_step"].min(), plot_df["_step"].max()
        x_range = x_max - x_min
        x_min_ext = x_min - 0.01 * x_range
        x_max_ext = x_max + 0.01 * x_range

        plt.axvspan(x_min_ext, adjusted_steps[0], alpha=0.2, color=strategy_colors[0])
        plt.axvspan(adjusted_steps[0], adjusted_steps[1], alpha=0.2, color=strategy_colors[1])
        plt.axvspan(adjusted_steps[1], x_max_ext, alpha=0.2, color=strategy_colors[2])

    if "val_loss" in plot_df.columns:
        plt.plot(plot_df["_step"], plot_df["val_loss"], linewidth=2, color="black")

    ax = plt.gca()
    style_report_axis(ax, xlabel="Training Step", ylabel="Loss")

    if divergence_steps and len(divergence_steps) >= 2:
        strategy_labels = [r"$A^*_1$", r"$A^*_{1:2}$", r"$A^*_{1:3}$"]
        legend_patches = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.2) for c in strategy_colors]
        ax.legend(
            legend_patches,
            strategy_labels,
            fontsize=REPORT_LEGEND_SIZE,
            loc="upper right",
            framealpha=1,
        )

    plt.tight_layout()

    _save_or_show(plt.gcf(), save_name)

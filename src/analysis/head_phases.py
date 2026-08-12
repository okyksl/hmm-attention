"""Automated read-out of *which head learned which span, and when*.

The training loop already logs, per step, the attention mass and cosine
similarity between every (head, ground-truth span) pair — see
`log_attention_span_mass` / `log_attention_alignment` in `src/visualizer.py`.
That is enough to reconstruct the specialization story automatically:

    H2:off-1@120 -> H1:off-2@480 -> H3:off-3@1500

This module turns those scalar series into

  * a one-to-one **head -> span assignment** (which head owns which position),
  * an **acquisition step** per span (when that head locked on),
  * the resulting **learning order**,

and compares the observed order against the order **predicted by the config**:
the teacher's outer `lag_spectrum` says how important each span is, and the
hypothesis under test is that heads acquire spans in decreasing importance.

Typical use:

    from notebooks.utils import fetch_runs, get_runs_data
    from src.analysis.head_phases import metric_keys, phase_table

    runs = fetch_runs(tags_any=["power-spectrum-grid"])
    df = get_runs_data(runs, metric_keys(num_heads=3, num_spans=3))
    phase_table(df, num_heads=3, num_spans=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPAN_MASS = "span_mass"
COS_SIM = "align_cos_sim"


# --- metric keys --------------------------------------------------------------


def metric_key(
    head: int, span: int, layer: str = "L1", split: str = "train", metric: str = SPAN_MASS
) -> str:
    """The wandb key for one (head, span) series. `head`/`span` are 1-indexed."""
    return f"attn/{layer}/{metric}_head{head}_span{span}/{split}"


def metric_keys(
    num_heads: int,
    num_spans: int,
    layer: str = "L1",
    split: str = "train",
    metric: str = SPAN_MASS,
) -> List[str]:
    """All (head, span) keys, in row-major head-then-span order."""
    return [
        metric_key(h, k, layer, split, metric)
        for h in range(1, num_heads + 1)
        for k in range(1, num_spans + 1)
    ]


def span_offsets(
    span_lengths: Sequence[int], stride: Optional[int] = None
) -> List[str]:
    """Human labels for each span as an offset range behind the query token.

    With `span_lengths=[1,1,1]` this is `['-3', '-2', '-1']`: span 1 is the
    oldest position, span `W` the most recent.
    """
    if stride is not None:
        context_length = (len(span_lengths) - 1) * stride + span_lengths[-1]
    else:
        context_length = sum(span_lengths)

    labels = []
    for k, span_len in enumerate(span_lengths):
        start = k * stride if stride is not None else sum(span_lengths[:k])
        first = -(context_length - start)  # offset of the span's first position
        last = first + span_len - 1
        labels.append(str(first) if span_len == 1 else f"{first}..{last}")
    return labels


# --- per-run extraction -------------------------------------------------------


def head_span_series(
    run_df: pd.DataFrame,
    num_heads: int,
    num_spans: int,
    layer: str = "L1",
    split: str = "train",
    metric: str = SPAN_MASS,
    step_key: str = "_step",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract `(steps, values)` with `values` of shape `(T, num_heads, num_spans)`.

    Rows where every (head, span) entry is NaN are dropped — attention is logged
    on its own `log_attention_frequency` cadence, so most rows of a merged
    history frame carry no attention data.
    """
    cols = metric_keys(num_heads, num_spans, layer, split, metric)
    missing = [c for c in cols if c not in run_df.columns]
    if missing:
        raise KeyError(
            f"missing {len(missing)}/{len(cols)} attention keys, e.g. {missing[0]!r}. "
            "Was the run logged with a TransformerDecoder student and "
            "misc.log_attention_frequency set?"
        )

    sub = run_df[[step_key] + cols].dropna(how="all", subset=cols)
    sub = sub.sort_values(step_key)
    steps = sub[step_key].to_numpy()
    values = sub[cols].to_numpy(dtype=float).reshape(len(sub), num_heads, num_spans)
    return steps, values


def assign_heads(final: np.ndarray) -> List[int]:
    """One-to-one head -> span assignment maximizing total affinity.

    `final` is `(num_heads, num_spans)`. Returns a list where entry `h` is the
    span index owned by head `h`, or `-1` if head `h` is left unassigned
    (more heads than spans). Exact for small problems, greedy beyond that.
    """
    num_heads, num_spans = final.shape
    n = min(num_heads, num_spans)

    if num_heads <= 7 and num_spans <= 7:
        best_score, best = -np.inf, None
        for spans in permutations(range(num_spans), n):
            for heads in permutations(range(num_heads), n):
                score = sum(final[h, k] for h, k in zip(heads, spans))
                if score > best_score:
                    best_score, best = score, dict(zip(heads, spans))
        return [best.get(h, -1) for h in range(num_heads)]

    # Greedy fallback: repeatedly take the largest remaining affinity.
    order = np.dstack(np.unravel_index(np.argsort(-final, axis=None), final.shape))[0]
    taken_h, taken_k, out = set(), set(), {}
    for h, k in order:
        if h in taken_h or k in taken_k:
            continue
        out[int(h)] = int(k)
        taken_h.add(h)
        taken_k.add(k)
    return [out.get(h, -1) for h in range(num_heads)]


def acquisition_step(
    steps: np.ndarray,
    series: np.ndarray,
    threshold: float = 0.5,
    relative: bool = True,
    persist: bool = True,
    min_rise: float = 0.05,
) -> Optional[float]:
    """First step at which `series` crosses its lock-on threshold.

    With `relative=True` the threshold sits `threshold` of the way from the
    series' baseline (its minimum) to its final value — measuring the *rise*,
    not the level, since attention mass and cosine similarity live on different
    scales. A head that never specializes has a flat series, which would cross
    any pure level threshold immediately; `min_rise` is the absolute rise
    required before a crossing counts at all, so those return `None`.

    With `persist=True` the series must stay above the threshold for the rest of
    training, ignoring transient spikes during the early collaborative phase.
    """
    if len(series) == 0:
        return None
    final = float(series[-1])
    if not np.isfinite(final) or final <= 0:
        return None

    if relative:
        base = float(np.min(series))
        if final - base < min_rise:
            return None  # never actually specialized
        level = base + threshold * (final - base)
    else:
        level = threshold
    if not np.isfinite(level):
        return None

    above = series >= level
    if not above.any():
        return None
    if persist:
        # Last index where it is *below*; lock-on is the step right after.
        below = np.flatnonzero(~above)
        idx = int(below[-1]) + 1 if len(below) else 0
        if idx >= len(steps):
            return None
    else:
        idx = int(np.flatnonzero(above)[0])
    return float(steps[idx])


@dataclass
class RunPhases:
    """The specialization story of a single run."""

    run_id: str
    run_name: str
    num_heads: int
    num_spans: int
    span_labels: List[str]
    assignment: List[int]  # head -> span index (-1 = unassigned)
    acquired: Dict[int, Optional[float]]  # span index -> step
    final_affinity: np.ndarray
    observed_order: List[int] = field(default_factory=list)
    predicted_order: List[int] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """e.g. `H2:-1@120 -> H1:-2@480 -> H3:-3@1500`."""
        owner = {k: h for h, k in enumerate(self.assignment) if k >= 0}
        parts = []
        for k in self.observed_order:
            step = self.acquired.get(k)
            when = "never" if step is None else f"{step:g}"
            parts.append(f"H{owner.get(k, -1) + 1}:{self.span_labels[k]}@{when}")
        return " -> ".join(parts)

    @property
    def order_matches_importance(self) -> Optional[bool]:
        if not self.predicted_order or len(self.observed_order) != len(self.predicted_order):
            return None
        return self.observed_order == self.predicted_order

    @property
    def order_rank_corr(self) -> Optional[float]:
        """Spearman correlation between predicted and observed acquisition order."""
        if not self.predicted_order or len(self.observed_order) != len(self.predicted_order):
            return None
        n = len(self.predicted_order)
        if n < 2:
            return None
        pred_rank = {k: i for i, k in enumerate(self.predicted_order)}
        obs_rank = {k: i for i, k in enumerate(self.observed_order)}
        d2 = sum((pred_rank[k] - obs_rank[k]) ** 2 for k in pred_rank)
        return 1.0 - 6.0 * d2 / (n * (n**2 - 1))


def analyze_run(
    run_df: pd.DataFrame,
    num_heads: int,
    num_spans: int,
    span_lengths: Optional[Sequence[int]] = None,
    stride: Optional[int] = None,
    layer: str = "L1",
    split: str = "train",
    metric: str = SPAN_MASS,
    threshold: float = 0.5,
    predicted_order: Optional[Sequence[int]] = None,
) -> RunPhases:
    """Reconstruct the head-specialization story for one run's history frame."""
    steps, values = head_span_series(
        run_df, num_heads, num_spans, layer=layer, split=split, metric=metric
    )
    final = values[-1]
    assignment = assign_heads(final)

    acquired: Dict[int, Optional[float]] = {}
    for h, k in enumerate(assignment):
        if k < 0:
            continue
        acquired[k] = acquisition_step(steps, values[:, h, k], threshold=threshold)

    # Spans that never locked on sort last, in final-affinity order.
    def sort_key(k: int) -> Tuple[int, float, float]:
        step = acquired.get(k)
        owner = next((h for h, kk in enumerate(assignment) if kk == k), None)
        strength = -float(final[owner, k]) if owner is not None else 0.0
        return (1, 0.0, strength) if step is None else (0, step, strength)

    observed_order = sorted(acquired.keys(), key=sort_key)

    labels = (
        span_offsets(span_lengths, stride)
        if span_lengths is not None
        else [f"span{k + 1}" for k in range(num_spans)]
    )

    return RunPhases(
        run_id=str(run_df["_run_id"].iloc[0]) if "_run_id" in run_df else "",
        run_name=str(run_df["_run_name"].iloc[0]) if "_run_name" in run_df else "",
        num_heads=num_heads,
        num_spans=num_spans,
        span_labels=labels,
        assignment=assignment,
        acquired=acquired,
        final_affinity=final,
        observed_order=observed_order,
        predicted_order=list(predicted_order) if predicted_order is not None else [],
    )


# --- config-derived prediction ------------------------------------------------


def predicted_span_order(cfg: Dict[str, Any], window: int) -> List[int]:
    """Span indices ordered by the *config's* importance, most important first.

    Reads the teacher's outer `lag_spectrum` and materializes it with
    `src.spectra`, so the prediction always tracks whatever law the sweep used
    (geometric, power, flat, reversed, ...) rather than a hard-coded guess.
    """
    from src.spectra import SpectrumSpec, spectrum

    spec_fields = {
        key: cfg[f"cfg.teacher.lag_spectrum.{key}"]
        for key in ("law", "decay", "alpha", "rank", "normalize", "reverse")
        if f"cfg.teacher.lag_spectrum.{key}" in cfg
    }
    weights = spectrum(SpectrumSpec(**spec_fields), window).numpy()
    # Stable descending sort: ties keep span order.
    return list(np.argsort(-weights, kind="stable"))


def phase_table(
    df: pd.DataFrame,
    num_heads: int,
    num_spans: int,
    span_lengths: Optional[Sequence[int]] = None,
    stride: Optional[int] = None,
    layer: str = "L1",
    split: str = "train",
    metric: str = SPAN_MASS,
    threshold: float = 0.5,
    group_cols: Sequence[str] = (
        "cfg.teacher.spectrum.alpha",
        "cfg.teacher.lag_spectrum.alpha",
    ),
    use_config_prediction: bool = True,
) -> pd.DataFrame:
    """One row per run: assignment, per-span acquisition step, and order match.

    `df` is the frame returned by `notebooks.utils.get_runs_data`, which carries
    both history and flattened config columns.
    """
    rows = []
    for run_id, run_df in df.groupby("_run_id", dropna=False):
        cfg = run_df.iloc[0].to_dict()
        predicted = None
        if use_config_prediction:
            try:
                predicted = predicted_span_order(cfg, num_spans)
            except Exception:  # config shape varies across older runs
                predicted = None

        phases = analyze_run(
            run_df,
            num_heads,
            num_spans,
            span_lengths=span_lengths,
            stride=stride,
            layer=layer,
            split=split,
            metric=metric,
            threshold=threshold,
            predicted_order=predicted,
        )

        row: Dict[str, Any] = {
            "_run_id": run_id,
            "_run_name": phases.run_name,
            "summary": phases.summary,
        }
        for col in group_cols:
            if col in cfg:
                row[col] = cfg[col]
        for h, k in enumerate(phases.assignment):
            row[f"head{h + 1}_span"] = phases.span_labels[k] if k >= 0 else None
        for k in range(num_spans):
            row[f"acq_{phases.span_labels[k]}"] = phases.acquired.get(k)
        row["observed_order"] = [phases.span_labels[k] for k in phases.observed_order]
        row["predicted_order"] = [phases.span_labels[k] for k in phases.predicted_order]
        row["order_matches"] = phases.order_matches_importance
        row["order_rank_corr"] = phases.order_rank_corr
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    return out.sort_values(sort_cols) if sort_cols else out

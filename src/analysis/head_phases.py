"""Automated read-out of *which head learned which span, and when*.

The training loop already logs, per step, the attention mass and cosine
similarity between every (head, ground-truth span) pair — see
`log_attention_span_mass` / `log_attention_alignment` in `src/visualizer.py`.
That is enough to reconstruct the specialization story automatically:

    H2:off-1@120 -> H1:off-2@480 -> H3:off-3@1500

This module turns those scalar series into

  * a per-head **trajectory**: the sequence of spans each head attends to over
    training, so a head that starts on the dominant position and later migrates
    to its own is described as `-1@0 -> -3@600` rather than collapsed to its
    endpoint. Migration is the phenomenon, not noise: early in training all
    heads pile onto the most important span, then break away one at a time.
  * a **coverage step** per span — when that span first acquired a stable owner,
  * the resulting **learning order**,

and compares the observed order against the order **predicted by the config**:
the teacher's outer `lag_spectrum` says how important each span is, and the
hypothesis under test is that spans are covered in decreasing importance.

Typical use:

    from notebooks.utils import fetch_runs, get_runs_data
    from src.analysis.head_phases import metric_keys, phase_table

    runs = fetch_runs(tags_any=["power-spectrum-grid"])
    df = get_runs_data(runs, metric_keys(num_heads=3, num_spans=3))
    phase_table(df, num_heads=3, num_spans=3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPAN_MASS = "span_mass"
COS_SIM = "align_cos_sim"

#: Label for a head that is not focused on any single span at a given step.
DIFFUSE = -1

#: How the per-step focus cutoff is applied. See `label_focused_span`.
FOCUS_STRATEGIES = ("share", "absolute", "margin", "ratio")


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


@dataclass(frozen=True)
class Segment:
    """A stretch of training during which one head held one span (or nothing).

    `span == DIFFUSE` means the head was not committed to any single span over
    this stretch — the early collaborative regime, or a head that never
    specializes at all.
    """

    span: int
    start_step: float
    end_step: float
    n_samples: int


def label_focused_span(
    values: np.ndarray, cutoff: float = 0.5, strategy: str = "share"
) -> np.ndarray:
    """`(T, H, K)` affinities -> `(T, H)` span labels, `DIFFUSE` where unfocused.

    Each head is judged **independently**: several heads may focus on the same
    span (the early collaborative regime, or genuine redundancy), and a span may
    end up with no head at all. There is no matching and no exclusivity.

    A head is credited with its strongest span only if that span clears
    `cutoff` under `strategy`:

    * ``share``    — the span's fraction of the head's total affinity across
      spans. Scale-free and independent of sequence length. Uniform attention
      gives every span `1/K`, so any `cutoff > 1/K` reads as unfocused.
      Good default for ``span_mass``.
    * ``absolute`` — the raw value. Natural for ``align_cos_sim``, where
      `cutoff=0.7` means "genuinely aligned with this span's shape".
    * ``margin``   — how far the best span beats the runner-up, in raw units.
      Use when what matters is *separation* rather than level.
    * ``ratio``    — best / runner-up. Like ``margin`` but scale-free;
      `cutoff=2.0` means "twice the next-best span".

    Negative affinities (possible for cosine similarity) are clipped to zero for
    the ``share`` and ``ratio`` denominators.
    """
    if strategy not in FOCUS_STRATEGIES:
        raise ValueError(f"strategy must be one of {FOCUS_STRATEGIES}; got {strategy!r}")
    if values.shape[-1] < 2 and strategy in ("margin", "ratio"):
        raise ValueError(f"strategy {strategy!r} needs at least 2 spans to compare")

    best = values.argmax(axis=-1)
    best_val = np.take_along_axis(values, best[..., None], axis=-1)[..., 0]

    if strategy == "absolute":
        score = best_val
    elif strategy == "share":
        pos = np.clip(values, 0.0, None)
        totals = pos.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(totals > 0, np.clip(best_val, 0.0, None) / totals, 0.0)
    else:  # margin / ratio, both against the runner-up
        ordered = np.sort(values, axis=-1)
        second = ordered[..., -2]
        if strategy == "margin":
            score = best_val - second
        else:
            second_pos = np.clip(second, 0.0, None)
            with np.errstate(invalid="ignore", divide="ignore"):
                score = np.where(
                    second_pos > 0,
                    np.clip(best_val, 0.0, None) / second_pos,
                    np.inf,  # runner-up at zero => unbounded dominance
                )

    score = np.nan_to_num(score, nan=0.0, posinf=np.inf)
    return np.where(score >= cutoff, best, DIFFUSE).astype(int)


def segment_labels(
    steps: np.ndarray, labels: np.ndarray, min_dwell: int = 2
) -> List[Segment]:
    """Run-length encode one head's label series into `Segment`s.

    Runs shorter than `min_dwell` samples are treated as flicker and absorbed
    into the preceding surviving run (or the following one, at the start), so a
    single noisy sample does not manufacture a phase transition.
    """
    if len(labels) == 0:
        return []

    # Raw run-length encoding.
    runs: List[List[int]] = []  # [label, start_idx, end_idx_inclusive]
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([int(lab), i, i])

    # Absorb short runs, then re-merge neighbours that became equal.
    if min_dwell > 1 and len(runs) > 1:
        kept: List[List[int]] = []
        for run in runs:
            length = run[2] - run[1] + 1
            if length < min_dwell and kept:
                kept[-1][2] = run[2]  # extend the previous surviving run
            elif length < min_dwell and not kept:
                run_next = run  # leading flicker: hand its span to the next run
                kept.append(run_next)
            else:
                kept.append(run)
        merged: List[List[int]] = []
        for run in kept:
            if merged and merged[-1][0] == run[0]:
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        runs = merged

    return [
        Segment(
            span=lab,
            start_step=float(steps[start]),
            end_step=float(steps[end]),
            n_samples=end - start + 1,
        )
        for lab, start, end in runs
    ]


def head_trajectories(
    steps: np.ndarray,
    values: np.ndarray,
    cutoff: float = 0.5,
    strategy: str = "share",
    min_dwell: int = 2,
) -> List[List[Segment]]:
    """Per-head phase sequence over the whole run: `[head][segment]`."""
    labels = label_focused_span(values, cutoff=cutoff, strategy=strategy)
    return [segment_labels(steps, labels[:, h], min_dwell) for h in range(values.shape[1])]


def span_coverage(
    trajectories: Sequence[Sequence[Segment]], num_spans: int
) -> Dict[int, Optional[float]]:
    """First step at which each span acquired a stable owner (any head).

    Unlike an endpoint-based reading, this credits the span at the moment it was
    first held — even if the head that held it later moved on and a different
    head inherited it.
    """
    first: Dict[int, Optional[float]] = {k: None for k in range(num_spans)}
    for segments in trajectories:
        for seg in segments:
            if seg.span == DIFFUSE:
                continue
            current = first.get(seg.span)
            if current is None or seg.start_step < current:
                first[seg.span] = seg.start_step
    return first


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
    assignment: List[int]  # head -> span index at the FINAL step (-1 = diffuse)
    acquired: Dict[int, Optional[float]]  # span index -> first stable coverage
    final_affinity: np.ndarray
    trajectories: List[List[Segment]] = field(default_factory=list)
    observed_order: List[int] = field(default_factory=list)
    predicted_order: List[int] = field(default_factory=list)

    def _label(self, span: int) -> str:
        return "diffuse" if span == DIFFUSE else self.span_labels[span]

    @property
    def summary(self) -> str:
        """Span coverage order, e.g. `-1@120 -> -2@480 -> -3@1500`."""
        parts = []
        for k in self.observed_order:
            step = self.acquired.get(k)
            when = "never" if step is None else f"{step:g}"
            parts.append(f"{self.span_labels[k]}@{when}")
        return " -> ".join(parts)

    def head_path(self, head: int) -> str:
        """One head's migration path, e.g. `diffuse@0 > -1@150 > -3@600`."""
        segs = self.trajectories[head] if head < len(self.trajectories) else []
        return " > ".join(f"{self._label(s.span)}@{s.start_step:g}" for s in segs)

    @property
    def head_story(self) -> str:
        """Every head's path, e.g. `H1:diffuse@0>-1@150 | H2:-1@0>-3@600`."""
        return " | ".join(
            f"H{h + 1}:{self.head_path(h)}" for h in range(len(self.trajectories))
        )

    @property
    def final_owners(self) -> Dict[int, List[int]]:
        """span -> heads focused on it at the end. May be empty, or several.

        Heads are judged independently, so this is not a permutation: a span can
        attract two heads (redundancy, or an unfinished break-away) while another
        gets none.
        """
        owners: Dict[int, List[int]] = {k: [] for k in range(self.num_spans)}
        for h, k in enumerate(self.assignment):
            if k != DIFFUSE:
                owners[k].append(h)
        return owners

    @property
    def shared_spans(self) -> int:
        """Spans held by more than one head at the end."""
        return sum(1 for heads in self.final_owners.values() if len(heads) > 1)

    @property
    def uncovered_spans(self) -> int:
        """Spans no head is focused on at the end."""
        return sum(1 for heads in self.final_owners.values() if not heads)

    @property
    def migrations(self) -> int:
        """How many times a head moved from one committed span to another.

        Zero means every head picked a span and stayed; a positive count is the
        collaborative-then-break-away dynamic actually happening.
        """
        total = 0
        for segments in self.trajectories:
            committed = [s.span for s in segments if s.span != DIFFUSE]
            total += max(0, len(committed) - 1)
        return total

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
    cutoff: float = 0.5,
    strategy: str = "share",
    min_dwell: int = 2,
    predicted_order: Optional[Sequence[int]] = None,
) -> RunPhases:
    """Reconstruct the head-specialization story for one run's history frame.

    The read-out is *trajectory-based*: each head is labelled at every logged
    step with its strongest span if that span clears `cutoff` under `strategy`
    (see `label_focused_span`), those labels are segmented into phases, and a
    span counts as acquired the first time any head holds it stably. Heads are
    judged independently, so several may share a span and some spans may go
    uncovered. A head that starts on the dominant position and later migrates
    keeps both phases in its path.
    """
    steps, values = head_span_series(
        run_df, num_heads, num_spans, layer=layer, split=split, metric=metric
    )
    final = values[-1]
    trajectories = head_trajectories(
        steps, values, cutoff=cutoff, strategy=strategy, min_dwell=min_dwell
    )

    # Final configuration: each head's label in its last segment.
    assignment = [
        segments[-1].span if segments else DIFFUSE for segments in trajectories
    ]
    acquired = span_coverage(trajectories, num_spans)

    # Spans never covered sort last, strongest final affinity first among them.
    def sort_key(k: int) -> Tuple[int, float, float]:
        step = acquired.get(k)
        strength = -float(final[:, k].max())
        return (1, 0.0, strength) if step is None else (0, step, strength)

    observed_order = sorted(range(num_spans), key=sort_key)

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
        trajectories=trajectories,
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
    cutoff: float = 0.5,
    strategy: str = "share",
    min_dwell: int = 2,
    group_cols: Sequence[str] = (
        "cfg.teacher.spectrum.alpha",
        "cfg.teacher.lag_spectrum.alpha",
    ),
    use_config_prediction: bool = True,
) -> pd.DataFrame:
    """One row per run: coverage order, per-head migration path, and order match.

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
            cutoff=cutoff,
            strategy=strategy,
            min_dwell=min_dwell,
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
            row[f"head{h + 1}_final"] = phases._label(k)
            row[f"head{h + 1}_path"] = phases.head_path(h)
        for k in range(num_spans):
            row[f"acq_{phases.span_labels[k]}"] = phases.acquired.get(k)
        row["migrations"] = phases.migrations
        row["shared_spans"] = phases.shared_spans
        row["uncovered_spans"] = phases.uncovered_spans
        row["observed_order"] = [phases.span_labels[k] for k in phases.observed_order]
        row["predicted_order"] = [phases.span_labels[k] for k in phases.predicted_order]
        row["order_matches"] = phases.order_matches_importance
        row["order_rank_corr"] = phases.order_rank_corr
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    return out.sort_values(sort_cols) if sort_cols else out

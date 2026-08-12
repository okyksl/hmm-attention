"""Unit tests for `src.analysis.head_phases` (the automated specialization read-out).

Uses synthetic history frames shaped exactly like `notebooks.utils.get_runs_data`
output, so the logic is verified without touching wandb.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.head_phases import (
    DIFFUSE,
    acquisition_step,
    analyze_run,
    head_trajectories,
    FOCUS_STRATEGIES,
    label_focused_span,
    metric_key,
    metric_keys,
    phase_table,
    predicted_span_order,
    segment_labels,
    span_coverage,
    span_offsets,
)


# ---- keys and labels ---------------------------------------------------------


def test_metric_keys_are_row_major():
    keys = metric_keys(num_heads=2, num_spans=2)
    assert keys == [
        "attn/L1/span_mass_head1_span1/train",
        "attn/L1/span_mass_head1_span2/train",
        "attn/L1/span_mass_head2_span1/train",
        "attn/L1/span_mass_head2_span2/train",
    ]


def test_span_offsets_unit_spans_are_negative_positions():
    assert span_offsets([1, 1, 1]) == ["-3", "-2", "-1"]


def test_span_offsets_multi_token_spans_are_ranges():
    assert span_offsets([2, 2, 2]) == ["-6..-5", "-4..-3", "-2..-1"]


def test_span_offsets_with_stride():
    # context = (3-1)*1 + 2 = 4; spans start at 0, 1, 2.
    assert span_offsets([2, 2, 2], stride=1) == ["-4..-3", "-3..-2", "-2..-1"]


# ---- acquisition timing ------------------------------------------------------


def test_acquisition_step_finds_the_relative_crossing():
    steps = np.array([0, 10, 20, 30, 40])
    series = np.array([0.0, 0.1, 0.2, 0.8, 1.0])
    # threshold = 0.5 * final(1.0) = 0.5; first persistent crossing is step 30.
    assert acquisition_step(steps, series) == 30


def test_acquisition_step_ignores_a_transient_spike_when_persisting():
    steps = np.array([0, 10, 20, 30, 40])
    series = np.array([0.9, 0.1, 0.2, 0.8, 1.0])
    assert acquisition_step(steps, series, persist=True) == 30
    assert acquisition_step(steps, series, persist=False) == 0


def test_acquisition_step_returns_none_when_never_learned():
    steps = np.array([0, 10, 20])
    assert acquisition_step(steps, np.array([0.0, 0.0, 0.0])) is None


def test_acquisition_step_returns_none_for_a_flat_unspecialized_series():
    # A head that never locks on sits at a constant low mass. Without a
    # minimum-rise guard this would spuriously "cross" at step 0.
    steps = np.array([0, 10, 20, 30])
    assert acquisition_step(steps, np.full(4, 0.05)) is None


def test_acquisition_step_measures_rise_not_level():
    # Series rises from a high baseline; the crossing is the rise midpoint.
    steps = np.array([0, 10, 20, 30])
    assert acquisition_step(steps, np.array([0.4, 0.4, 0.9, 1.0])) == 20


# ---- end-to-end run analysis -------------------------------------------------


def _synthetic_run(
    onsets=(200, 50, 800), run_id="r0", num_heads=3, num_spans=3, steps=None
):
    """A run where head h owns span h and locks on at `onsets[h]`."""
    steps = np.arange(0, 1000, 50) if steps is None else steps
    data = {"_step": steps, "_run_id": run_id, "_run_name": f"name-{run_id}"}
    for h in range(num_heads):
        for k in range(num_spans):
            key = metric_key(h + 1, k + 1)
            if h == k:
                data[key] = np.where(steps >= onsets[h], 0.9, 0.05)
            else:
                data[key] = np.full(len(steps), 0.05)
    return pd.DataFrame(data)


def test_analyze_run_recovers_assignment_and_order():
    # Span 1 (offset -2) is covered first, then span 0 (-3), then span 2 (-1).
    df = _synthetic_run(onsets=(200, 50, 800))
    phases = analyze_run(df, num_heads=3, num_spans=3, span_lengths=[1, 1, 1])

    assert phases.assignment == [0, 1, 2]
    assert phases.acquired == {0: 200.0, 1: 50.0, 2: 800.0}
    assert phases.observed_order == [1, 0, 2]
    assert phases.summary == "-2@50 -> -3@200 -> -1@800"


def test_analyze_run_sorts_never_learned_spans_last():
    steps = np.arange(0, 500, 50)
    df = _synthetic_run(onsets=(100, 50, 10_000), steps=steps)
    phases = analyze_run(df, num_heads=3, num_spans=3, span_lengths=[1, 1, 1])
    assert phases.observed_order[-1] == 2
    assert phases.acquired[2] is None
    assert phases.summary.endswith("-1@never")


def test_analyze_run_drops_rows_without_attention_logs():
    # Attention logs on a slower cadence than scalars: most rows are NaN.
    df = _synthetic_run()
    sparse = df.copy()
    attn_cols = [c for c in df.columns if c.startswith("attn/")]
    sparse.loc[sparse.index % 2 == 1, attn_cols] = np.nan
    phases = analyze_run(sparse, num_heads=3, num_spans=3, span_lengths=[1, 1, 1])
    assert phases.assignment == [0, 1, 2]


def test_analyze_run_raises_a_helpful_error_on_missing_keys():
    df = pd.DataFrame({"_step": [0, 1], "loss": [1.0, 0.5]})
    with pytest.raises(KeyError, match="missing 9/9 attention keys"):
        analyze_run(df, num_heads=3, num_spans=3)


# ---- config-derived prediction ----------------------------------------------


def test_predicted_span_order_from_a_reversed_power_law():
    # reverse=true puts the most weight on the last (most recent) span.
    cfg = {
        "cfg.teacher.lag_spectrum.law": "power",
        "cfg.teacher.lag_spectrum.alpha": 0.5,
        "cfg.teacher.lag_spectrum.normalize": "none",
        "cfg.teacher.lag_spectrum.reverse": True,
    }
    assert predicted_span_order(cfg, window=3) == [2, 1, 0]


def test_predicted_span_order_from_a_forward_geometric_law():
    # The historic default: decay=1.7, reverse=false -> weights grow with index.
    cfg = {
        "cfg.teacher.lag_spectrum.law": "geometric",
        "cfg.teacher.lag_spectrum.decay": 1.7,
        "cfg.teacher.lag_spectrum.normalize": "none",
        "cfg.teacher.lag_spectrum.reverse": False,
    }
    assert predicted_span_order(cfg, window=3) == [2, 1, 0]


# ---- cross-run table ---------------------------------------------------------


def _cfg_columns(df, alpha_in, alpha_out):
    df = df.copy()
    df["cfg.teacher.spectrum.alpha"] = alpha_in
    df["cfg.teacher.lag_spectrum.alpha"] = alpha_out
    df["cfg.teacher.lag_spectrum.law"] = "power"
    df["cfg.teacher.lag_spectrum.normalize"] = "none"
    df["cfg.teacher.lag_spectrum.reverse"] = True
    return df


def test_phase_table_flags_agreement_with_the_configs_importance_order():
    # Recency-ordered acquisition (-1 first, then -2, then -3) is exactly what a
    # reversed power outer law predicts.
    matching = _cfg_columns(_synthetic_run(onsets=(800, 400, 50), run_id="a"), 1.0, 0.5)
    # Reverse the acquisition order -> should be flagged as a mismatch.
    mismatch = _cfg_columns(_synthetic_run(onsets=(50, 400, 800), run_id="b"), 2.0, 1.0)

    table = phase_table(
        pd.concat([matching, mismatch], ignore_index=True),
        num_heads=3,
        num_spans=3,
        span_lengths=[1, 1, 1],
    )

    by_run = table.set_index("_run_id")
    assert by_run.loc["a", "observed_order"] == ["-1", "-2", "-3"]
    assert by_run.loc["a", "predicted_order"] == ["-1", "-2", "-3"]
    assert bool(by_run.loc["a", "order_matches"]) is True
    assert by_run.loc["a", "order_rank_corr"] == pytest.approx(1.0)

    assert by_run.loc["b", "observed_order"] == ["-3", "-2", "-1"]
    assert bool(by_run.loc["b", "order_matches"]) is False
    assert by_run.loc["b", "order_rank_corr"] == pytest.approx(-1.0)


def test_label_focused_span_share_strategy_needs_a_dominant_span():
    # head 0 is focused on span 2; head 1 is spread evenly -> DIFFUSE.
    values = np.array([[[0.05, 0.05, 0.90], [0.33, 0.33, 0.34]]])
    labels = label_focused_span(values, cutoff=0.5, strategy="share")
    assert labels.tolist() == [[2, DIFFUSE]]


def test_label_focused_span_allows_several_heads_on_one_span():
    # All three heads pile onto span 0 — the early collaborative regime. No
    # matching, no exclusivity: every head is credited with span 0.
    values = np.array([[[0.9, 0.05, 0.05]] * 3])
    labels = label_focused_span(values, cutoff=0.5, strategy="share")
    assert labels.tolist() == [[0, 0, 0]]


def test_label_focused_span_absolute_strategy_uses_raw_value():
    values = np.array([[[0.4, 0.9, 0.1]]])
    assert label_focused_span(values, cutoff=0.8, strategy="absolute").tolist() == [[1]]
    assert label_focused_span(values, cutoff=0.95, strategy="absolute").tolist() == [[DIFFUSE]]


def test_label_focused_span_margin_strategy_uses_gap_to_runner_up():
    values = np.array([[[0.5, 0.9, 0.1]]])  # gap = 0.4
    assert label_focused_span(values, cutoff=0.3, strategy="margin").tolist() == [[1]]
    assert label_focused_span(values, cutoff=0.5, strategy="margin").tolist() == [[DIFFUSE]]


def test_label_focused_span_ratio_strategy_is_scale_free():
    values = np.array([[[0.2, 0.8, 0.1]]])  # ratio = 4x
    assert label_focused_span(values, cutoff=3.0, strategy="ratio").tolist() == [[1]]
    assert label_focused_span(values, cutoff=5.0, strategy="ratio").tolist() == [[DIFFUSE]]
    # Scaling every entry leaves the verdict unchanged.
    assert label_focused_span(values * 100, cutoff=3.0, strategy="ratio").tolist() == [[1]]


def test_label_focused_span_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="strategy must be one of"):
        label_focused_span(np.zeros((1, 1, 3)), strategy="bogus")


def test_label_focused_span_handles_negative_cosine_similarities():
    # cos-sim can go negative; share must not be corrupted by a negative total.
    values = np.array([[[-0.3, 0.9, -0.2]]])
    assert label_focused_span(values, cutoff=0.5, strategy="share").tolist() == [[1]]


# ---- trajectories ------------------------------------------------------------


def _traj_values(paths, steps):
    """paths[h] = list of (span, until_step); build a (T,H,K) focus tensor."""
    T, H = len(steps), len(paths)
    values = np.full((T, H, 3), 0.05)
    for h, path in enumerate(paths):
        for t, step in enumerate(steps):
            span = next(sp for sp, until in path if step < until)
            values[t, h, span] = 0.9
    return values


def test_head_trajectories_capture_migration():
    # Head 0 starts on the dominant span 2, then breaks away to span 0.
    steps = np.arange(0, 1000, 50)
    values = _traj_values([[(2, 600), (0, np.inf)]], steps)
    (traj,) = head_trajectories(steps, values, cutoff=0.5, strategy="share")
    assert [s.span for s in traj] == [2, 0]
    assert traj[0].start_step == 0
    assert traj[1].start_step == 600


def test_head_trajectories_absorb_single_sample_flicker():
    steps = np.arange(0, 500, 50)
    values = np.full((len(steps), 1, 3), 0.05)
    values[:, 0, 2] = 0.9
    values[4, 0, 2] = 0.05          # one-sample dropout to span 0
    values[4, 0, 0] = 0.9
    (traj,) = head_trajectories(steps, values, min_dwell=2)
    assert [s.span for s in traj] == [2]   # flicker absorbed, no phantom phase


def test_span_coverage_credits_the_first_holder_not_the_last():
    # Span 2 is held by head 0 early, then head 0 leaves and head 1 takes it.
    steps = np.arange(0, 1000, 50)
    values = _traj_values(
        [[(2, 400), (0, np.inf)], [(1, 600), (2, np.inf)]], steps
    )
    trajectories = head_trajectories(steps, values)
    coverage = span_coverage(trajectories, num_spans=3)
    assert coverage[2] == 0      # credited to the first holder, not head 1 at 600
    assert coverage[0] == 400
    assert coverage[1] == 0


def test_analyze_run_reports_migrations_and_sharing():
    steps = np.arange(0, 1000, 50)
    # Heads 0 and 1 both end on span 2; head 2 migrates 2 -> 0. Span 1 uncovered.
    values = _traj_values(
        [[(2, np.inf)], [(2, np.inf)], [(2, 500), (0, np.inf)]], steps
    )
    df = _frame_from_values(steps, values)
    phases = analyze_run(df, num_heads=3, num_spans=3, span_lengths=[1, 1, 1])

    assert phases.final_owners[2] == [0, 1]   # two heads share span 2
    assert phases.final_owners[0] == [2]
    assert phases.final_owners[1] == []       # nobody covers span 1
    assert phases.shared_spans == 1
    assert phases.uncovered_spans == 1
    assert phases.migrations == 1
    assert "2:" not in phases.head_path(2) or "-3" in phases.head_path(2)


def _frame_from_values(steps, values):
    T, H, K = values.shape
    data = {"_step": steps, "_run_id": "r", "_run_name": "r"}
    for h in range(H):
        for k in range(K):
            data[metric_key(h + 1, k + 1)] = values[:, h, k]
    return pd.DataFrame(data)


def test_phase_table_reports_per_head_spans_and_acquisition_columns():
    df = _cfg_columns(_synthetic_run(onsets=(200, 50, 800)), 1.0, 0.5)
    table = phase_table(df, num_heads=3, num_spans=3, span_lengths=[1, 1, 1])
    row = table.iloc[0]
    assert row["head1_final"] == "-3"
    assert row["head2_final"] == "-2"
    assert row["acq_-2"] == 50.0
    assert row["acq_-1"] == 800.0
    assert row["cfg.teacher.spectrum.alpha"] == 1.0

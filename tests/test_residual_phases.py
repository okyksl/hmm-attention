"""Tests for depth-aware residual-stream probe learning-order analysis."""

from math import log

import numpy as np
import pandas as pd
import pytest

from src.analysis.residual_phases import (
    ProbeSpec,
    acquisition_table,
    adjacent_layer_deltas,
    layer_order_by_target,
    missing_probe_cells,
    probe_history_long,
    probe_metric_key,
    probe_metric_keys,
    probe_spec_from_config,
    sustained_acquisition_step,
    target_order_by_layer,
)


def _config():
    return {
        "teacher": {
            "levels": [
                {"chunk_dim": 5, "chunk_size": 2},
                {"chunk_dim": 9, "chunk_size": 3},
            ],
            "base_teacher": {
                "dim": 7,
                "window": 2,
                "span_lengths": [1, 1],
            },
        },
        "student": {"num_blocks": 3},
        "misc": {
            "evaluation": {"slot_mode": "surface"},
            "probe": {
                "offsets": None,
                "offset_mode": "auto",
            }
        },
        "dataset": {"window": 2},
    }


def test_probe_spec_from_multilevel_config():
    spec = probe_spec_from_config(_config())
    assert spec.num_layers == 4  # L0 plus three transformer blocks
    assert spec.slots_per_level == [6, 3, 1]
    assert spec.class_counts == [7, 5, 9]
    assert spec.offsets_by_level == [
        [-2, -1, 0, 1],
        [-4, -3, -2, -1, 0, 1, 2],
        list(range(-12, 7)),
    ]
    assert spec.level_spans == [6, 3, 1]


def test_probe_spec_from_single_level_hierarchical_config():
    config = {
        "teacher": {
            "chunk_size": 4,
            "chunk_dim": 13,
            "base_teacher": {"dim": 11, "burn_in": 3},
        },
        "student": {"num_blocks": 1},
        "misc": {
            "evaluation": {"slot_mode": "surface"},
            "probe": {"offsets": [-1, 0]},
        },
    }
    assert probe_spec_from_config(config) == ProbeSpec(
        num_layers=2,
        slots_per_level=[4, 1],
        offsets=[[-1, 0], [-1, 0]],
        class_counts=[11, 13],
        slot_mode="surface",
    )


def test_probe_spec_from_parallel_multilevel_config_includes_surface():
    config = {
        "teacher": {
            "chunk_sizes": [2, 3],
            "chunk_dims": [5, -1],
            "base_teacher": {"dim": 7, "burn_in": 2},
        },
        "student": {"num_blocks": 1},
        "dataset": {"dim": 9},
        "misc": {
            "evaluation": {"slot_mode": "surface"},
            "probe": {"offsets": [-1, 0, 1]},
        },
    }

    assert probe_spec_from_config(config) == ProbeSpec(
        num_layers=2,
        slots_per_level=[6, 3, 1],
        offsets=[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        class_counts=[7, 5, 9],
        slot_mode="surface",
    )


def test_probe_spec_retains_coarse_slot_mode():
    config = _config()
    config["misc"]["evaluation"]["slot_mode"] = "coarse"

    spec = probe_spec_from_config(config)

    assert spec.slots_per_level == [2, 3, 1]
    assert spec.level_spans == [6, 3, 1]


def test_probe_spec_reads_legacy_probe_scoped_slot_mode():
    config = _config()
    config["misc"].pop("evaluation")
    config["misc"]["probe"]["slot_mode"] = "coarse"

    assert probe_spec_from_config(config).slots_per_level == [2, 3, 1]


def test_probe_keys_match_logger_schema():
    assert probe_metric_key(2, 1, 0, 1, "excess_nll") == (
        "probe/L2/level1/slot0/k+1/excess_nll/val"
    )
    spec = ProbeSpec(2, [1], [-1, 0], [4])
    keys = probe_metric_keys(spec, metrics=["acc"])
    assert keys == [
        "probe/L0/level0/slot0/k-1/acc/val",
        "probe/L1/level0/slot0/k-1/acc/val",
        "probe/L0/level0/slot0/k0/acc/val",
        "probe/L1/level0/slot0/k0/acc/val",
    ]


def test_probe_keys_omit_unlogged_planning_bayes_metrics():
    spec = ProbeSpec(1, [1], [1], [4])
    assert probe_metric_keys(spec, metrics=["acc", "bayes_acc", "excess_nll"]) == [
        "probe/L0/level0/slot0/k+1/acc/val",
    ]


def _history():
    spec = ProbeSpec(num_layers=2, slots_per_level=[1], offsets=[-1, 0], class_counts=[4])
    steps = np.arange(0, 60, 10)
    history = pd.DataFrame({"_step": steps})
    scores = {
        # Final stable crossing at 30 for L0; transient at 10 must not count.
        (0, -1): [0.1, 0.9, 0.2, 0.85, 0.9, 0.95],
        # L1 acquires the same target earlier, at 20.
        (1, -1): [0.1, 0.2, 0.85, 0.9, 0.95, 0.95],
        # Current target: L0 never reaches threshold, L1 acquires at 30.
        (0, 0): [0.0, 0.2, 0.4, 0.5, 0.6, 0.7],
        (1, 0): [0.0, 0.2, 0.4, 0.85, 0.9, 0.95],
    }
    chance = 0.25
    for (layer, offset), progress in scores.items():
        ceiling = 1.0 if offset < 0 else 0.75
        bayes_nll = 0.0 if offset < 0 else 0.4
        acc = chance + np.asarray(progress) * (ceiling - chance)
        excess_nll = (1.0 - np.asarray(progress)) * (log(4) - bayes_nll)
        history[probe_metric_key(layer, 0, 0, offset, "acc")] = acc
        history[probe_metric_key(layer, 0, 0, offset, "bayes_acc")] = ceiling
        history[probe_metric_key(layer, 0, 0, offset, "bayes_nll")] = bayes_nll
        history[probe_metric_key(layer, 0, 0, offset, "excess_nll")] = excess_nll
        history[probe_metric_key(layer, 0, 0, offset, "nll")] = (
            bayes_nll + excess_nll
        )
    return spec, history


def test_probe_history_long_normalizes_against_chance_and_bayes():
    spec, history = _history()
    long = probe_history_long(history, spec)
    cell = long[(long.layer == 1) & (long.offset == 0)].sort_values("_step")
    assert cell["chance_acc"].unique().tolist() == [0.25]
    assert cell["attainable_acc"].unique().tolist() == [0.75]
    assert cell["ceiling_source"].unique().tolist() == ["teacher_bayes"]
    assert cell["acc_progress"].to_numpy() == pytest.approx(
        [0.0, 0.2, 0.4, 0.85, 0.9, 0.95]
    )
    assert cell["nll_progress"].to_numpy() == pytest.approx(
        [0.0, 0.2, 0.4, 0.85, 0.9, 0.95]
    )


def test_probe_history_long_marks_planning_ceiling_as_fallback():
    spec = ProbeSpec(1, [1], [1], [5])
    key = probe_metric_key(0, 0, 0, 1, "acc")
    long = probe_history_long(pd.DataFrame({"_step": [0, 1], key: [0.2, 0.6]}), spec)
    assert long["ceiling_source"].unique().tolist() == ["unit_fallback"]
    assert long["acc_progress"].tolist() == pytest.approx([0.0, 0.5])


def test_missing_probe_cells_reports_absent_grid_cells():
    spec, history = _history()
    missing_key = probe_metric_key(1, 0, 0, 0, "acc")
    history = history.drop(columns=[missing_key])
    missing = missing_probe_cells(history, spec)
    assert missing["key"].tolist() == [missing_key]


def test_sustained_acquisition_ignores_transient_and_requires_dwell():
    steps = [0, 10, 20, 30, 40, 50]
    scores = [0.1, 0.9, 0.2, 0.85, 0.9, 0.95]
    assert sustained_acquisition_step(steps, scores, min_dwell=3) == 30
    assert sustained_acquisition_step(steps, scores, min_dwell=4) is None
    assert sustained_acquisition_step(
        steps, scores, min_dwell=1, require_final=False
    ) == 10


def test_acquisition_tables_answer_both_learning_order_questions():
    spec, history = _history()
    long = probe_history_long(history, spec)
    events = acquisition_table(long, threshold=0.8, min_dwell=3)

    retention = events[events.offset == -1].set_index("layer")
    assert retention.loc[0, "acquisition_step"] == 30
    assert retention.loc[1, "acquisition_step"] == 20

    by_target = layer_order_by_target(events).set_index("target")
    assert by_target.loc["level0/slot0/k-1", "first_layer"] == "L1"
    assert by_target.loc["level0/slot0/k-1", "layer_order"] == "L1@20 -> L0@30"
    assert by_target.loc["level0/slot0/k0", "layer_order"] == "L1@30 -> L0@never"

    by_layer = target_order_by_layer(events).set_index("residual")
    assert by_layer.loc["L0", "first_target"] == "level0/slot0/k-1"
    assert by_layer.loc["L1", "first_step"] == 20


def test_adjacent_layer_delta_is_negative_when_deeper_layer_arrives_first():
    spec, history = _history()
    events = acquisition_table(probe_history_long(history, spec), min_dwell=3)
    deltas = adjacent_layer_deltas(events).set_index("target")
    assert deltas.loc["level0/slot0/k-1", "L1-L0"] == -10
    # L0 never acquires k0, so an adjacent comparison is undefined.
    assert np.isnan(deltas.loc["level0/slot0/k0", "L1-L0"])

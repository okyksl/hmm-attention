import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from notebooks.utils import fetch_attention_tables, fetch_step_metrics
from src.analysis.training_visualizations import (
    attention_table_to_array,
    plot_attention_alignment,
    plot_attention_heatmaps,
    plot_probe_heatmap,
    plot_probe_overview,
    plot_probe_slice,
    plot_probe_trajectory,
    probe_matrix_from_metrics,
    probe_layout_from_config,
    probe_slice_from_metrics,
)
from src.visualizer import build_attention_table


def _table_frame(attention: np.ndarray) -> pd.DataFrame:
    table = build_attention_table(attention)
    return pd.DataFrame(table.data, columns=table.columns)


def test_attention_table_round_trip():
    attention = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    frame = _table_frame(attention)

    reconstructed = attention_table_to_array(frame)

    assert np.array_equal(reconstructed, attention)


def test_attention_table_rejects_missing_cells():
    frame = _table_frame(np.ones((1, 2, 2), dtype=np.float32)).iloc[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        attention_table_to_array(frame)


def test_attention_plot_collections_recreate_current_figures():
    attention = np.zeros((2, 4, 4), dtype=np.float32)
    attention[:, :, 0] = 1.0

    heatmaps = plot_attention_heatmaps(attention, step=25)
    alignment = plot_attention_alignment(
        attention,
        span_lengths=[1, 1],
        context_length=2,
        step=25,
    )

    assert set(heatmaps) == {"average", "head1", "head2"}
    assert set(alignment) == {
        "align_cos_sim",
        "align_proj_norm",
        "offset_head1",
        "offset_head2",
    }
    for figure in [*heatmaps.values(), *alignment.values()]:
        plt.close(figure)


def test_dense_attention_heatmap_thins_tick_labels_and_keeps_endpoints():
    attention = np.zeros((1, 30, 30), dtype=np.float32)

    figures = plot_attention_heatmaps(attention, step=25)
    axis = figures["average"].axes[0]
    xlabels = [label.get_text() for label in axis.get_xticklabels()]
    ylabels = [label.get_text() for label in axis.get_yticklabels()]

    assert len(xlabels) == 10
    assert len(ylabels) == 10
    assert (xlabels[0], xlabels[-1]) == ("0", "29")
    assert (ylabels[0], ylabels[-1]) == ("0", "29")
    for figure in figures.values():
        plt.close(figure)


def test_probe_heatmap_reconstructs_scalar_grid():
    metrics = {
        f"probe/L{layer}/level0/slot{slot}/k-1/acc/val": layer + slot / 10
        for layer in range(3)
        for slot in range(2)
    }
    matrix = probe_matrix_from_metrics(
        metrics,
        level=0,
        offset=-1,
        metric="acc",
        split="val",
        num_layers=3,
        num_slots=2,
    )

    assert matrix.shape == (3, 2)
    assert matrix[2, 1] == pytest.approx(2.1)
    figure = plot_probe_heatmap(matrix, 0, -1, "acc", step=100)
    plt.close(figure)


def test_probe_trajectory_shows_one_selected_metric():
    prefix = "probe/L1/level0/slot1/k-1"
    rows = {
        0: {
            f"{prefix}/acc/val": 0.25,
            f"{prefix}/excess_nll/val": 1.2,
        },
        100: {
            f"{prefix}/acc/val": 0.75,
            f"{prefix}/excess_nll/val": 0.3,
        },
    }

    accuracy = plot_probe_trajectory(rows, 1, 0, 1, -1)
    excess_nll = plot_probe_trajectory(
        rows, 1, 0, 1, -1, metric="excess_nll"
    )

    assert len(accuracy.axes) == 1
    assert list(accuracy.axes[0].lines[0].get_xdata()) == [0, 100]
    assert list(accuracy.axes[0].lines[0].get_ydata()) == [0.25, 0.75]
    assert len(excess_nll.axes) == 1
    assert list(excess_nll.axes[0].lines[0].get_ydata()) == [1.2, 0.3]
    plt.close(accuracy)
    plt.close(excess_nll)


def test_planning_probe_trajectory_rejects_unavailable_excess_nll():
    key = "probe/L0/level0/slot0/k+1/acc/val"
    rows = {0: {key: 0.2}, 100: {key: 0.4}}

    figure = plot_probe_trajectory(rows, 0, 0, 0, 1)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_title().startswith("Accuracy —")
    with pytest.raises(ValueError, match="no probe excess_nll values"):
        plot_probe_trajectory(rows, 0, 0, 0, 1, metric="excess_nll")
    plt.close(figure)


def test_probe_slice_supports_arbitrary_axes_and_ragged_slots():
    metrics = {
        f"probe/L0/level{level}/slot{slot}/k0/acc/val": level + slot / 10
        for level, slot_count in enumerate([2, 3])
        for slot in range(slot_count)
    }
    coordinates = {"layer": 0, "level": 0, "slot": 0, "offset": 0}

    matrix, x_labels, y_labels = probe_slice_from_metrics(
        metrics, "level", "slot", coordinates, "acc", "val",
        num_layers=2, slots_per_level=[2, 3], offsets=[-1, 0, 1],
    )

    assert matrix.shape == (3, 2)
    assert np.isnan(matrix[2, 0])
    assert matrix[2, 1] == pytest.approx(1.2)
    assert x_labels == ["level0", "level1"]
    assert y_labels == ["slot0", "slot1", "slot2"]
    figure = plot_probe_slice(
        metrics, 100, "level", "slot", coordinates, "acc", "val",
        num_layers=2, slots_per_level=[2, 3], offsets=[-1, 0, 1],
    )
    assert figure.axes[0].get_xlabel() == "level"
    assert figure.axes[0].get_ylabel() == "slot"
    plt.close(figure)


def test_probe_overview_facets_levels_and_offsets_in_one_figure():
    metrics = {
        f"probe/L{layer}/level{level}/slot{slot}/k{offset}/acc/val": 0.5
        for layer in range(2)
        for level, slot_count in enumerate([1, 2])
        for slot in range(slot_count)
        for offset in [-1, 0]
    }

    figure = plot_probe_overview(
        metrics, 100, "acc", "val", num_layers=2,
        slots_per_level=[1, 2], offsets=[-1, 0],
    )

    assert len([axis for axis in figure.axes[:-1] if axis.get_visible()]) == 3
    assert figure._suptitle.get_text() == "All probe accuracy at step 100"
    plt.close(figure)


def test_probe_overview_supports_level_specific_offset_ranges():
    offsets = [[-1, 0], [-2, -1, 0, 1]]
    metrics = {
        f"probe/L0/level{level}/slot0/"
        f"{'k+' if offset > 0 else 'k'}{offset}/acc/val": 0.5
        for level, level_offsets in enumerate(offsets)
        for offset in level_offsets
    }

    figure = plot_probe_overview(
        metrics, 100, "acc", "val", num_layers=1,
        slots_per_level=[1, 1], offsets=offsets,
    )

    visible_axes = [axis for axis in figure.axes[:-1] if axis.get_visible()]
    assert len(visible_axes) == 2
    assert {axis.get_title() for axis in visible_axes} == {
        "level0, slot0", "level1, slot0",
    }
    by_title = {axis.get_title(): axis for axis in visible_axes}
    assert [tick.get_text() for tick in by_title["level0, slot0"].get_xticklabels()] == [
        "k=-1", "k=0",
    ]
    assert [tick.get_text() for tick in by_title["level1, slot0"].get_xticklabels()] == [
        "k=-2", "k=-1", "k=0", "k=+1",
    ]
    plt.close(figure)


def test_probe_layout_matches_resolved_multilevel_config():
    config = {
        "dataset": {"window": 3},
        "teacher": {
            "base_teacher": {"span_lengths": [1, 1, 1], "stride": None},
            "levels": [{"chunk_size": 2}, {"chunk_size": 3}],
        },
        "student": {"num_blocks": 2},
        "misc": {
            "probe": {
                "offsets": None,
                "offset_mode": "auto",
                "slot_mode": "surface",
            }
        },
    }

    layout = probe_layout_from_config(config)

    assert layout.num_layers == 3
    assert layout.slots_per_level == [6, 3, 1]
    assert layout.offsets_by_level == [
        [-3, -2, -1, 0, 1],
        list(range(-6, 3)),
        list(range(-18, 7)),
    ]


def test_probe_layout_matches_parallel_chunk_size_config():
    config = {
        "dataset": {"window": 3},
        "teacher": {
            "base_teacher": {"span_lengths": [1, 1, 1], "stride": None},
            "chunk_sizes": [2, 3],
        },
        "student": {"num_blocks": 3},
        "misc": {"probe": {"offsets": None, "slot_mode": "surface"}},
    }

    layout = probe_layout_from_config(config)

    assert layout.num_layers == 4
    assert layout.slots_per_level == [6, 3, 1]
    # No offset_mode marks an older run, whose null offsets used the same
    # base-level range at every level.
    assert layout.offsets_by_level == [
        [-3, -2, -1, 0, 1],
        [-3, -2, -1, 0, 1],
        [-3, -2, -1, 0, 1],
    ]


class _FakeDownload:
    def __init__(self, path: Path):
        self.name = str(path)


class _FakeFile:
    def __init__(self, relative_path: str, payload: dict):
        self.relative_path = relative_path
        self.payload = payload

    def download(self, root: str, replace: bool):
        target = Path(root) / self.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.payload))
        return _FakeDownload(target)


class _FakeRun:
    def __init__(self, rows, files):
        self.rows = rows
        self.files = files
        self.scan_calls = 0

    def scan_history(self, keys):
        self.scan_calls += 1
        for row in self.rows:
            yield {key: row.get(key) for key in keys}

    def file(self, path):
        return self.files[path]


class _FakeApi:
    def __init__(self, run):
        self._run = run

    def run(self, path):
        return self._run


def test_fetch_attention_tables_uses_exact_steps_and_cache(tmp_path):
    key = "attn/L1/weights/val"
    frame = _table_frame(np.ones((1, 2, 2), dtype=np.float32))
    payload = {"columns": list(frame.columns), "data": frame.values.tolist()}
    relative_path = "media/table/attention-25.table.json"
    run = _FakeRun(
        rows=[{"_step": 25, key: {"path": relative_path}}],
        files={relative_path: _FakeFile(relative_path, payload)},
    )
    api = _FakeApi(run)

    first = fetch_attention_tables("abc", [25], cache_dir=tmp_path, api=api)
    second = fetch_attention_tables("abc", [25], cache_dir=tmp_path, api=api)

    assert len(first[25]) == 4
    assert first[25].equals(second[25])
    assert (tmp_path / "abc" / relative_path).exists()


def test_fetch_step_metrics_returns_exact_unsampled_rows():
    run = _FakeRun(
        rows=[
            {"_step": 25, "metric": 1.0},
            {"_step": 50, "metric": 2.0},
            {"_step": 50, "other": 3.0},
        ],
        files={},
    )

    rows = fetch_step_metrics(
        "abc", [50], ["metric", "other"], api=_FakeApi(run)
    )

    assert rows[50]["metric"] == 2.0
    assert rows[50]["other"] == 3.0

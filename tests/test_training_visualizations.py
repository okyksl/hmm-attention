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
    probe_matrix_from_metrics,
    probe_layout_from_config,
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


def test_probe_layout_matches_resolved_multilevel_config():
    config = {
        "dataset": {"window": 3},
        "teacher": {
            "base_teacher": {"span_lengths": [1, 1, 1], "stride": None},
            "levels": [{"chunk_size": 2}, {"chunk_size": 3}],
        },
        "student": {"num_blocks": 2},
        "misc": {"probe": {"offsets": None}},
    }

    layout = probe_layout_from_config(config)

    assert layout.num_layers == 3
    assert layout.slots_per_level == [2, 3]
    assert layout.offsets == [-3, -2, -1, 0, 1]


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

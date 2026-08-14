import ipywidgets as widgets
import matplotlib.pyplot as plt

from notebooks.interactive import (
    fetch_logged_rows,
    prepare_figure_gallery,
    show_figure_gallery,
    show_layer_step_selector,
    show_probe_selector,
    show_probe_snapshot_selector,
)


class _FakeRun:
    def __init__(self, rows):
        self.rows = rows

    def scan_history(self, keys):
        for row in self.rows:
            yield {key: row.get(key) for key in keys}


def test_fetch_logged_rows_merges_sparse_duplicate_steps():
    run = _FakeRun([
        {"_step": 25, "first": 1.0},
        {"_step": 25, "second": 2.0},
        {"_step": 50},
    ])

    rows = fetch_logged_rows(run, ["first", "second"], "test history")

    assert rows == {25: {"_step": 25, "first": 1.0, "second": 2.0}}


def test_prepare_figure_gallery_returns_cached_png_bytes_and_closes_figures():
    figure, axis = plt.subplots()
    axis.plot([0, 1])
    figure_number = figure.number

    gallery = prepare_figure_gallery({"line": figure})

    assert gallery.items[0][0] == "line"
    assert gallery.items[0][1].startswith(b"\x89PNG")
    assert not plt.fignum_exists(figure_number)


def test_show_figure_gallery_reuses_its_widget(monkeypatch):
    figure, _ = plt.subplots()
    gallery = prepare_figure_gallery({"plot": figure})
    displayed = []
    monkeypatch.setattr("notebooks.interactive.display", displayed.append)

    first = show_figure_gallery(gallery)
    second = show_figure_gallery(gallery)

    assert first is second is gallery.frame
    assert displayed == [gallery.frame, gallery.frame]


def test_probe_selector_links_level_to_valid_slots(monkeypatch):
    displayed = []
    rendered = []
    monkeypatch.setattr("notebooks.interactive.display", displayed.append)

    controls, _ = show_probe_selector(
        num_layers=4,
        slots_per_level=[2, 3],
        offsets=[[-1, 0, 1], [-3, -2, -1, 0, 1, 2]],
        renderer=lambda layer, level, slot, offset, metric: rendered.append(
            (layer, level, slot, offset, metric)
        ),
    )

    assert rendered[-1] == (0, 0, 0, 0, "acc")
    assert isinstance(controls["offset"], widgets.Dropdown)
    assert [value for _, value in controls["slot"].options] == [0, 1]
    controls["level"].value = 1
    assert [value for _, value in controls["slot"].options] == [0, 1, 2]
    assert [value for _, value in controls["offset"].options] == [
        -3, -2, -1, 0, 1, 2,
    ]
    controls["layer"].value = 3
    controls["slot"].value = 2
    controls["offset"].value = -1
    assert rendered[-1] == (3, 1, 2, -1, "acc")
    controls["metric"].value = "excess_nll"
    assert rendered[-1] == (3, 1, 2, -1, "excess_nll")
    controls["offset"].value = 1
    assert controls["metric"].value == "acc"
    assert list(controls["metric"].options) == [("Accuracy", "acc")]
    assert len(displayed) == 1


def test_probe_snapshot_selector_switches_axes_and_overview(monkeypatch):
    displayed = []
    rendered = []
    monkeypatch.setattr("notebooks.interactive.display", displayed.append)

    controls, _ = show_probe_snapshot_selector(
        steps=[0, 100],
        num_layers=4,
        slots_per_level=[2, 3],
        offsets=[[-1, 0, 1], [-3, -2, -1, 0, 1, 2]],
        renderer=lambda **selection: rendered.append(selection),
    )

    assert rendered[-1]["step"] == 100
    assert isinstance(controls["offset"], widgets.Dropdown)
    assert rendered[-1]["x_axis"] == "slot"
    assert rendered[-1]["y_axis"] == "layer"
    assert controls["slot"].disabled
    assert controls["layer"].disabled
    controls["x_axis"].value = "level"
    assert controls["y_axis"].value != "level"
    assert [value for _, value in controls["slot"].options] == [0, 1, 2]
    assert [value for _, value in controls["offset"].options] == [
        -3, -2, -1, 0, 1, 2,
    ]
    controls["offset"].value = 1
    assert list(controls["metric"].options) == [("Accuracy", "acc")]
    controls["view"].value = "overview"
    assert controls["metric"].options == (
        ("Accuracy", "acc"), ("Excess NLL", "excess_nll")
    )
    assert all(
        controls[name].disabled for name in ("layer", "level", "slot", "offset")
    )
    assert rendered[-1]["view"] == "overview"
    assert len(displayed) == 1


def test_layer_step_selector_uses_only_steps_available_for_selected_layer(
    monkeypatch,
):
    displayed = []
    rendered = []
    monkeypatch.setattr("notebooks.interactive.display", displayed.append)

    def render(layer, step):
        rendered.append((layer, step))
        return widgets.HTML(value=f"{layer}:{step}")

    controls, output = show_layer_step_selector(
        {"L1": [0, 100], "L2": [0, 200], "L3": []},
        renderer=render,
        empty_message="none",
    )

    assert rendered[-1] == ("L1", 100)
    plot_slot = output.children[1]
    assert len(plot_slot.children) == 1
    assert plot_slot.children[0].value == "L1:100"
    assert list(controls["layer"].options) == ["L1", "L2"]
    controls["layer"].value = "L2"
    assert list(controls["step"].options) == [0, 200]
    assert controls["step"].value == 200
    assert rendered[-1] == ("L2", 200)
    assert len(plot_slot.children) == 1
    assert plot_slot.children[0].value == "L2:200"
    assert len(displayed) == 1

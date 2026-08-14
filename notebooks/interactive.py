"""Reusable interactive controls for analysis notebooks."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape
from io import BytesIO
from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from matplotlib.figure import Figure
from tqdm.auto import tqdm

from src.probe_offsets import all_probe_offsets, normalize_offsets_by_level


@dataclass
class FigureGallery:
    """Rendered plot images and their lazily constructed reusable widget."""

    items: list[tuple[str, bytes]]
    frame: widgets.Widget | None = None
    single_frame: widgets.Widget | None = None


def fetch_logged_rows(
    run: Any,
    keys: Sequence[str],
    description: str = "Scanning history",
) -> dict[int, dict[str, Any]]:
    """Merge sparse W&B history rows and retain every logged step."""
    unique_keys = list(dict.fromkeys(keys))
    rows: dict[int, dict[str, Any]] = {}
    history = run.scan_history(keys=["_step", *unique_keys])
    for raw in tqdm(
        history,
        desc=description,
        unit="row",
        leave=False,
        dynamic_ncols=True,
    ):
        if raw.get("_step") is None:
            continue
        values = {
            key: raw[key]
            for key in unique_keys
            if raw.get(key) is not None
        }
        if not values:
            continue
        step = int(raw["_step"])
        rows.setdefault(step, {"_step": step}).update(values)
    return rows


def prepare_figure_gallery(
    figures: Mapping[str, Figure],
    dpi: int = 130,
) -> FigureGallery:
    """Render figures once into reusable in-memory PNGs and close them."""
    items: list[tuple[str, bytes]] = []
    for name, figure in figures.items():
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        items.append((name, buffer.getvalue()))
        plt.close(figure)
    return FigureGallery(items)


def _image_widget(image: bytes, width: str) -> widgets.Image:
    return widgets.Image(
        value=image,
        format="png",
        layout=widgets.Layout(width=width, height="auto", object_fit="contain"),
    )


def figure_gallery_widget(gallery: FigureGallery) -> widgets.Widget | None:
    """Return a reusable gallery widget without appending notebook output."""
    if gallery.frame is not None:
        return gallery.frame

    items = gallery.items
    if not items:
        print("No plots are available for this step.")
        return None

    mode = widgets.ToggleButtons(
        options=[("All plots", "all"), ("One plot", "single")],
        value="all",
        description="View",
        style={"description_width": "initial"},
    )
    selector = widgets.Dropdown(
        options=[(name, index) for index, (name, _) in enumerate(items)],
        value=0,
        description="Plot",
        layout=widgets.Layout(width="520px", max_width="70%"),
        style={"description_width": "initial"},
    )
    previous_button = widgets.Button(
        icon="chevron-left",
        tooltip="Previous plot",
        layout=widgets.Layout(width="42px"),
        disabled=len(items) == 1,
    )
    next_button = widgets.Button(
        icon="chevron-right",
        tooltip="Next plot",
        layout=widgets.Layout(width="42px"),
        disabled=len(items) == 1,
    )
    counter = widgets.HTML()
    single_image = _image_widget(items[0][1], "100%")
    single_image.layout.max_width = "900px"

    cards = [
        widgets.VBox(
            [
                widgets.HTML(f"<b>{escape(name)}</b>"),
                _image_widget(image, "430px"),
            ],
            layout=widgets.Layout(
                width="450px",
                min_width="450px",
                padding="6px",
                border="1px solid #ddd",
            ),
        )
        for name, image in items
    ]
    all_panel = widgets.HBox(
        cards,
        layout=widgets.Layout(
            width="100%",
            overflow="auto hidden",
            flex_flow="row nowrap",
            align_items="flex-start",
        ),
    )
    single_panel = widgets.VBox(
        [
            widgets.HBox(
                [previous_button, selector, next_button, counter],
                layout=widgets.Layout(align_items="center"),
            ),
            single_image,
        ]
    )
    body = widgets.Box()

    def select_plot(change: dict[str, Any] | None = None) -> None:
        del change
        index = selector.value
        single_image.value = items[index][1]
        counter.value = f"<b>{index + 1} / {len(items)}</b>"

    def move_plot(offset: int) -> None:
        selector.value = (selector.value + offset) % len(items)

    def select_view(change: dict[str, Any] | None = None) -> None:
        del change
        body.children = (all_panel,) if mode.value == "all" else (single_panel,)

    selector.observe(select_plot, names="value")
    previous_button.on_click(lambda _: move_plot(-1))
    next_button.on_click(lambda _: move_plot(1))
    mode.observe(select_view, names="value")
    select_plot()
    select_view()

    frame = widgets.VBox(
        [mode, body],
        layout=widgets.Layout(
            width="100%",
            padding="8px",
            border="1px solid #bbb",
        ),
    )
    gallery.frame = frame
    return frame


def show_figure_gallery(gallery: FigureGallery) -> widgets.Widget | None:
    """Show all plots side by side or browse one plot with arrow controls."""
    frame = figure_gallery_widget(gallery)
    if frame is not None:
        display(frame)
    return frame


def single_figure_widget(gallery: FigureGallery) -> widgets.Widget | None:
    """Return one reusable figure widget without appending notebook output."""
    if gallery.single_frame is not None:
        return gallery.single_frame
    if not gallery.items:
        print("No plot is available for this selection.")
        return None

    name, image = gallery.items[0]
    frame = widgets.VBox(
        [widgets.HTML(f"<b>{escape(name)}</b>"), _image_widget(image, "100%")],
        layout=widgets.Layout(
            width="100%", max_width="1050px", padding="8px",
            border="1px solid #bbb",
        ),
    )
    gallery.single_frame = frame
    return frame


def show_single_figure(gallery: FigureGallery) -> widgets.Widget | None:
    """Display the first cached figure without gallery navigation controls."""
    frame = single_figure_widget(gallery)
    if frame is not None:
        display(frame)
    return frame


def _replaceable_widget_output(
    renderer: Callable[..., widgets.Widget | None],
    controls: Mapping[str, widgets.Widget],
) -> widgets.VBox:
    """Bind controls to one plot slot whose child is replaced atomically."""
    status = widgets.Output()
    plot_slot = widgets.Box(layout=widgets.Layout(width="100%"))

    def render(change: dict[str, Any] | None = None) -> None:
        del change
        values = {name: widget.value for name, widget in controls.items()}
        plot_slot.children = ()
        with status:
            clear_output(wait=True)
            child = renderer(**values)
        plot_slot.children = () if child is None else (child,)

    for control in controls.values():
        control.observe(render, names="value")
    render()
    return widgets.VBox([status, plot_slot], layout=widgets.Layout(width="100%"))


def show_probe_selector(
    num_layers: int,
    slots_per_level: Sequence[int],
    offsets: Sequence[int] | Sequence[Sequence[int]],
    renderer: Callable[[int, int, int, int, str], widgets.Widget | None],
) -> tuple[dict[str, widgets.Widget], widgets.VBox]:
    """Display linked coordinates and metric controls for one probe plot."""
    if num_layers < 1 or not slots_per_level:
        raise ValueError("probe selectors require non-empty layout dimensions")
    offsets_by_level = normalize_offsets_by_level(offsets, len(slots_per_level))

    layer = widgets.Dropdown(
        options=[(f"L{index}", index) for index in range(num_layers)],
        value=0, description="Layer",
        layout=widgets.Layout(width="145px"),
        style={"description_width": "initial"},
    )
    level = widgets.Dropdown(
        options=[(f"level{index}", index) for index in range(len(slots_per_level))],
        value=0, description="Level",
        layout=widgets.Layout(width="165px"),
        style={"description_width": "initial"},
    )
    slot = widgets.Dropdown(
        description="Slot",
        layout=widgets.Layout(width="135px"),
        style={"description_width": "initial"},
    )
    offset_values = offsets_by_level[0]
    offset = widgets.Dropdown(
        options=[(f"k={value:+d}" if value else "k=0", value)
                 for value in offset_values],
        value=0 if 0 in offset_values else offset_values[0],
        description="Offset",
        layout=widgets.Layout(width="165px"),
        style={"description_width": "initial"},
    )
    metric = widgets.ToggleButtons(
        options=[("Accuracy", "acc"), ("Excess NLL", "excess_nll")],
        value="acc", description="Metric",
        style={"description_width": "initial"},
    )

    def update_slots(change: dict[str, Any] | None = None) -> None:
        del change
        selected_level = int(level.value)
        options = [
            (f"slot{index}", index)
            for index in range(int(slots_per_level[selected_level]))
        ]
        current_slot = slot.value
        slot.options = options
        valid_slots = {value for _, value in options}
        if current_slot not in valid_slots:
            fallback = (
                0
                if current_slot is None
                else min(int(current_slot), len(options) - 1)
            )
            slot.value = fallback

    def update_offsets(change: dict[str, Any] | None = None) -> None:
        del change
        values = offsets_by_level[int(level.value)]
        options = [
            (f"k={value:+d}" if value else "k=0", value) for value in values
        ]
        current_offset = offset.value
        offset.options = options
        if current_offset not in set(values):
            offset.value = 0 if 0 in values else values[0]

    level.observe(update_slots, names="value")
    level.observe(update_offsets, names="value")
    update_slots()
    update_offsets()

    def update_metrics(change: dict[str, Any] | None = None) -> None:
        del change
        options = (
            [("Accuracy", "acc")]
            if int(offset.value) > 0
            else [("Accuracy", "acc"), ("Excess NLL", "excess_nll")]
        )
        current_metric = metric.value
        metric.options = options
        valid_metrics = {value for _, value in options}
        if current_metric not in valid_metrics:
            metric.value = "acc"

    offset.observe(update_metrics, names="value")
    update_metrics()
    controls = {
        "layer": layer, "level": level, "slot": slot,
        "offset": offset, "metric": metric,
    }
    output = _replaceable_widget_output(renderer, controls)
    control_row = widgets.HBox(
        list(controls.values()),
        layout=widgets.Layout(width="100%", flex_flow="row wrap"),
    )
    display(widgets.VBox([control_row, output]))
    return controls, output


def show_probe_snapshot_selector(
    steps: Sequence[int],
    num_layers: int,
    slots_per_level: Sequence[int],
    offsets: Sequence[int] | Sequence[Sequence[int]],
    renderer: Callable[..., widgets.Widget | None],
) -> tuple[dict[str, widgets.Widget], widgets.VBox]:
    """Display controls for a fixed-step probe slice or complete overview."""
    available_steps = sorted({int(step) for step in steps})
    if not available_steps or num_layers < 1 or not slots_per_level:
        raise ValueError("probe snapshot selectors require non-empty dimensions")
    offsets_by_level = normalize_offsets_by_level(offsets, len(slots_per_level))

    step = widgets.SelectionSlider(
        options=available_steps, value=available_steps[-1],
        description="Step", continuous_update=False,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "initial"},
    )
    view = widgets.ToggleButtons(
        options=[("2-D slice", "slice"), ("All probes", "overview")],
        value="slice", description="View",
        style={"description_width": "initial"},
    )
    metric = widgets.ToggleButtons(
        options=[("Accuracy", "acc"), ("Excess NLL", "excess_nll")],
        value="acc", description="Metric",
        style={"description_width": "initial"},
    )
    dimension_options = [
        ("Layer", "layer"), ("Level", "level"),
        ("Slot", "slot"), ("Offset", "offset"),
    ]
    x_axis = widgets.Dropdown(
        options=dimension_options, value="slot", description="X axis",
        layout=widgets.Layout(width="170px"),
        style={"description_width": "initial"},
    )
    y_axis = widgets.Dropdown(
        options=[option for option in dimension_options if option[1] != "slot"],
        value="layer", description="Y axis",
        layout=widgets.Layout(width="170px"),
        style={"description_width": "initial"},
    )
    layer = widgets.Dropdown(
        options=[(f"L{index}", index) for index in range(num_layers)],
        value=0, description="Layer",
        layout=widgets.Layout(width="145px"),
        style={"description_width": "initial"},
    )
    level = widgets.Dropdown(
        options=[(f"level{index}", index) for index in range(len(slots_per_level))],
        value=0, description="Level",
        layout=widgets.Layout(width="165px"),
        style={"description_width": "initial"},
    )
    slot = widgets.Dropdown(
        description="Slot", layout=widgets.Layout(width="135px"),
        style={"description_width": "initial"},
    )
    offset_values = offsets_by_level[0]
    offset = widgets.Dropdown(
        options=[(f"k={value:+d}" if value else "k=0", value)
                 for value in offset_values],
        value=0 if 0 in offset_values else offset_values[0],
        description="Offset", layout=widgets.Layout(width="165px"),
        style={"description_width": "initial"},
    )

    def update_y_axis(change: dict[str, Any] | None = None) -> None:
        del change
        current = y_axis.value
        options = [option for option in dimension_options
                   if option[1] != x_axis.value]
        y_axis.options = options
        valid = {value for _, value in options}
        if current not in valid:
            y_axis.value = options[0][1]

    def update_slots(change: dict[str, Any] | None = None) -> None:
        del change
        if level.value is None:
            return
        slot_count = (
            max(int(count) for count in slots_per_level)
            if "level" in {x_axis.value, y_axis.value}
            else int(slots_per_level[int(level.value)])
        )
        options = [(f"slot{index}", index) for index in range(slot_count)]
        current = slot.value
        slot.options = options
        valid = {value for _, value in options}
        if current not in valid:
            slot.value = 0 if current is None else min(int(current), slot_count - 1)

    def update_offsets(change: dict[str, Any] | None = None) -> None:
        del change
        level_is_axis = "level" in {x_axis.value, y_axis.value}
        values = (
            all_probe_offsets(offsets_by_level)
            if level_is_axis
            else offsets_by_level[int(level.value)]
        )
        options = [
            (f"k={value:+d}" if value else "k=0", value) for value in values
        ]
        current = offset.value
        offset.options = options
        if current not in set(values):
            offset.value = 0 if 0 in values else values[0]

    def update_metric(change: dict[str, Any] | None = None) -> None:
        del change
        offset_is_axis = "offset" in {x_axis.value, y_axis.value}
        options = (
            [("Accuracy", "acc")]
            if (view.value == "slice" and not offset_is_axis
                and int(offset.value) > 0)
            else [("Accuracy", "acc"), ("Excess NLL", "excess_nll")]
        )
        current = metric.value
        metric.options = options
        if current not in {value for _, value in options}:
            metric.value = "acc"

    fixed_widgets = {
        "layer": layer, "level": level, "slot": slot, "offset": offset,
    }
    slice_row = widgets.HBox(
        [x_axis, y_axis, *fixed_widgets.values()],
        layout=widgets.Layout(width="100%", flex_flow="row wrap"),
    )

    def update_view(change: dict[str, Any] | None = None) -> None:
        del change
        is_slice = view.value == "slice"
        slice_row.layout.display = "flex" if is_slice else "none"
        selected_axes = {x_axis.value, y_axis.value}
        for dimension, widget in fixed_widgets.items():
            widget.disabled = not is_slice or dimension in selected_axes

    x_axis.observe(update_y_axis, names="value")
    x_axis.observe(update_slots, names="value")
    y_axis.observe(update_slots, names="value")
    level.observe(update_slots, names="value")
    x_axis.observe(update_offsets, names="value")
    y_axis.observe(update_offsets, names="value")
    level.observe(update_offsets, names="value")
    x_axis.observe(update_metric, names="value")
    y_axis.observe(update_metric, names="value")
    offset.observe(update_metric, names="value")
    view.observe(update_metric, names="value")
    x_axis.observe(update_view, names="value")
    y_axis.observe(update_view, names="value")
    view.observe(update_view, names="value")
    update_y_axis()
    update_slots()
    update_offsets()
    update_metric()
    update_view()

    controls = {
        "step": step, "view": view, "metric": metric,
        "x_axis": x_axis, "y_axis": y_axis,
        **fixed_widgets,
    }
    output = _replaceable_widget_output(renderer, controls)
    primary_row = widgets.HBox(
        [view, metric],
        layout=widgets.Layout(width="100%", flex_flow="row wrap"),
    )
    display(widgets.VBox([step, primary_row, slice_row, output]))
    return controls, output


def show_step_slider(
    steps: Sequence[int],
    renderer: Callable[[int], widgets.Widget | None],
    empty_message: str,
) -> tuple[widgets.SelectionSlider, widgets.VBox] | None:
    """Display a slider over exact available steps and capture plot output."""
    available_steps = sorted({int(step) for step in steps})
    if not available_steps:
        print(empty_message)
        return None
    slider = widgets.SelectionSlider(
        options=available_steps,
        value=available_steps[-1],
        description="Step",
        continuous_update=False,
        readout=True,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "initial"},
    )
    output = _replaceable_widget_output(renderer, {"step": slider})
    display(widgets.VBox([slider, output]))
    return slider, output


def show_layer_step_selector(
    steps_by_layer: Mapping[str, Sequence[int]],
    renderer: Callable[[str, int], widgets.Widget | None],
    empty_message: str,
) -> tuple[dict[str, widgets.Widget], widgets.VBox] | None:
    """Display linked layer and exact-step selectors for logged snapshots."""
    available = {
        str(layer): sorted({int(step) for step in steps})
        for layer, steps in steps_by_layer.items()
        if steps
    }
    if not available:
        print(empty_message)
        return None

    layers = list(available)
    layer = widgets.Dropdown(
        options=layers, value=layers[0], description="Layer",
        layout=widgets.Layout(width="180px"),
        style={"description_width": "initial"},
    )
    step = widgets.SelectionSlider(
        options=available[layer.value], value=available[layer.value][-1],
        description="Step", continuous_update=False, readout=True,
        layout=widgets.Layout(width="100%"),
        style={"description_width": "initial"},
    )

    def update_steps(change: dict[str, Any] | None = None) -> None:
        del change
        choices = available[str(layer.value)]
        current = step.value
        step.options = choices
        step.value = current if current in choices else choices[-1]

    layer.observe(update_steps, names="value")
    controls = {"layer": layer, "step": step}
    output = _replaceable_widget_output(renderer, controls)
    display(widgets.VBox([widgets.HBox([layer]), step, output]))
    return controls, output

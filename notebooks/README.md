# Notebooks

Interactive analysis of training runs. Everything here reads **W&B logs** after
the fact — nothing in this folder is imported by training.

## What lives where

| Path | Role |
| --- | --- |
| `head_phases.ipynb` | **The analysis notebook.** Fetch runs by tag, detect stage-wise head specialization, plot the trajectories. |
| `post_training_visualizations.ipynb` | Recreate attention, alignment, value, and probe images from numerical W&B logs. |
| `report_head_phases.py` | Same read-out as a one-shot CLI, for when you don't want a notebook. |
| `utils.py` | W&B fetching (`fetch_runs`, `get_runs_data`) and shared plot styling. |
| `figures/` | Saved plots. Gitignored — regenerate from the notebook. |

The *logic* lives in `src/analysis/head_phases.py` (unit-tested in
`tests/test_head_phases.py`); this folder is only the interactive shell around
it. Keep it that way — anything worth a test belongs in `src/analysis/`.

## Running

Always from the repository root, so `src.` and `notebooks.` both resolve:

```bash
uv run jupyter lab                                     # then open head_phases.ipynb
uv run python -m notebooks.report_head_phases --demo   # no W&B needed
uv run python -m notebooks.report_head_phases --tags power-spectrum
```

`python notebooks/report_head_phases.py` does **not** work — Python puts the
script's own directory on `sys.path`, not the repo root, so `import src` fails.
Use `-m`.

## W&B target

Training writes to the project in `conf/misc/default.yaml`:

```
entity: okyksl      project: hmm-attention
```

`utils.py`'s `DEFAULT_ENTITY` / `DEFAULT_PROJECT` match these settings. Pointing
at the wrong project returns an **empty frame with no error**, so override them
explicitly when analyzing runs from another W&B project.

## Post-training images

Training keeps the requested attention/probe cadence but logs numerical data
only. The attention tables under `attn/L{layer}/weights/{split}` contain
batch-averaged attention **activations**, not learned parameter weights. Probe
layer×slot grids are represented by their per-cell scalar series. This avoids
Matplotlib rendering, PNG encoding, and image uploads in the training loop.

Open `post_training_visualizations.ipynb`, enter a run ID and exact logged steps,
then render any of the former images locally. Numerical attention tables are
cached under `notebooks/.cache/`; optional PNG/PDF output belongs under
`notebooks/figures/`. Both directories are gitignored.

## What the analysis answers

Given a run, which attention head learned which context position, and in what
order? The training loop logs, per (head, span) pair and per step:

- `attn/L1/span_mass_head{h}_span{k}/{split}` — attention mass head `h` puts on span `k`
- `attn/L1/align_cos_sim_head{h}_span{k}/{split}` — the scale-free counterpart

With `teacher.span_lengths=[1,1,1]` each span is a single position, so span `k`
*is* offset `-(W-k)`. The notebook turns those series into a one-line story per
run, e.g. `H3:-1@50 -> H2:-2@400 -> H1:-3@800`, and checks it against the order
the run's own `lag_spectrum` config predicts.

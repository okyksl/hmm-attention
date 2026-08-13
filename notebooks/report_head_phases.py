"""CLI: print the head-specialization story for a set of runs.

    uv run python analysis/report_head_phases.py --tags power-spectrum
    uv run python analysis/report_head_phases.py --demo        # no wandb needed

Reads the per-(head, span) attention scalars logged during training and prints,
per run, which head owns which span, when each locked on, and whether that order
matches the one the run's own `lag_spectrum` config predicts. See
`src/analysis/head_phases.py` for the underlying logic.
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.analysis.head_phases import (
    COS_SIM,
    FOCUS_STRATEGIES,
    SPAN_MASS,
    VALUE_COS,
    VALUE_INNER,
    metric_key,
    metric_keys,
    phase_table,
)

# Training writes to conf/misc/default.yaml's wandb target, which is NOT the
# default baked into notebooks/utils.py — pointing at the wrong project returns
# an empty frame with no error, so keep these in sync with the train config.
DEFAULT_ENTITY = "okyksl"
DEFAULT_PROJECT = "hmm-attention"


def _demo_frame(num_heads: int, num_spans: int) -> pd.DataFrame:
    """A synthetic two-run frame, so the output format can be seen offline."""
    frames = []
    for run_id, onsets, a_in, a_out in [
        ("demo-interleaved", (800, 400, 50), 1.0, 0.5),
        ("demo-control", (50, 400, 800), 0.5, 1.0),
    ]:
        steps = np.arange(0, 1000, 25)
        data = {"_step": steps, "_run_id": run_id, "_run_name": run_id}
        for h in range(num_heads):
            for k in range(num_spans):
                data[metric_key(h + 1, k + 1)] = (
                    np.where(steps >= onsets[h], 0.9, 0.05)
                    if h == k
                    else np.full(len(steps), 0.05)
                )
        df = pd.DataFrame(data)
        df["cfg.teacher.spectrum.alpha"] = a_in
        df["cfg.teacher.lag_spectrum.alpha"] = a_out
        df["cfg.teacher.lag_spectrum.law"] = "power"
        df["cfg.teacher.lag_spectrum.normalize"] = "none"
        df["cfg.teacher.lag_spectrum.reverse"] = True
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", nargs="+", default=["power-spectrum"])
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--heads", type=int, default=3)
    p.add_argument("--spans", type=int, default=3)
    p.add_argument("--span-lengths", type=int, nargs="+", default=[1, 1, 1])
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--layer", default="L1")
    p.add_argument(
        "--metric",
        default=SPAN_MASS,
        choices=[SPAN_MASS, COS_SIM, VALUE_COS, VALUE_INNER],
        help="span_mass/align_cos_sim = where a head looks (attention); "
        "value_cos/value_inner = which teacher matrix its value projection "
        "implements. The two are independent read-outs of the same story.",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="how much a span must dominate before a head counts as focused on it",
    )
    p.add_argument(
        "--strategy",
        default="share",
        choices=list(FOCUS_STRATEGIES),
        help="how --cutoff is applied: share of the head's mass, absolute value, "
        "margin over the runner-up, or ratio to it",
    )
    p.add_argument(
        "--min-dwell",
        type=int,
        default=2,
        help="samples a label must persist before it counts as a phase",
    )
    p.add_argument("--demo", action="store_true", help="use a synthetic frame")
    args = p.parse_args(argv)

    if args.demo:
        df = _demo_frame(args.heads, args.spans)
    else:
        from notebooks.utils import fetch_runs, get_runs_data

        runs = fetch_runs(entity=args.entity, project=args.project, tags_any=args.tags)
        if not runs:
            print(
                f"No runs with tags {args.tags} in {args.entity}/{args.project}.",
                file=sys.stderr,
            )
            return 1
        print(f"Fetched {len(runs)} run(s) from {args.entity}/{args.project}")
        keys = metric_keys(
            args.heads, args.spans, layer=args.layer, split=args.split, metric=args.metric
        )
        df = get_runs_data(runs, keys)
        if df.empty:
            print(
                f"No history rows carried all {len(keys)} attention keys. Was the run "
                "trained with a TransformerDecoder student and attention logging on?",
                file=sys.stderr,
            )
            return 1

    table = phase_table(
        df,
        num_heads=args.heads,
        num_spans=args.spans,
        span_lengths=args.span_lengths,
        layer=args.layer,
        split=args.split,
        metric=args.metric,
        cutoff=args.cutoff,
        strategy=args.strategy,
        min_dwell=args.min_dwell,
    )

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        # `label` already carries the varying config, so the raw wandb name and
        # the individual axis columns are redundant noise in the printout.
        axis_cols = [c for c in table.columns if c.startswith("cfg.")]
        cols = [c for c in table.columns if c not in {"_run_id", "_run_name", *axis_cols}]
        print(table[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

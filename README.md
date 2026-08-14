# hmm-attention

A research repository for studying **how a transformer discovers the abstract
compositional units of a hierarchical grammar**. A fixed "teacher" process
generates sequences from a stratified PCFG (equivalently, a hierarchical HMM); a
transformer "student" is trained to imitate it; probes then measure *where and
when* the student represents each latent level.

## Setting

- **Teacher.** An autoregressive base process over a hidden alphabet, wrapped in
    one or more chunk levels. Each level maps a token to fixed-length surface
    tuples via a codebook with globally disjoint supports, so surface→latent
    decoding is exact. `num_tuples > 1` gives each unit several interchangeable
    spellings — the knob that forces *abstraction* over surface memorization.
    Next-token probabilities are computed by an exact Bayes fold over the nested
    open chunks, so every latent has a known optimal posterior.
- **Student.** A decoder-only transformer trained to match the teacher's
    next-token distribution.
- **Instruments.** Linear probes on the residual stream decode each level's
    latent, compared against the teacher's Bayes-optimal ceiling; a post-training
    tool tests whether a probe generalizes across a unit's spellings.

## Layout

- `src/teachers/` — base processes (`LinearARTeacher`, `AttentionARTeacher`) and
    the chunk stack (`ChunkCode`, `MultiLevelHierarchicalTeacher`, and its `L=1`
    `HierarchicalTeacher`).
- `src/predictors/`, `src/data.py` — turn a teacher into a sampler and generate
    the AR dataset.
- `src/model/` — the transformer student.
- `src/trainer/` — training loop, evaluation, and the residual `ProbeLogger`.
- `src/analysis/` — post-training analyses (e.g. spelling generalization).
- `src/runner/`, `conf/` — Hydra config and entry glue.
- `tests/` — correctness suite for teachers, probes, and analyses.

## Usage

Run from the `torch` conda env or with `uv`:

```bash
uv run python train.py experiments=multilevel_dissection
uv run python -m pytest
```

Training checkpoints are content-addressed by the fully resolved config and
automatically resume the same W&B run. To start one config over on a cluster,
set a new non-null reset token in its config or Hydra overrides:

```bash
uv run python train.py experiments=multilevel_dissection misc.checkpoint.reset=restart-1
```

The first worker that owns that config removes its local checkpoint and
completion marker, starts at step 0, and creates a new W&B run. The token is
recorded separately, so cluster retries with the same value resume normally;
use `restart-2` (and so on) for later resets. Remote W&B runs are never deleted
by the code and remain yours to remove manually. Stop the old worker before
resetting so it no longer owns the config lock.

Teachers, chunk levels, base process, and probes are all configured under
conf/ (see conf/experiments/ for worked examples).

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
import wandb


@dataclass
class SchedulerConfig:
    """LR scheduler + calling convention (pass metric? step every iter?)."""

    scheduler: torch.optim.lr_scheduler._LRScheduler
    pass_metric: bool = False
    update_on_iter: bool = False


@dataclass
class NgramConfig:
    """Auxiliary ngram models + their optimizers + training-phase knobs.

    `use_teacher_target=True` swaps the ngram's own targets for the teacher's
    soft distribution during training — useful when the teacher is the
    supervising signal rather than the raw sequence.
    """

    models: Dict[str, torch.nn.Module] = field(default_factory=dict)
    optimizers: Dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    steps: int = 0
    use_teacher_target: bool = False


@dataclass
class LoggingConfig:
    """W&B writer + logging cadence.

    `log_frequency` controls how often the val loop runs, how often teacher
    KL / true-loss metrics are computed against the current train batch, and
    how often we write to W&B + reset accumulator metrics. Student loss/acc
    still update every step (accumulated between logs). Default 1 preserves
    the pre-existing "every step" behavior.
    """

    writer: Optional[wandb.run] = None
    attention_frequency: int = 100
    log_frequency: int = 1
    # Hidden-state probe (see src/trainer/probe_logger.py).
    # `probe_mode` selects fitting strategy:
    #   "off"        — disabled, no hooks, no cost.
    #   "warm_start" — per-eval LBFGS fit from previous weights. Hooks installed
    #                  only during val forward. Best for standard eval cadence.
    #   "sgd"        — persistent Adam update per training step. Hooks installed
    #                  permanently; per-step cost is minimal. Best for
    #                  high-frequency probing.
    # `probe_offsets=None` + `probe_offset_mode="auto"` uses a shared
    # surface-token horizon: full teacher context for retention and one
    # top-level unit for planning. Lower levels therefore receive more k's.
    # A flat explicit list broadcasts to every level; a nested list configures
    # each level independently. `probe_offset_mode="legacy"` restores the old
    # common `[-base_teacher.context_length, ..., +1]` range.
    # `probe_sharing` selects whether slots share one linear readout or fit
    # independent readouts: "shared" | "per_slot".
    # `probe_slot_mode` selects the reported phase resolution:
    #   "surface" — every surface-relative position within the target unit.
    #   "coarse"  — only the immediate-child slot (legacy behavior).
    probe_mode: str = "off"
    probe_frequency: int = 100
    probe_offsets: Optional[Union[List[int], List[List[int]]]] = None
    probe_offset_mode: str = "auto"
    probe_sharing: str = "shared"
    probe_slot_mode: str = "surface"
    probe_max_iters: int = 20
    probe_l2: float = 1e-3
    probe_lr: float = 1e-2
    probe_train_frac: float = 0.8

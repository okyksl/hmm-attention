import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from src.predictors import build_predictor
from src.runner.data import get_loaders, get_optimizer
from src.runner.preprocess import configure_positional_encoding, preprocess_cfg
from src.runner.verbose import log_student_summary, log_teacher_summary
from src.trainer import LoggingConfig, NgramConfig, SchedulerConfig, Trainer
from src.trainer.checkpoint import (
    CHECKPOINT_FILENAME,
    apply_checkpoint_reset,
    assert_config_matches,
    checkpoint_config_hash,
    load_checkpoint,
    reset_token_applied,
)
from src.trainer import lock as work_lock


def _pass_sched_metric(scheduler: torch.optim.lr_scheduler._LRScheduler) -> bool:
    """True if the scheduler expects a metric on `.step(metric)` (e.g. plateau)."""
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _update_scheduler_each_iter(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> bool:
    """True if the scheduler advances per iter rather than per epoch."""
    return isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def _init_wandb(
    cfg: DictConfig,
    resume_run_id: Optional[str] = None,
    rewind_step: Optional[int] = None,
):
    """Initialise the wandb run.

    Two resume modes, gated by `cfg.misc.wandb.rewind`:

    - **rewind=False (default):** `id=<run_id>, resume="must"`. Reattaches to
      the same run; wandb silently drops any `log(step=k)` where
      k ≤ last_logged_step. Safe when `checkpoint.frequency=1` because the
      last checkpoint step equals the last wandb-logged step — no drops.

    - **rewind=True:** `resume_from="<id>?_step=<step>"`. Uses wandb Run
      Rewind (SDK >= 0.17.1) to truncate history past `rewind_step`, then
      resume. Cleanly overwrites phantom curves when `checkpoint.frequency>1`.
      Rewind is currently **in private preview on wandb.ai** — enabling it
      without account access raises `CommError: Rewind is in private
      preview`. Leave off unless your workspace has it enabled.

    Both modes hard-fail if the server-side run is missing, preserving the
    single-wandb-run guarantee across resumes.
    """
    kwargs = dict(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        entity=cfg.misc.wandb.entity,
        project=cfg.misc.wandb.project,
        tags=cfg.misc.wandb.tags,
        settings=wandb.Settings(start_method="thread"),
    )
    if resume_run_id is not None:
        if cfg.misc.wandb.get("rewind", False):
            assert rewind_step is not None, (
                "rewind_step is required when resume_run_id is provided"
            )
            kwargs["resume_from"] = f"{resume_run_id}?_step={rewind_step}"
        else:
            kwargs["id"] = resume_run_id
            kwargs["resume"] = "must"
    return wandb.init(**kwargs)


def _read_checkpoint_metadata(
    path: Path, device: torch.device
) -> Tuple[Optional[str], Optional[int], Optional[Dict[str, Any]]]:
    """Peek at a checkpoint to get its wandb id + saved step + full payload.

    Returns `(wandb_id, saved_step, payload)`. Payload is loaded once here and
    reused when building the trainer — avoids two `torch.load` passes.
    `saved_step` is None for the pre-stub case (no checkpoint) and 0 for the
    stub (rewind-to-0 is a no-op that still keeps the run id stable).
    """
    if not path.exists():
        return None, None, None
    payload = load_checkpoint(path, device)
    return (
        payload.get("wandb_run_id"),
        payload.get("current_step", 0),
        payload,
    )


def _build_student(
    cfg: DictConfig, teacher: torch.nn.Module
) -> Tuple[torch.nn.Module, Optional[int], Optional[torch.Tensor]]:
    """Instantiate the student. When `teacher_readout` is set, warm-start its
    value projections from the teacher weights.

    Returns `(student, window, teacher_matrices)`; the latter two are forwarded
    to ngram model instantiation so all auxiliary models share the same
    readout configuration.
    """
    if "teacher_readout" in cfg.student and cfg.student.teacher_readout:
        window = teacher.window
        teacher_matrices = teacher._get_weights().detach().clone()
        student = instantiate(
            cfg.student, window=window, teacher_matrices=teacher_matrices
        )
        return student, window, teacher_matrices
    return instantiate(cfg.student), None, None


def _build_ngram_models(
    cfg: DictConfig, window: Optional[int], teacher_matrices: Optional[torch.Tensor]
) -> Dict[str, torch.nn.Module]:
    if "ngrams" not in cfg:
        return {}
    return {
        name: instantiate(m_cfg, window=window, teacher_matrices=teacher_matrices)
        for name, m_cfg in cfg.ngrams.items()
    }


def get_trainer(cfg: DictConfig) -> Optional[Trainer]:
    """Construct the trainer, or return None if this config is claimed.

    Returns None when either:
    - the config has already completed (a `done` marker exists in its ckpt dir), or
    - another live worker holds the flock on the config's `lock` file.

    In both cases the caller (`train.py`) exits cleanly without touching wandb,
    so parallel sweep workers cooperatively partition the configs with zero
    coordination overhead beyond a filesystem lock.
    """
    cfg = preprocess_cfg(cfg)

    teacher = instantiate(cfg.teacher)
    # Burn-in seeds the dataset before autoregression: the teacher's minimum
    # prefix. Equals context_length for bounded teachers; smaller for adaptive
    # ones (which report context_length == ADAPTIVE).
    prefix_length = getattr(teacher, "burn_in", None)
    if prefix_length is None:
        prefix_length = sum(teacher.span_lengths)
    configure_positional_encoding(cfg, teacher)

    with open("config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # Checkpointing: content-addressed dir keyed by config hash. Same
    # resolved config → same dir → resume works. Hydra's per-launch
    # timestamped dir (still cwd, since `hydra.job.chdir: true`) keeps
    # config.yaml and stdout separated per launch; only `checkpoint.pt` moves
    # to the canonical hash-keyed location. This handles Runai job restarts
    # (fresh Hydra timestamp on each pod launch) and sweeps (each config
    # auto-partitions into its own hash dir with its own wandb run).
    ckpt_cfg = cfg.misc.get("checkpoint", {})
    ckpt_enabled = ckpt_cfg.get("enabled", True)
    resume_enabled = ckpt_cfg.get("resume", True)
    reset_token = ckpt_cfg.get("reset", None)
    ckpt_root = Path(ckpt_cfg.get("root", "outputs/checkpoints"))

    cfg_container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    current_cfg_hash = checkpoint_config_hash(cfg_container)

    checkpoint_path: Optional[Path] = None
    ckpt_dir: Optional[Path] = None
    if ckpt_enabled:
        # Truncate to 16 chars — collision risk is negligible (2^64 space)
        # while keeping the path short enough to be human-legible.
        ckpt_dir = ckpt_root / current_cfg_hash[:16]
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / CHECKPOINT_FILENAME

    # A non-null reset token is a one-shot generation: the first worker to see
    # a new value clears this identity's checkpoint + done marker. The applied
    # token remains on disk, so cluster retries with the same config resume the
    # new run instead of repeatedly starting over. Bump the token for another
    # intentional restart. Remote W&B state is never touched here.
    reset_pending = (
        ckpt_dir is not None
        and reset_token is not None
        and not reset_token_applied(ckpt_dir, reset_token)
    )

    # Work coordination: bail out BEFORE wandb.init if this config is either
    # already completed or claimed by a live worker. A pending reset bypasses
    # the done check but still must own the same worker lock.
    if ckpt_dir is not None:
        if not reset_pending and work_lock.is_completed(ckpt_dir):
            logging.getLogger().info(
                f"Config {current_cfg_hash[:16]} already completed; skipping."
            )
            return None
        if not work_lock.try_acquire(ckpt_dir):
            logging.getLogger().info(
                f"Config {current_cfg_hash[:16]} is being worked on by another "
                "worker; skipping."
            )
            return None

        # Re-check after acquiring in case reset coordination changes in a
        # future blocking-lock implementation.
        if reset_pending and not reset_token_applied(ckpt_dir, reset_token):
            previous_wandb_id = None
            if checkpoint_path is not None and checkpoint_path.exists():
                try:
                    old_payload = load_checkpoint(
                        checkpoint_path, torch.device("cpu")
                    )
                    previous_wandb_id = old_payload.get("wandb_run_id")
                except Exception as exc:
                    logging.getLogger().warning(
                        f"Could not read checkpoint metadata before reset: {exc}"
                    )
            removed = apply_checkpoint_reset(ckpt_dir, reset_token)
            logging.getLogger().warning(
                f"Applied checkpoint reset token {reset_token!r} to "
                f"{ckpt_dir}; removed {[p.name for p in removed]}. "
                f"Previous W&B run id: {previous_wandb_id or 'unavailable'}. "
                "Remote W&B state was not changed."
            )

    resume_payload: Optional[Dict[str, Any]] = None
    resume_wandb_id: Optional[str] = None
    resume_step: Optional[int] = None
    if resume_enabled and checkpoint_path is not None:
        resume_wandb_id, resume_step, resume_payload = _read_checkpoint_metadata(
            checkpoint_path, torch.device(cfg.misc.device)
        )
        if resume_payload is not None:
            assert_config_matches(resume_payload, current_cfg_hash)

    writer = (
        _init_wandb(
            cfg,
            resume_run_id=resume_wandb_id,
            rewind_step=resume_step,
        )
        if cfg.misc.wandb.use
        else None
    )

    # Persist the wandb id NOW, before any construction work that could OOM /
    # crash. Anything between `wandb.init` and this write is a hole where the
    # wandb run exists on the server but no stub is on disk — the next launch
    # would then call `wandb.init` fresh and orphan the first run. We saw
    # exactly this (helpful-universe-117 crashed pre-first-log, then a resume
    # attempt spawned royal-armadillo-118 as a new run).
    if (
        resume_payload is None
        and checkpoint_path is not None
        and writer is not None
    ):
        _save_wandb_stub(checkpoint_path, writer.id, current_cfg_hash)

    student, window, teacher_matrices = _build_student(cfg, teacher)

    predictor_cfg = (
        OmegaConf.to_container(cfg.predictor, resolve=True) if "predictor" in cfg else {}
    )
    predictor = build_predictor(teacher, **predictor_cfg)

    train_loader, val_loader = get_loaders(
        cfg.dataset, cfg.trainer, predictor=predictor, prefix_length=prefix_length
    )
    loss_fn = instantiate(cfg.loss)
    optimizer = get_optimizer(cfg.optimizer, student)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    ngram_models = _build_ngram_models(cfg, window, teacher_matrices)
    optim_ngram = {
        name: get_optimizer(cfg.optimizer, model)
        for name, model in ngram_models.items()
    }

    if cfg.misc.verbose:
        log_teacher_summary(teacher)
        log_student_summary(student)

    scheduler_cfg = SchedulerConfig(
        scheduler=scheduler,
        pass_metric=_pass_sched_metric(scheduler),
        update_on_iter=_update_scheduler_each_iter(scheduler),
    )
    ngram_cfg = NgramConfig(
        models=ngram_models,
        optimizers=optim_ngram,
        steps=cfg.trainer.ngram_steps,
        use_teacher_target=(
            cfg.trainer.use_teacher_target
            if "use_teacher_target" in cfg.trainer
            else False
        ),
    )
    probe_cfg = cfg.misc.get("probe", {})
    probe_offsets = probe_cfg.get("offsets", None)
    if probe_offsets is not None:
        probe_offsets = OmegaConf.to_container(probe_offsets, resolve=True)
    logging_cfg = LoggingConfig(
        writer=writer,
        attention_frequency=cfg.misc.log_attention_frequency,
        log_frequency=cfg.misc.get("log_frequency", 1),
        probe_mode=probe_cfg.get("mode", "off"),
        probe_frequency=probe_cfg.get("frequency", 100),
        probe_offsets=probe_offsets,
        probe_offset_mode=probe_cfg.get("offset_mode", "auto"),
        probe_sharing=probe_cfg.get("sharing", "shared"),
        probe_slot_mode=probe_cfg.get("slot_mode", "surface"),
        probe_max_iters=probe_cfg.get("max_iters", 20),
        probe_l2=probe_cfg.get("l2", 1e-3),
        probe_lr=probe_cfg.get("lr", 1e-2),
        probe_train_frac=probe_cfg.get("train_frac", 0.8),
    )

    # Construct the trainer class directly. Hydra's `instantiate` cannot merge
    # kwargs containing `torch.nn.Module` (or dataclasses wrapping them) into a
    # DictConfig, so we resolve `_target_` manually and call the constructor.
    trainer_cls = get_class(cfg.trainer._target_)
    max_grad_norm = (
        cfg.trainer.max_grad_norm if "max_grad_norm" in cfg.trainer else None
    )
    trainer = trainer_cls(
        steps=cfg.trainer.steps,
        max_grad_norm=max_grad_norm,
        device=cfg.misc.device,
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler_cfg=scheduler_cfg,
        ngram_cfg=ngram_cfg,
        logging_cfg=logging_cfg,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path if resume_payload is not None else None,
        wandb_run_id=(writer.id if writer is not None else None),
        config_hash=current_cfg_hash,
        checkpoint_frequency=int(ckpt_cfg.get("frequency", 1)),
    )

    return trainer


def _save_wandb_stub(path: Path, wandb_run_id: str, cfg_hash: str) -> None:
    """Persist just the wandb id + config hash before training starts.

    Distinguished from a full checkpoint by the absence of a `student` key —
    `Trainer._maybe_resume` treats stub payloads as "no state to restore" but
    the runner still uses the id to reattach to the wandb run.
    """
    import os

    payload = {
        "current_step": 0,
        "wandb_run_id": wandb_run_id,
        "config_hash": cfg_hash,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

"""Trainer checkpointing for crash recovery + single-wandb-run resume.

Design (see conversation with the user for the full rationale):

- **Scope**: "resumable-only" reproducibility. On resume, training continues
  from the same weights/optimizer/scheduler/RNGs, but the specific batches
  drawn after resume may differ from what a crash-free run would have seen
  (because `RandomSampler` re-draws its permutation on `iter()` — see
  `src/runner/data.py`). Loss curves match in expectation.

- **Atomic writes**: save to `checkpoint.pt.tmp`, then `os.replace`. A crash
  mid-save never leaves a corrupt file that would silently spawn a new wandb
  run on next launch.

- **Single wandb run guarantee**: the checkpoint stores `wandb_run_id`. The
  runner uses it with `resume="must"` — either resumes the exact prior run
  or hard-fails. See `src/runner/main.py`.

- **Config-drift protection**: `config_hash` is set on the initial checkpoint
  and asserted on resume. Prevents accidental cross-experiment contamination
  in the same wandb run. ``misc.checkpoint.reset`` is excluded because it is
  an operation on an existing identity, not part of the training identity.
"""

import hashlib
import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

CHECKPOINT_FILENAME = "checkpoint.pt"
RESET_TOKEN_FILENAME = "reset-token"


def config_hash(cfg_container: Any) -> str:
    """SHA256 of the resolved config, for detecting drift across resumes.

    Accepts a plain container (from `OmegaConf.to_container(..., resolve=True)`).
    Sorted-key JSON dump makes the hash insensitive to key ordering.
    """
    payload = json.dumps(cfg_container, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def checkpoint_config_hash(cfg_container: Any) -> str:
    """Hash training identity while excluding the one-shot reset token.

    ``misc.checkpoint.reset`` controls what to do with an existing identity;
    it is not part of that identity. Removing it also keeps hashes compatible
    with checkpoints created before the reset option existed.
    """
    identity = deepcopy(cfg_container)
    try:
        identity["misc"]["checkpoint"].pop("reset", None)
    except (KeyError, TypeError, AttributeError):
        pass
    return config_hash(identity)


def _serialized_reset_token(token: Any) -> str:
    return json.dumps(token, sort_keys=True, default=str)


def reset_token_applied(ckpt_dir: Path, token: Any) -> bool:
    """Return whether this reset generation was already applied."""
    marker = ckpt_dir / RESET_TOKEN_FILENAME
    try:
        return marker.read_text() == _serialized_reset_token(token)
    except FileNotFoundError:
        return False


def apply_checkpoint_reset(ckpt_dir: Path, token: Any) -> list[Path]:
    """Clear resumable state and atomically record the applied generation.

    The caller must hold the config's worker lock. The lock file and directory
    are deliberately preserved so cleanup cannot race a newly-created lock
    inode. Unknown files are also preserved.
    """
    targets = (
        ckpt_dir / CHECKPOINT_FILENAME,
        ckpt_dir / f"{CHECKPOINT_FILENAME}.tmp",
        ckpt_dir / "done",
        ckpt_dir / "done.tmp",
    )
    removed = []
    for path in targets:
        if path.exists():
            path.unlink()
            removed.append(path)

    marker = ckpt_dir / RESET_TOKEN_FILENAME
    tmp = ckpt_dir / f"{RESET_TOKEN_FILENAME}.tmp"
    tmp.write_text(_serialized_reset_token(token))
    os.replace(tmp, marker)
    return removed


def _rng_state() -> Dict[str, Any]:
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    # `torch.set_rng_state` requires a CPU ByteTensor. `torch.load(...,
    # map_location=device)` moves every tensor in the payload — including
    # this RNG state — onto the device, so we force it back to CPU before
    # restoring. Same for the per-device CUDA RNG list.
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [s.cpu() for s in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def _metrics_state(trainer) -> Dict[str, Dict[str, Any]]:
    """Snapshot each metric's `__dict__`.

    Preserves reset/non-reset distinction: metrics that survive `_end_step`
    (MinMetric, ConstantLossMetric, ConstantAccuracyMetric) carry real values;
    others carry their post-reset defaults. On resume this is exactly the
    state we need — checkpointing happens *after* `_end_step`.
    """
    return {name: dict(m.__dict__) for name, m in trainer.metrics.items()}


def _restore_metrics(trainer, state: Dict[str, Dict[str, Any]]) -> None:
    for name, fields in state.items():
        if name not in trainer.metrics:
            # Metric was registered on the previous run but not on resume
            # (e.g. ngram phase already finished, keys were popped). Skip.
            continue
        trainer.metrics[name].__dict__.update(fields)


def save_checkpoint(
    trainer,
    path: Path,
    wandb_run_id: Optional[str],
    cfg_hash: Optional[str],
) -> None:
    """Atomically write trainer state to `path`.

    `wandb_run_id` and `cfg_hash` are metadata the runner injects — the
    trainer doesn't know about wandb identity or the raw config.
    """
    ngram_state = {
        name: {
            "model": ne.model.state_dict(),
            "optimizer": ne.optimizer.state_dict(),
        }
        for name, ne in trainer.ngram_evals.items()
    }
    payload = {
        "current_step": trainer.current_step,
        "student": trainer.student.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler_cfg.scheduler.state_dict(),
        "ngram": ngram_state,
        "metrics": _metrics_state(trainer),
        "history": dict(trainer.history),
        "rng": _rng_state(),
        "wandb_run_id": wandb_run_id,
        "config_hash": cfg_hash,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Read the raw payload. Restoring into a trainer is a separate step.

    Split from `restore_into_trainer` so the runner can inspect `wandb_run_id`
    and `config_hash` *before* building the trainer (needed to init wandb with
    the right id and to fail-fast on config drift).
    """
    return torch.load(path, map_location=device, weights_only=False)


def is_stub_payload(payload: Dict[str, Any]) -> bool:
    """A stub carries just wandb id + config hash — no trainer state.

    Written by the runner right after `wandb.init` so an early crash preserves
    the wandb run id. On resume, a stub means "reattach to the wandb run but
    start training from scratch"; a full checkpoint means "actually resume".
    """
    return "student" not in payload


def restore_into_trainer(trainer, payload: Dict[str, Any]) -> None:
    """Apply a loaded payload to a freshly-constructed trainer.

    Assumes `trainer._init_loop()` has already run (metrics registered, ngram
    evaluators built). Skips the constant `_dry_loop` — those values ride in
    the metrics snapshot.
    """
    # `_end_step(step=k)` saves with `current_step==k` and the training loop
    # increments to k+1 immediately after. To resume seamlessly we advance to
    # the *next* step, so `while current_step < steps` picks up where we left
    # off instead of redoing step k with a different (post-crash) batch.
    trainer.current_step = payload["current_step"] + 1
    trainer.student.load_state_dict(payload["student"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.scheduler_cfg.scheduler.load_state_dict(payload["scheduler"])

    for name, state in payload.get("ngram", {}).items():
        if name not in trainer.ngram_evals:
            continue
        trainer.ngram_evals[name].model.load_state_dict(state["model"])
        trainer.ngram_evals[name].optimizer.load_state_dict(state["optimizer"])

    _restore_metrics(trainer, payload["metrics"])
    trainer.history.update(payload["history"])
    _restore_rng(payload["rng"])

    logging.getLogger().info(
        f"Resumed from checkpoint at step {trainer.current_step}"
    )


def assert_config_matches(payload: Dict[str, Any], current_hash: str) -> None:
    """Hard-fail if the resolved config has drifted since the checkpoint.

    Prevents the "half the metrics are from experiment A, half from B in the
    same wandb run" failure mode. Start fresh by assigning a new non-null
    ``misc.checkpoint.reset`` token.
    """
    saved = payload.get("config_hash")
    if saved is None:
        # Legacy / initial checkpoint written before hash existed — trust it.
        return
    if saved != current_hash:
        raise RuntimeError(
            "Config hash mismatch: cannot resume this checkpoint into a "
            "modified config (would contaminate the wandb run with mixed "
            f"metrics). Saved hash: {saved[:12]}, current: {current_hash[:12]}. "
            "Set a new misc.checkpoint.reset token to start fresh."
        )

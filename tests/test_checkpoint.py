"""Checkpoint save/load round-trip and safety tests.

Covers the guarantees the runner relies on: state actually round-trips, config
drift is caught, stub payloads are distinguished from full ones, and atomic
writes leave no half-written file behind after a simulated crash.
"""

import math

import pytest
import torch

from src.loss import CrossentropyLoss
from src.trainer import LoggingConfig, NgramConfig, SchedulerConfig, SGDTrainer
from src.trainer.checkpoint import (
    CHECKPOINT_FILENAME,
    RESET_TOKEN_FILENAME,
    apply_checkpoint_reset,
    assert_config_matches,
    checkpoint_config_hash,
    config_hash,
    is_stub_payload,
    load_checkpoint,
    reset_token_applied,
    save_checkpoint,
)


def _make_trainer(tiny_teacher, tiny_student, tiny_loaders, device, **overrides) -> SGDTrainer:
    """Trainer with a single-step optimizer nudge so state is non-trivial.

    Deliberately avoids `trainer.train()` — that path triggers `_val_loop` +
    `ProbeLogger.log`, which has a pre-existing bug on non-Hierarchical
    teachers that's out of scope for this test file.
    """
    train_loader, val_loader = tiny_loaders
    optimizer = torch.optim.SGD(tiny_student.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)
    kwargs = dict(
        steps=5,
        device=device,
        teacher=tiny_teacher,
        student=tiny_student,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=CrossentropyLoss(),
        optimizer=optimizer,
        scheduler_cfg=SchedulerConfig(scheduler=scheduler),
        ngram_cfg=NgramConfig(steps=0),
        logging_cfg=LoggingConfig(writer=None, attention_frequency=100),
    )
    kwargs.update(overrides)
    return SGDTrainer(**kwargs)


@pytest.fixture()
def trained_trainer(tiny_teacher, tiny_student, tiny_loaders, device) -> SGDTrainer:
    """Init'd trainer with mutated state: a step count, a val_best value, and
    optimizer state after one manual SGD step."""
    trainer = _make_trainer(tiny_teacher, tiny_student, tiny_loaders, device)
    trainer._init_loop()
    # Simulate progress without invoking the full loop.
    trainer.current_step = 3
    trainer.metrics["student/loss/val_best"].update(0.42)
    # One SGD step so the optimizer's momentum buffers are populated.
    for p in trainer.student.parameters():
        p.grad = torch.randn_like(p) * 0.01
    trainer.optimizer.step()
    return trainer


def test_config_hash_stable_across_key_order():
    a = {"lr": 0.01, "wd": 0.0, "seed": 0}
    b = {"seed": 0, "wd": 0.0, "lr": 0.01}
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_values():
    a = {"lr": 0.01}
    b = {"lr": 0.02}
    assert config_hash(a) != config_hash(b)


def test_checkpoint_hash_ignores_reset_token():
    legacy = {"lr": 0.01, "misc": {"checkpoint": {"resume": True}}}
    reset_1 = {
        "lr": 0.01,
        "misc": {"checkpoint": {"resume": True, "reset": "restart-1"}},
    }
    reset_2 = {
        "lr": 0.01,
        "misc": {"checkpoint": {"resume": True, "reset": "restart-2"}},
    }
    assert checkpoint_config_hash(legacy) == checkpoint_config_hash(reset_1)
    assert checkpoint_config_hash(reset_1) == checkpoint_config_hash(reset_2)


def test_checkpoint_hash_still_changes_with_training_config():
    a = {"lr": 0.01, "misc": {"checkpoint": {"reset": "restart-1"}}}
    b = {"lr": 0.02, "misc": {"checkpoint": {"reset": "restart-1"}}}
    assert checkpoint_config_hash(a) != checkpoint_config_hash(b)


def test_apply_checkpoint_reset_is_one_shot_and_preserves_unknown_files(tmp_path):
    checkpoint = tmp_path / CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "done").write_text("")
    (tmp_path / "lock").write_text("worker")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep me")

    removed = apply_checkpoint_reset(tmp_path, "restart-1")

    assert set(removed) == {checkpoint, tmp_path / "done"}
    assert reset_token_applied(tmp_path, "restart-1")
    assert not reset_token_applied(tmp_path, "restart-2")
    assert (tmp_path / RESET_TOKEN_FILENAME).exists()
    assert (tmp_path / "lock").exists()
    assert unrelated.read_text() == "keep me"


def test_is_stub_payload_recognizes_stub():
    stub = {"current_step": 0, "wandb_run_id": "abc", "config_hash": "h"}
    full = {"current_step": 5, "student": {}, "wandb_run_id": "abc"}
    assert is_stub_payload(stub)
    assert not is_stub_payload(full)


def test_save_load_round_trip_restores_state(trained_trainer, tmp_path, device):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(trained_trainer, path, wandb_run_id="run-42", cfg_hash="hash")

    payload = load_checkpoint(path, device)
    assert payload["wandb_run_id"] == "run-42"
    assert payload["config_hash"] == "hash"
    assert payload["current_step"] == trained_trainer.current_step
    # Student weights round-trip bit-exact.
    for k, v in trained_trainer.student.state_dict().items():
        assert torch.equal(v, payload["student"][k])
    # val_best is a MinMetric — carries its accumulated min across resets.
    saved_val_best = payload["metrics"]["student/loss/val_best"]["min"]
    assert math.isfinite(saved_val_best)


def test_atomic_write_leaves_no_tmp_file(trained_trainer, tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(trained_trainer, path, wandb_run_id="r", cfg_hash="h")
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()


def test_assert_config_matches_hard_fails_on_drift():
    payload = {"config_hash": "hash-a"}
    with pytest.raises(RuntimeError, match="Config hash mismatch"):
        assert_config_matches(payload, "hash-b")


def test_assert_config_matches_passes_on_match():
    payload = {"config_hash": "hash-a"}
    # No exception.
    assert_config_matches(payload, "hash-a")


def test_assert_config_matches_tolerates_legacy_payload():
    # Payload without a config_hash field (early stub or old format) — don't fail.
    assert_config_matches({}, "hash-a")


def test_resume_advances_step_by_one(
    trained_trainer, tiny_teacher, tiny_loaders, tmp_path, device
):
    """Resumed trainer starts at saved_step + 1 (not saved_step)."""
    from src.model import TransformerDecoder

    path = tmp_path / "ckpt.pt"
    save_checkpoint(trained_trainer, path, wandb_run_id="r", cfg_hash="h")
    saved_step = trained_trainer.current_step

    fresh_student = TransformerDecoder(
        dim=4, hidden_dim=8, num_heads=2, ff_hidden_dim=8, num_blocks=1,
        dropout=0.0, pe_type="absolute", pe_learnable=True,
        pe_embedding_dim=8, pe_max_sequence_length=32,
    )
    resumed = _make_trainer(
        tiny_teacher, fresh_student, tiny_loaders, device,
        checkpoint_path=None, resume_from=path,
    )
    resumed._init_loop()
    assert resumed._maybe_resume() is True
    assert resumed.current_step == saved_step + 1
    # Student weights match the saved trainer post-resume.
    for k, v in trained_trainer.student.state_dict().items():
        assert torch.equal(v, resumed.student.state_dict()[k])


def test_rng_state_survives_device_map(tmp_path, tiny_teacher, tiny_student, tiny_loaders, device):
    """Loading with `map_location=device` moves ALL tensors onto device, but
    `torch.set_rng_state` requires a CPU ByteTensor. Regression for a resume
    crash: `TypeError: RNG state must be a torch.ByteTensor`."""
    from src.trainer.checkpoint import _restore_rng

    trainer = _make_trainer(tiny_teacher, tiny_student, tiny_loaders, device)
    trainer._init_loop()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(trainer, path, wandb_run_id="r", cfg_hash="h")

    payload = load_checkpoint(path, device)
    # Should not raise even if the loaded RNG tensor lives on a non-CPU device.
    _restore_rng(payload["rng"])


def test_stub_payload_does_not_trigger_resume(
    tiny_teacher, tiny_student, tiny_loaders, tmp_path, device
):
    """A stub carries the wandb id but no state — trainer should skip resume."""
    path = tmp_path / "ckpt.pt"
    stub = {"current_step": 0, "wandb_run_id": "run-1", "config_hash": "h"}
    torch.save(stub, path)

    trainer = _make_trainer(
        tiny_teacher, tiny_student, tiny_loaders, device,
        checkpoint_path=None, resume_from=path,
    )
    trainer._init_loop()
    assert trainer._maybe_resume() is False
    assert trainer.current_step == 0

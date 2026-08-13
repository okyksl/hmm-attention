import torch
import torch.nn.functional as F

from src.loss import CrossentropyLoss
from src.metrics import ConstantAccuracyMetric, ConstantLossMetric
from src.teachers import ChunkCode, MultiLevelHierarchicalTeacher
from src.trainer import MetricRegistry, TeacherEvaluator


def _make_data(teacher, batch=2, length=None):
    """Build a valid one-hot batch of shape (batch, teacher.context_length + 4, dim)."""
    T = length if length is not None else teacher.context_length + 4
    ids = torch.randint(0, teacher.dim, (batch, T))
    return torch.nn.functional.one_hot(ids, num_classes=teacher.dim).float()


# ---- construction ------------------------------------------------------------


def test_evaluator_for_non_ar_teacher_is_empty(device):
    ev = TeacherEvaluator(teacher=object(), device=device)  # non-ARTeacher
    assert ev.is_ar is False
    assert ev._teacher_by_k == {}
    assert ev.prefix_ks == []
    assert ev.metric_keys() == []


def test_evaluator_builds_lag_restricted_cache(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    # window=2 → cache holds only k=1 (k=window is a no-op, skipped).
    assert set(ev._teacher_by_k.keys()) == {1}
    assert ev.prefix_ks == [1, 2]


def test_metric_keys_include_kl_teacher_and_per_k(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    keys = set(ev.metric_keys())
    assert "teacher/kl/train" in keys
    assert "teacher/kl/val" in keys
    assert "teacher_k1/kl/train" in keys
    assert "teacher_k2/kl/val" in keys
    assert "teacher_k1/loss/train" in ev.loss_metric_keys()
    assert "teacher_k2/acc/val" in ev.acc_metric_keys()


def test_non_hierarchical_teacher_has_no_location_metrics(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    assert ev.location_names() == []
    assert not any("/level" in key for key in ev.loss_metric_keys())
    assert not any("/offset" in key for key in ev.acc_metric_keys())


def test_multilevel_metric_keys_cover_each_level_offset(tiny_teacher, device):
    teacher = MultiLevelHierarchicalTeacher(
        base_teacher=tiny_teacher,
        levels=[
            ChunkCode(in_dim=4, out_dim=3, size=2, chunk_seed=10),
            ChunkCode(in_dim=3, out_dim=4, size=3, chunk_seed=11),
        ],
    )
    ev = TeacherEvaluator(teacher, device=device)

    assert ev.location_names() == [
        "level0/offset0",
        "level0/offset1",
        "level1/offset0",
        "level1/offset1",
        "level1/offset2",
    ]
    assert "teacher/level0/offset1/loss/train" in ev.loss_metric_keys()
    assert "teacher_k1/level1/offset2/acc/val" in ev.acc_metric_keys()


# ---- _align_data -------------------------------------------------------------


def test_align_data_trims_by_context_length_diff(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    restricted = ev._teacher_by_k[1]
    trimmed = ev._align_data(data, restricted)
    diff = tiny_teacher.context_length - restricted.context_length
    assert trimmed.shape[1] == data.shape[1] - diff


def test_align_data_noop_when_full_teacher(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    assert ev._align_data(data, tiny_teacher).shape == data.shape


# ---- run ---------------------------------------------------------------------


def test_run_unnormalized_returns_log_probs(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    out, log_probs, targets = ev.run(data, prefix=-1, normalize=False)
    # Unnormalized: out is log-probs (same object as log_probs).
    assert torch.equal(out, log_probs)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones_like(log_probs[..., 0]), atol=1e-5)


def test_run_normalized_returns_probs_and_log_probs(tiny_teacher, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    data = _make_data(tiny_teacher)
    probs, log_probs, _ = ev.run(data, prefix=-1, normalize=True)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-5)
    assert torch.allclose(probs, log_probs.exp(), atol=1e-5)


def test_hierarchical_location_metrics_match_manual_masks(
    tiny_hier_teacher, device
):
    ev = TeacherEvaluator(tiny_hier_teacher, device=device)
    reg = MetricRegistry()
    for key in ev.loss_metric_keys():
        reg.register(key, ConstantLossMetric())
    for key in ev.acc_metric_keys():
        reg.register(key, ConstantAccuracyMetric(k=1))

    n_base = tiny_hier_teacher.base_teacher.burn_in + 3
    data = tiny_hier_teacher.sample_surface_prefix(
        n_base * tiny_hier_teacher.total, batch_size=3
    )
    _, log_probs, targets = ev.run(data, prefix=-1, normalize=False)
    ev.update_loss_acc_metrics(
        log_probs,
        targets,
        prefix=-1,
        split="train",
        metrics=reg,
        loss_fn=CrossentropyLoss(),
    )

    for offset in range(tiny_hier_teacher.levels[0].size):
        mask = torch.arange(log_probs.shape[1]) % tiny_hier_teacher.total == offset
        expected_loss = F.cross_entropy(
            log_probs[:, mask, :].reshape(-1, tiny_hier_teacher.dim),
            targets[:, mask, :].reshape(-1, tiny_hier_teacher.dim),
        ).item()
        expected_acc = (
            log_probs[:, mask, :].argmax(dim=-1)
            == targets[:, mask, :].argmax(dim=-1)
        ).float().mean().item()
        stub = f"teacher/level0/offset{offset}"
        assert abs(reg[f"{stub}/loss/train"].compute() - expected_loss) < 1e-6
        assert abs(reg[f"{stub}/acc/train"].compute() - expected_acc) < 1e-6


# ---- update_kl_metrics -------------------------------------------------------


def test_update_kl_metrics_populates_registry(tiny_teacher, tiny_student, device):
    ev = TeacherEvaluator(tiny_teacher, device=device)
    reg = MetricRegistry()
    for key in ev.metric_keys():
        from src.metrics import LossMetric
        reg.register(key, LossMetric())

    data = _make_data(tiny_teacher)
    # Student output shape must match teacher output shape.
    out_teacher, _, _ = ev.run(data, prefix=-1, normalize=False)
    fake_student_out = out_teacher.clone()  # perfect student → KL = 0.

    ev.update_kl_metrics(fake_student_out, data, split="train", metrics=reg)
    # KL of a perfect student is mathematically 0, but sum(p * (log p - log q))
    # with p == q accumulates roundoff that can land either side of zero, so the
    # bound needs a tolerance rather than a bare `>= 0`.
    assert reg["teacher/kl/train"].compute() >= -1e-6
    assert reg["teacher_k1/kl/train"].compute() >= -1e-6


# ---- adaptive (attention) teacher --------------------------------------------


def _adaptive_teacher(dim=6, burn_in=2, seed=0):
    from src.teachers import AttentionARTeacher

    return AttentionARTeacher(
        dim=dim, hidden_dim=12, unbounded=True, burn_in=burn_in, seed=seed
    )


def _adaptive_data(teacher, batch=2, extra=4):
    T = teacher.burn_in + extra
    ids = torch.randint(0, teacher.dim, (batch, T))
    return torch.nn.functional.one_hot(ids, num_classes=teacher.dim).float()


def test_adaptive_teacher_has_no_lag_cache(device):
    ev = TeacherEvaluator(_adaptive_teacher(), device=device)
    assert ev.is_ar is True
    # Attention has no lag structure: only the full-teacher metric, no crash on
    # with_lag_restriction, no context_length arithmetic.
    assert ev._teacher_by_k == {}
    assert ev.prefix_ks == []
    assert ev.context_names() == ["teacher"]
    assert "teacher/kl/train" in ev.metric_keys()


def test_adaptive_teacher_align_data_noop(device):
    t = _adaptive_teacher(burn_in=2)
    ev = TeacherEvaluator(t, device=device)
    data = _adaptive_data(t)
    assert ev._align_data(data, t).shape == data.shape  # burn_ins match -> no trim


def test_adaptive_teacher_run_and_kl(device):
    t = _adaptive_teacher(burn_in=2)
    ev = TeacherEvaluator(t, device=device)
    reg = MetricRegistry()
    from src.metrics import LossMetric

    for key in ev.metric_keys():
        reg.register(key, LossMetric())

    data = _adaptive_data(t)
    out, log_probs, _ = ev.run(data, prefix=-1, normalize=False)
    assert out.shape == (2, data.shape[1] - t.burn_in, t.dim)
    assert torch.allclose(
        log_probs.exp().sum(-1), torch.ones_like(log_probs[..., 0]), atol=1e-5
    )
    ev.update_kl_metrics(out.clone(), data, split="train", metrics=reg)
    assert reg["teacher/kl/train"].compute() >= -1e-6  # perfect student -> KL ~ 0

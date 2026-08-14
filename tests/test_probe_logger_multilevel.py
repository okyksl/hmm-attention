"""Unit tests for the multi-level path of `ProbeLogger`."""
import math

import pytest
import torch
import torch.nn.functional as F

from src.model.decoder import TransformerDecoder
from src.teachers import ChunkCode, LinearARTeacher, MultiLevelHierarchicalTeacher
from src.trainer.config import LoggingConfig
from src.trainer.probe_logger import ProbeLogger
from src.trainer.teacher_eval import TeacherEvaluator


def _make_teacher(k=(2, 3), dims=(6, 4, 8), tuples=(1, 1), base_window=2, seed=0):
    base = LinearARTeacher.from_parameters(
        dim=dims[0], span_lengths=[1] * base_window, spectrum={"rank": dims[0]},
        window=base_window, lag_spectrum={"law": "geometric", "decay": 1.7}, scale=10.0,
    )
    levels = [
        ChunkCode(in_dim=dims[l], out_dim=dims[l + 1], size=k[l],
                  num_tuples=tuples[l], chunk_seed=10 + l)
        for l in range(len(k))
    ]
    return MultiLevelHierarchicalTeacher(base_teacher=base, levels=levels)


def _make_student(dim, hidden_dim, num_blocks=2):
    return TransformerDecoder(
        dim=dim, hidden_dim=hidden_dim, num_heads=1, ff_hidden_dim=hidden_dim,
        num_blocks=num_blocks, dropout=0.0, pe_type="none",
        encoder_layer=True, decoder_layer=True, layer_normalization=False,
    )


def _logger(teacher, mode="warm_start", **cfg_kw):
    student = _make_student(dim=teacher.dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode=mode, **cfg_kw)
    return ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg), student


class _FakeWriter:
    def __init__(self):
        self.logged = {}

    def log(self, metrics, step):
        self.logged.update(metrics)


def test_enabled_and_level_specs():
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8))
    pl, _ = _logger(teacher)
    assert pl.enabled
    assert pl.num_latent_levels == 2
    assert pl.num_levels == 3
    assert pl.sharing == "shared"
    assert pl.slot_mode == "surface"
    assert pl.level_arity == [6, 3, 1]
    assert pl.level_coarse_arity == [2, 3, 1]
    assert pl.level_alphabet == [6, 4, 8]  # base + mid + surface
    assert pl.level_spans == [6, 3, 1, 1]
    # Common surface horizons become progressively more level units.
    assert pl.offsets_by_level == [
        [-2, -1, 0, 1],
        [-4, -3, -2, -1, 0, 1, 2],
        list(range(-12, 7)),
    ]


@pytest.mark.parametrize(
    ("slot_mode", "expected"),
    [("surface", [6, 3, 1]), ("coarse", [2, 3, 1])],
)
def test_probe_and_teacher_eval_share_slot_layout(slot_mode, expected):
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8))
    pl, _ = _logger(teacher, hierarchy_slot_mode=slot_mode)
    teacher_eval = TeacherEvaluator(
        teacher, device=torch.device("cpu"), slot_mode=slot_mode
    )

    assert pl.level_arity == expected
    assert teacher_eval.slot_counts == expected


def test_gather_level_digit_and_targets():
    teacher = _make_teacher(k=(2, 3))
    pl, _ = _logger(teacher)
    total = teacher.total  # 6

    N, n_chunks = 3, 6
    data = teacher.sample_surface_prefix(n_chunks * total, batch_size=N)
    T = data.shape[1] - 1
    residual = torch.randn(N, T, 8)

    # Bottom level (level 1): span=3, arity 3. slot=1 -> positions t % 3 == 1.
    labels1 = pl._decode_level(data, 1)
    X, y, belief, bvalid = pl._gather_level(residual, labels1, None, level=1, slot=1, offset=0)
    tsel = torch.arange(T)[torch.arange(T) % 3 == 1]
    assert X.shape == (N * tsel.numel(), 8)
    assert y.shape == (N * tsel.numel(),)
    assert belief is None  # no belief tensor passed

    # Top level (level 0): full-resolution slot 0 is the first surface phase.
    labels0 = pl._decode_level(data, 0)
    X0, y0, _, _ = pl._gather_level(residual, labels0, None, level=0, slot=0, offset=0)
    tsel0 = torch.arange(T)[torch.arange(T) % 6 == 0]
    assert X0.shape[0] == N * tsel0.numel()

    # Terminal surface level: span=1, one slot, and offsets are surface-token
    # offsets. The final +1 target is out of bounds and is dropped.
    labels2 = pl._decode_level(data, 2)
    assert torch.equal(labels2, data.argmax(dim=-1))
    X2, y2, _, _ = pl._gather_level(
        residual, labels2, None, level=2, slot=0, offset=1
    )
    assert X2.shape == (N * T, 8)
    assert torch.equal(y2.reshape(N, T), labels2[:, 1 : T + 1])


@pytest.mark.parametrize(
    ("sharing", "slot_mode", "expected_slots", "expected_probe_count"),
    [
        ("shared", "surface", [6, 3, 1], 3),
        ("shared", "coarse", [2, 3, 1], 3),
        ("per_slot", "surface", [6, 3, 1], 10),
        ("per_slot", "coarse", [2, 3, 1], 6),
    ],
)
def test_sharing_and_slot_modes_are_independent(
    sharing, slot_mode, expected_slots, expected_probe_count
):
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8))
    pl, _ = _logger(
        teacher,
        mode="sgd",
        probe_offsets=[0],
        probe_sharing=sharing,
        hierarchy_slot_mode=slot_mode,
    )
    data = teacher.sample_surface_prefix(4 * teacher.total, batch_size=3)
    residual = torch.randn(3, data.shape[1] - 1, 16)
    pl._current_data = data
    pl._current_residuals = ((0, residual),)

    pl.sgd_step()

    assert pl.level_arity == expected_slots
    assert len(pl.probes) == expected_probe_count
    fitted_slots = {key[2] for key in pl.probes}
    if sharing == "shared":
        assert fitted_slots == {None}
    else:
        assert None not in fitted_slots


def test_auto_grid_logs_only_offsets_resolved_for_each_level():
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8), base_window=2)
    writer = _FakeWriter()
    student = _make_student(dim=teacher.dim, hidden_dim=16)
    cfg = LoggingConfig(
        writer=None, probe_mode="sgd", probe_frequency=1,
        probe_train_frac=0.5,
    )
    pl = ProbeLogger(writer=writer, teacher=teacher, student=student, cfg=cfg)
    data = teacher.sample_surface_prefix(5 * teacher.total, batch_size=3)
    pl._current_data = data
    pl._current_residuals = ((0, torch.randn(3, data.shape[1] - 1, 16)),)
    pl.collect_val_batch()

    pl.log(step=0, split="val")

    keys = set(writer.logged)
    assert any("/level2/slot0/k+6/acc/val" in key for key in keys)
    assert any("/level1/slot0/k+2/acc/val" in key for key in keys)
    assert any("/level0/slot0/k+1/acc/val" in key for key in keys)
    assert not any("/level0/" in key and "/k+2/" in key for key in keys)
    assert not any("/level1/" in key and "/k+3/" in key for key in keys)


def test_bayes_ceiling_retention_and_refinement():
    pl, _ = _logger(_make_teacher())
    C = 4
    y = torch.tensor([0, 1, 2, 3])
    logits = torch.randn(4, C)

    # Retention (offset < 0): deterministic optimum.
    acc, nll, excess = pl._bayes_ceiling(-1, None, None, slice(0, 4), logits, y)
    assert acc == 1.0 and nll == 0.0
    assert abs(excess - F.cross_entropy(logits, y).item()) < 1e-6

    # Refinement (offset 0), belief = delta on truth -> acc 1, nll ~0.
    bvalid = torch.ones(4, dtype=torch.bool)
    delta = F.one_hot(y, C).float().clamp(min=1e-30).log()
    acc, nll, excess = pl._bayes_ceiling(0, delta, bvalid, slice(0, 4), logits, y)
    assert acc == 1.0 and nll < 1e-3

    # Refinement, uniform belief -> acc chance, nll = log C.
    uniform = torch.full((4, C), 1.0 / C).log()
    acc_u, nll_u, _ = pl._bayes_ceiling(0, uniform, bvalid, slice(0, 4), logits, y)
    assert abs(nll_u - math.log(C)) < 1e-4

    # Planning (offset > 0): deferred.
    assert pl._bayes_ceiling(1, None, None, slice(0, 4), logits, y) is None


@pytest.mark.parametrize(
    ("level", "slot", "completes_unit"),
    [
        (0, 2, False),  # top latent, still inside its six-surface-token unit
        (0, 5, True),   # top latent boundary
        (1, 1, False),  # intermediate latent, still inside its three-token unit
        (1, 2, True),   # intermediate latent boundary
        (2, 0, True),   # observed terminal surface token
    ],
)
def test_current_bayes_belief_uses_information_through_residual_position(
    level, slot, completes_unit
):
    """k0 predicts the unit covering t from a residual that has consumed x_t.

    At unfinished units this is the teacher's pre-token posterior at t+1. At
    each level boundary it must instead be a delta on the just-completed unit;
    the posterior at t+1 would refer to a different unit there.
    """
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8), base_window=2)
    pl, _ = _logger(teacher)
    data = teacher.sample_surface_prefix(
        (teacher.base_teacher.burn_in + 3) * teacher.total, batch_size=2
    )
    residual = torch.randn(2, data.shape[1] - 1, 5)
    labels = pl._decode_level(data, level)
    pre_token_belief = (
        teacher.latent_beliefs(data)[level]
        if level < teacher.num_levels
        else None
    )

    _, y, belief, belief_valid = pl._gather_level(
        residual, labels, pre_token_belief, level, slot, offset=0
    )

    span = pl.level_spans[level]
    t = torch.arange(residual.shape[1])
    tsel = t[t % span == slot]
    b = belief.reshape(data.shape[0], tsel.numel(), -1)
    bv = belief_valid.reshape(data.shape[0], tsel.numel())
    y_grid = y.reshape(data.shape[0], tsel.numel())

    if completes_unit:
        assert bool(bv.all())
        assert torch.equal(b.argmax(dim=-1), y_grid)
        assert torch.allclose(
            b.exp(), F.one_hot(y_grid, b.shape[-1]).float(), atol=1e-6
        )
    else:
        next_idx = tsel + 1 - teacher.burn_in
        expected_valid = (next_idx >= 0) & (next_idx < pre_token_belief.shape[1])
        assert torch.equal(bv[0], expected_valid)
        assert torch.equal(bv, expected_valid.expand_as(bv))
        assert torch.allclose(
            b[:, expected_valid, :],
            pre_token_belief[:, next_idx[expected_valid], :],
        )


def test_current_bayes_alignment_with_coarse_multilevel_slots():
    """A coarse slot can contain unfinished and boundary surface phases."""
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8), base_window=2)
    pl, _ = _logger(teacher, hierarchy_slot_mode="coarse")
    data = teacher.sample_surface_prefix(
        (teacher.base_teacher.burn_in + 3) * teacher.total, batch_size=2
    )
    residual = torch.randn(2, data.shape[1] - 1, 5)
    level = 0
    slot = 1
    labels = pl._decode_level(data, level)
    pre_token_belief = teacher.latent_beliefs(data)[level]

    _, y, belief, belief_valid = pl._gather_level(
        residual, labels, pre_token_belief, level, slot, offset=0
    )

    span = pl.level_spans[level]
    child_span = pl.level_spans[level + 1]
    t = torch.arange(residual.shape[1])
    tsel = t[(t % span) // child_span == slot]
    complete = tsel % span == span - 1
    next_idx = tsel + 1 - teacher.burn_in
    expected_valid = ((next_idx >= 0) & (next_idx < pre_token_belief.shape[1])) | complete
    b = belief.reshape(data.shape[0], tsel.numel(), -1)
    bv = belief_valid.reshape(data.shape[0], tsel.numel())
    y_grid = y.reshape(data.shape[0], tsel.numel())

    assert torch.equal(bv, expected_valid.expand_as(bv))
    assert torch.allclose(
        b[:, ~complete & expected_valid, :],
        pre_token_belief[:, next_idx[~complete & expected_valid], :],
    )
    assert torch.equal(b[:, complete, :].argmax(dim=-1), y_grid[:, complete])


def test_probe_and_bayes_metrics_use_identical_valid_eval_rows():
    teacher = _make_teacher()
    pl, _ = _logger(teacher, probe_train_frac=0.5)
    num_classes = pl.level_alphabet[0]
    probe = torch.nn.Linear(3, num_classes)
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.zero_()
        probe.bias[0] = 1.0  # always predict class 0

    X = torch.zeros(6, 3)
    # Eval rows are [3, 4, 5]. Row 3 would be correct, but has no Bayes belief;
    # the two comparable rows are both deliberately incorrect for the probe.
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    belief = F.one_hot(y, num_classes).float().clamp(min=1e-30).log()
    belief_valid = torch.tensor([True, True, True, False, True, True])

    values = pl._evaluate_probe(
        probe, X, y, belief, belief_valid, offset=0
    )

    assert values["n"] == 2
    assert values["acc"] == 0.0
    assert values["bayes_acc"] == 1.0
    assert values["excess_nll"] == pytest.approx(
        values["nll"] - values["bayes_nll"]
    )


def test_probe_recovers_current_unit_per_level():
    """Residuals = one-hot(true current-unit id) -> the probe decodes it (offset 0)."""
    torch.manual_seed(0)
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8))
    pl, _ = _logger(teacher, mode="warm_start", probe_max_iters=50, probe_l2=1e-4)

    N, n_chunks = 8, 6
    data = teacher.sample_surface_prefix(n_chunks * teacher.total, batch_size=N)
    for level in range(pl.num_levels):
        labels = pl._decode_level(data, level)  # (N, L_level)
        span_l = pl.level_spans[level]
        T = data.shape[1]
        # Residual at position t = one-hot(current level-`level` unit covering t).
        unit_at_pos = labels[:, torch.arange(T) // span_l]  # (N, T)
        residual = F.one_hot(unit_at_pos, num_classes=pl.level_alphabet[level]).float()

        slot = min(1, pl.level_arity[level] - 1)
        X, y, _, _ = pl._gather_level(
            residual, labels, None, level, slot=slot, offset=0
        )
        probe, _ = pl._ensure_probe(
            layer=1, slot=slot, offset=0, in_dim=X.shape[-1], device=X.device,
            need_opt=False, level=level,
        )
        pl._lbfgs_fit(probe, X, y)
        acc = (probe(X).argmax(-1) == y).float().mean().item()
        assert acc > 0.98, f"level {level}: expected near-perfect decode, got {acc:.3f}"


def test_end_to_end_log_emits_per_level_metrics():
    torch.manual_seed(0)
    teacher = _make_teacher(k=(2, 3), dims=(6, 4, 8))
    writer = _FakeWriter()
    student = _make_student(dim=teacher.dim, hidden_dim=16)
    cfg = LoggingConfig(
        writer=None, probe_mode="warm_start", probe_frequency=1,
        probe_offsets=[-1, 0],
    )
    pl = ProbeLogger(writer=writer, teacher=teacher, student=student, cfg=cfg)

    N, n_chunks = 4, 6
    data = teacher.sample_surface_prefix(n_chunks * teacher.total, batch_size=N)

    pl.before_forward("val")
    _ = student(data[:, :-1, :])
    pl.after_forward(data)
    pl.collect_val_batch()
    pl.log(step=0, split="val")

    keys = writer.logged.keys()
    assert any("level0" in k for k in keys)
    assert any("level1" in k for k in keys)
    assert any("level2" in k for k in keys)
    assert any("bayes_acc" in k for k in keys)
    assert any("excess_nll" in k for k in keys)
    # Retention ceilings are exactly optimal.
    retention = [k for k in keys if "bayes_acc" in k and "k-1" in k]
    assert retention and all(writer.logged[k] == 1.0 for k in retention)
    surface_current = [
        k for k in keys
        if "/level2/" in k and "/k0/bayes_acc/" in k
    ]
    assert surface_current and all(writer.logged[k] == 1.0 for k in surface_current)
    # Accuracies are valid probabilities.
    accs = [v for k, v in writer.logged.items() if k.endswith("/acc/val")]
    assert accs and all(0.0 <= v <= 1.0 for v in accs)
    # Summary heatmaps are rebuilt from these scalars after training.
    assert not any("heatmap" in key for key in keys)
    assert all(isinstance(value, (int, float)) for value in writer.logged.values())

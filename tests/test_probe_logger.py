"""Unit tests for `src.trainer.probe_logger.ProbeLogger`."""

import torch
import torch.nn.functional as F

from src.model.decoder import TransformerDecoder
from src.teachers import HierarchicalTeacher, LinearARTeacher
from src.trainer.config import LoggingConfig
from src.trainer.probe_logger import ProbeLogger, _offset_name


def _make_teacher(
    hidden_dim: int = 6,
    chunk_dim: int = 8,
    chunk_size: int = 2,
    window: int = 3,
    seed: int = 0,
) -> HierarchicalTeacher:
    base = LinearARTeacher.from_parameters(
        dim=hidden_dim,
        span_lengths=[1] * window,
        spectrum={'rank': hidden_dim},
        window=window,
        lag_spectrum={'law': 'geometric', 'decay': 1.7},
        scale=10.0,
    )
    return HierarchicalTeacher(
        base_teacher=base,
        chunk_dim=chunk_dim,
        chunk_size=chunk_size,
        chunk_seed=seed,
    )


def _make_student(dim: int, hidden_dim: int, num_blocks: int = 2) -> TransformerDecoder:
    return TransformerDecoder(
        dim=dim,
        hidden_dim=hidden_dim,
        num_heads=1,
        ff_hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=0.0,
        pe_type="none",
        encoder_layer=True,
        decoder_layer=True,
        layer_normalization=False,
    )


def _sample_data(teacher: HierarchicalTeacher, batch: int, num_chunks: int) -> torch.Tensor:
    n_surf = num_chunks * teacher.chunk_size
    return torch.stack(
        [teacher.sample_surface_prefix(n_surf) for _ in range(batch)]
    )


def test_disabled_when_mode_off():
    teacher = _make_teacher()
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="off")
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)
    assert pl.enabled is False
    # No hooks installed → nothing to clean up.
    assert student.pos_encoder._forward_hooks == {}


def test_disabled_when_wrong_teacher():
    student = _make_student(dim=8, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start")
    pl = ProbeLogger(writer=None, teacher=torch.nn.Linear(1, 1), student=student, cfg=cfg)
    assert pl.enabled is False


def test_offsets_default_from_teacher():
    teacher = _make_teacher(window=3)
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start", probe_offsets=None)
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)
    # base_teacher.context_length == 3 → [-3, -2, -1, 0, +1]
    assert pl.offsets == [-3, -2, -1, 0, 1]


def test_single_code_level_also_probes_terminal_surface_level():
    teacher = _make_teacher(hidden_dim=6, chunk_dim=8, chunk_size=2)
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start")

    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)

    assert pl.num_latent_levels == 1
    assert pl.num_levels == 2
    assert pl.level_arity == [2, 1]
    assert pl.level_alphabet == [6, 8]
    assert pl.level_spans == [2, 1, 1]
    data = _sample_data(teacher, batch=2, num_chunks=4)
    assert torch.equal(pl._decode_level(data, 1), data.argmax(dim=-1))


def test_offset_metric_names_are_unambiguous():
    assert _offset_name(-3) == "k-3"
    assert _offset_name(-1) == "k-1"
    assert _offset_name(0) == "k0"
    assert _offset_name(1) == "k+1"


def test_hooks_installed_and_removable():
    teacher = _make_teacher()
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start")
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)
    assert pl.enabled
    # 1 hook on pos_encoder + one per transformer block.
    n_expected = 1 + student.num_blocks
    assert len(pl._handles) == n_expected
    pl.remove_hooks()
    assert pl._handles == []


def test_gather_positions_and_labels_alignment():
    teacher = _make_teacher(chunk_size=2)
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start")
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)

    N, T, D = 4, 10, 8
    L_h = T // 2
    residual = torch.randn(N, T, D)
    labels = torch.randint(0, teacher.hidden_dim, (N, L_h))

    # slot=0, offset=0 → positions 0,2,4,6,8; chunks 0..4; all valid.
    X, y, _, _ = pl._gather_level(residual, labels, None, level=0, slot=0, offset=0)
    assert X.shape == (N * 5, D)
    assert y.shape == (N * 5,)

    # slot=1, offset=+1 → positions 1,3,5,7,9 → chunks 0..4 → +1 → 1..5.
    # Chunk 5 out of bounds (L_h=5), so 4 valid → drop last.
    X, y, _, _ = pl._gather_level(residual, labels, None, level=0, slot=1, offset=1)
    assert X.shape == (N * 4, D)

    # offset = -L_h → nothing valid.
    X, y, _, _ = pl._gather_level(residual, labels, None, level=0, slot=0, offset=-L_h)
    assert X is None


def test_warm_start_recovers_perfect_features():
    """Given residuals that are one-hot(hidden_id), LBFGS should recover the
    identity map and hit near-100% accuracy in a single fit."""
    torch.manual_seed(0)
    teacher = _make_teacher(chunk_size=2)
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=teacher.hidden_dim)
    cfg = LoggingConfig(
        writer=None, probe_mode="warm_start", probe_max_iters=50, probe_l2=1e-4
    )
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)

    # Construct residuals = one-hot(hidden id at containing chunk), so probe
    # for (any layer, any slot, offset=0) should trivially learn identity.
    N, num_chunks = 8, 6
    data = _sample_data(teacher, batch=N, num_chunks=num_chunks)
    labels = pl._decode_level(data, 0)  # (N, L_h)
    T = num_chunks * teacher.chunk_size
    residual = F.one_hot(
        labels.repeat_interleave(teacher.chunk_size, dim=1), num_classes=teacher.hidden_dim
    ).float()  # (N, T, hidden_dim)

    X, y, _, _ = pl._gather_level(residual, labels, None, level=0, slot=1, offset=0)
    probe, _ = pl._ensure_probe(
        layer=1, slot=1, offset=0, in_dim=X.shape[-1], device=X.device, need_opt=False, level=0
    )
    pl._lbfgs_fit(probe, X, y)
    with torch.no_grad():
        acc = (probe(X).argmax(dim=-1) == y).float().mean().item()
    assert acc > 0.98, f"expected near-perfect fit, got acc={acc:.3f}"


def test_sgd_step_improves_over_random_init():
    """A few Adam steps on perfect features should raise accuracy meaningfully
    above the random-init baseline."""
    torch.manual_seed(0)
    teacher = _make_teacher(chunk_size=2)
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=teacher.hidden_dim)
    cfg = LoggingConfig(writer=None, probe_mode="sgd", probe_lr=0.1)
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)

    N, num_chunks = 32, 6
    data = _sample_data(teacher, batch=N, num_chunks=num_chunks)
    labels = pl._decode_level(data, 0)
    T = num_chunks * teacher.chunk_size
    residual = F.one_hot(
        labels.repeat_interleave(teacher.chunk_size, dim=1), num_classes=teacher.hidden_dim
    ).float()

    # Simulate per-step captures.
    pl._current_residuals = tuple([(0, residual), (1, residual), (2, residual)])
    pl._current_data = data

    # Baseline: the default shared probe, evaluated at level0/slot1/k0.
    X, y, _, _ = pl._gather_level(residual, labels, None, level=0, slot=1, offset=0)
    probe, _ = pl._ensure_probe(
        1, None, 0, X.shape[-1], X.device, need_opt=True, level=0
    )
    with torch.no_grad():
        acc0 = (probe(X).argmax(dim=-1) == y).float().mean().item()

    for _ in range(200):
        pl.sgd_step()

    with torch.no_grad():
        acc1 = (probe(X).argmax(dim=-1) == y).float().mean().item()
    assert acc1 > acc0 + 0.3, f"sgd didn't improve enough: {acc0:.3f} -> {acc1:.3f}"
    assert acc1 > 0.9, f"expected sgd to reach high acc on perfect features, got {acc1:.3f}"


def test_capture_gating_train_vs_val():
    """warm_start mode must NOT capture during training forwards."""
    teacher = _make_teacher()
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start")
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)

    pl.before_forward("train")
    assert pl._capture_enabled is False

    pl.before_forward("val")
    assert pl._capture_enabled is True

    # sgd mode captures both.
    cfg2 = LoggingConfig(writer=None, probe_mode="sgd")
    pl2 = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg2)
    pl2.before_forward("train")
    assert pl2._capture_enabled is True


def test_end_to_end_val_forward_captures_residuals():
    """Real forward through the student, from val context, populates captures
    and produces a well-shaped residual snapshot."""
    torch.manual_seed(0)
    teacher = _make_teacher(chunk_size=2)
    student = _make_student(dim=teacher.chunk_dim, hidden_dim=16, num_blocks=2)
    cfg = LoggingConfig(writer=None, probe_mode="warm_start")
    pl = ProbeLogger(writer=None, teacher=teacher, student=student, cfg=cfg)

    N, num_chunks = 3, 5
    data = _sample_data(teacher, batch=N, num_chunks=num_chunks)
    T = num_chunks * teacher.chunk_size - 1  # student sees data[:, :-1, :]

    pl.before_forward("val")
    _ = student(data[:, :-1, :])
    pl.after_forward(data)

    assert len(pl._current_residuals) == 1 + student.num_blocks  # pos_enc + blocks
    for layer_idx, r in pl._current_residuals:
        assert r.shape[0] == N
        assert r.shape[1] == T

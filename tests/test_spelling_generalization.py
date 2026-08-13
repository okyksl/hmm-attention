"""Functional tests for the spelling-generalization analysis code.

These check the *code* — tuple recovery, the leave-one-tuple-out plumbing runs
and returns well-formed results, edge cases, and the helper invariants. The
scientific question (does a trained model actually abstract?) lives in
notebooks/m_generalization.ipynb, not here.
"""
import torch

from src.analysis.spelling_generalization import (
    analyze_residuals,
    cross_spelling_divergence,
    render_spellings,
    spelling_generalization,
    summarize,
)
from src.model.decoder import TransformerDecoder
from src.teachers import ChunkCode, LinearARTeacher, MultiLevelHierarchicalTeacher


def _teacher(sizes=(2, 3), dims=(6, 4, 8), tuples=(1, 3), base_window=2, seed=0):
    torch.manual_seed(seed)
    base = LinearARTeacher.from_parameters(
        dim=dims[0], span_lengths=[1] * base_window, window=base_window,
        spectrum={"rank": dims[0]}, lag_spectrum={"law": "geometric", "decay": 1.7},
        scale=10.0,
    )
    levels = [
        ChunkCode(in_dim=dims[l], out_dim=dims[l + 1], size=sizes[l],
                  num_tuples=tuples[l], chunk_seed=10 + l)
        for l in range(len(sizes))
    ]
    return MultiLevelHierarchicalTeacher(base_teacher=base, levels=levels)


def test_tuple_decode_recovers_spelling():
    """decode_level(return_tuple=True) recovers the exact tuple used per chunk."""
    teacher = _teacher(tuples=(1, 3))
    torch.manual_seed(1)
    base_ids = torch.randint(0, teacher.base_teacher.dim, (6,))
    ids, chosen_bottom = base_ids, None
    for l, level in enumerate(teacher.levels):
        tup_ids = torch.randint(0, level.num_tuples, ids.shape)
        chunks = level.sample(ids, tuple_ids=tup_ids)
        if l == teacher.num_levels - 1:
            surface = chunks.reshape(1, -1, level.out_dim)
            chosen_bottom = tup_ids
        ids = chunks.argmax(-1).reshape(-1)
    _, decoded = teacher.decode_level(surface, teacher.num_levels - 1, return_tuple=True)
    assert torch.equal(decoded.squeeze(0), chosen_bottom)


def test_analyze_returns_wellformed_results():
    """Leave-one-tuple-out plumbing runs; only M>1 levels; valid ranges; gaps."""
    teacher = _teacher(sizes=(2, 3), dims=(6, 4, 8), tuples=(1, 3))
    torch.manual_seed(0)
    N, n_base = 16, 10
    data = teacher.sample_surface_prefix(n_base * teacher.total, batch_size=N)
    residuals = {2: torch.randn(N, data.shape[1] - 1, 8)}

    results = analyze_residuals(teacher, residuals, data, offsets=[-1, 0], seed=0)
    assert results, "expected some results for the M>1 level"
    for r in results:
        assert r.level == 1  # only the M=3 level; level 0 (M=1) skipped
        assert 0.0 <= r.seen_acc <= 1.0 and 0.0 <= r.heldout_acc <= 1.0
        assert abs(r.gap - (r.seen_acc - r.heldout_acc)) < 1e-9
        assert r.n_heldout > 0
    # Retention (offset -1) carries the deterministic Bayes anchor; offset 0 has one too.
    assert any(r.offset == -1 and r.bayes_acc == 1.0 for r in results)
    assert any(r.offset == 0 and r.bayes_acc is not None for r in results)

    summ = summarize(results)
    assert (2, 1) in summ and "heldout_acc" in summ[(2, 1)]


def test_skips_single_tuple_levels():
    """A teacher with M=1 everywhere yields no results (nothing to hold out)."""
    teacher = _teacher(tuples=(1, 1))
    torch.manual_seed(0)
    data = teacher.sample_surface_prefix(8 * teacher.total, batch_size=8)
    residuals = {1: torch.randn(8, data.shape[1] - 1, 16)}
    assert analyze_residuals(teacher, residuals, data, offsets=[0]) == []


def test_full_wrapper_captures_and_runs():
    """End-to-end: capture residuals from a real student forward and analyze."""
    teacher = _teacher(sizes=(2, 3), dims=(6, 4, 8), tuples=(1, 3))
    student = TransformerDecoder(
        dim=teacher.dim, hidden_dim=16, num_heads=1, ff_hidden_dim=16,
        num_blocks=2, dropout=0.0, pe_type="none",
        encoder_layer=True, decoder_layer=True, layer_normalization=False,
    )
    torch.manual_seed(0)
    data = teacher.sample_surface_prefix(10 * teacher.total, batch_size=16)
    results = spelling_generalization(teacher, student, data, offsets=[-1, 0])
    assert results and all(r.level == 1 for r in results)
    assert all(0.0 <= r.heldout_acc <= 1.0 for r in results)


def test_render_spellings_share_latents():
    """Every rendered spelling decodes back to the same base ids."""
    teacher = _teacher(sizes=(2, 3), dims=(6, 4, 8), tuples=(2, 3))
    torch.manual_seed(0)
    base_ids = torch.randint(0, teacher.base_teacher.dim, (2, 5))  # (B, n_base)
    spellings = render_spellings(teacher, base_ids, k=4, seed=1)
    assert spellings.shape[0] == 4
    for i in range(4):
        decoded = teacher.decode_chunk_aligned(spellings[i]).argmax(-1)
        assert torch.equal(decoded, base_ids), f"spelling {i} changed the latents"


def test_cross_spelling_divergence_invariants():
    lp = torch.log_softmax(torch.randn(3, 4, 7, 5), dim=-1)  # (k=3, B=4, T=7, dim=5)
    # Identical spellings -> zero divergence.
    same = cross_spelling_divergence(lp[:1].repeat(3, 1, 1, 1))
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-6)
    # Different spellings -> non-negative, and >0 somewhere.
    div = cross_spelling_divergence(lp)
    assert (div >= -1e-6).all() and div.max() > 0
    assert div.shape == (4, 7)

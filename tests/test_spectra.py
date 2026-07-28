"""Unit tests for `src.spectra` and the spectrum-driven teacher constructors."""


import pytest
import torch

from src import spectra
from src.spectra import SpectrumSpec
from src.teachers import AttentionARTeacher, LinearARTeacher


# ---- laws --------------------------------------------------------------------


def test_flat_law_is_all_ones():
    s = spectra.spectrum(SpectrumSpec(law="flat"), 5)
    assert torch.allclose(s, torch.ones(5))


def test_geometric_law_has_constant_ratio():
    s = spectra.spectrum(SpectrumSpec(law="geometric", decay=0.5, normalize="none"), 4)
    assert torch.allclose(s, torch.tensor([1.0, 0.5, 0.25, 0.125]))


def test_power_law_slope_matches_alpha():
    alpha = 1.5
    s = spectra.spectrum(SpectrumSpec(law="power", alpha=alpha, normalize="none"), 16)
    k = torch.arange(16, dtype=torch.float32) + 1.0
    slope = torch.linalg.lstsq(
        torch.stack([k.log(), torch.ones(16)], dim=1), s.log().unsqueeze(1)
    ).solution[0, 0]
    assert slope == pytest.approx(-alpha, abs=1e-4)


def test_rank_truncates_tail():
    s = spectra.spectrum(SpectrumSpec(law="flat", rank=3), 6)
    assert torch.allclose(s, torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))


def test_reverse_puts_surviving_mass_at_the_end():
    s = spectra.spectrum(
        SpectrumSpec(law="geometric", decay=0.5, rank=2, normalize="none", reverse=True), 4
    )
    assert torch.allclose(s, torch.tensor([0.0, 0.0, 0.5, 1.0]))


# ---- normalization -----------------------------------------------------------


@pytest.mark.parametrize("law,kw", [("flat", {}), ("power", {"alpha": 1.0}), ("geometric", {"decay": 0.7})])
def test_spectral_normalization_gives_unit_max(law, kw):
    s = spectra.spectrum(SpectrumSpec(law=law, normalize="spectral", **kw), 8)
    assert s.max() == pytest.approx(1.0)


@pytest.mark.parametrize("law,kw", [("flat", {}), ("power", {"alpha": 1.0}), ("geometric", {"decay": 0.7})])
def test_frobenius_normalization_gives_unit_l2(law, kw):
    s = spectra.spectrum(SpectrumSpec(law=law, normalize="frobenius", **kw), 8)
    assert float(torch.linalg.norm(s)) == pytest.approx(1.0, abs=1e-6)


# ---- matrix generation -------------------------------------------------------


def test_random_matrix_has_prescribed_singular_values():
    s = spectra.spectrum(SpectrumSpec(law="power", alpha=1.0), 6)
    A = spectra.random_matrix_with_spectrum(s, 6)
    assert torch.allclose(torch.linalg.svdvals(A), s, atol=1e-5)


def test_random_matrix_supports_rectangular_shapes():
    s = spectra.spectrum(SpectrumSpec(law="power", alpha=0.5), 4)
    A = spectra.random_matrix_with_spectrum(s, 4, 9)
    assert A.shape == (4, 9)
    assert torch.allclose(torch.linalg.svdvals(A), s, atol=1e-5)


def test_shared_bases_keeps_singular_vectors_and_varies_spectrum():
    # Distinct alphas keep the singular values non-degenerate, so the singular
    # directions are well-defined (up to sign) and comparable.
    specs = spectra.expand_specs({"law": "power", "alpha": [0.5, 2.0]}, 2)
    A, B = spectra.random_matrices(2, 6, specs, shared_bases=True)
    ua = torch.linalg.svd(A)[0].abs()
    ub = torch.linalg.svd(B)[0].abs()
    assert torch.allclose(ua, ub, atol=1e-5)
    # Same features, different importance.
    assert not torch.allclose(torch.linalg.svdvals(A), torch.linalg.svdvals(B))


def test_shared_bases_respects_per_lag_rank():
    specs = spectra.expand_specs({"law": "flat", "rank": [4, 2]}, 2)
    A, B = spectra.random_matrices(2, 6, specs, shared_bases=True)
    assert torch.linalg.matrix_rank(A) == 4
    assert torch.linalg.matrix_rank(B) == 2


def test_disjoint_orthogonality_preserves_spectrum_and_is_frobenius_orthogonal():
    specs = spectra.expand_specs({"law": "power", "alpha": 1.0, "rank": 3}, 3)
    mats = spectra.random_matrices(3, 12, specs, orthogonality="disjoint")
    target = spectra.spectrum(specs[0], 12)[:3]
    for i, A in enumerate(mats):
        assert torch.allclose(torch.linalg.svdvals(A)[:3], target, atol=1e-5)
        for B in mats[i + 1 :]:
            assert float((A * B).sum().abs()) < 1e-4


def test_disjoint_orthogonality_rejects_rank_overflow():
    specs = spectra.expand_specs({"law": "flat", "rank": 4}, 3)
    with pytest.raises(ValueError, match="sum to"):
        spectra.random_matrices(3, 8, specs, orthogonality="disjoint")


def test_gram_schmidt_warns_when_it_would_break_the_spectrum():
    specs = spectra.expand_specs({"law": "power", "alpha": 1.0}, 2)
    with pytest.warns(UserWarning, match="does not preserve"):
        spectra.random_matrices(2, 6, specs, orthogonality="gram_schmidt")


def test_gram_schmidt_matrices_are_frobenius_orthogonal():
    mats = spectra.random_matrices(3, 6, SpectrumSpec(), orthogonality="gram_schmidt")
    for i, A in enumerate(mats):
        for B in mats[i + 1 :]:
            assert float((A * B).sum().abs()) < 1e-4


# ---- per-index broadcasting --------------------------------------------------


def test_expand_specs_broadcasts_per_lag_fields():
    specs = spectra.expand_specs({"law": "power", "alpha": [0.5, 1.0, 2.0]}, 3)
    assert [s.alpha for s in specs] == [0.5, 1.0, 2.0]
    assert all(s.law == "power" for s in specs)


def test_expand_specs_rejects_wrong_length():
    with pytest.raises(ValueError, match="expected 3"):
        spectra.expand_specs({"rank": [1, 2]}, 3)


# ---- LinearARTeacher wiring --------------------------------------------------


def test_outer_law_sets_the_operator_norm_of_each_lag():
    t = LinearARTeacher.from_parameters(
        dim=8,
        window=3,
        span_lengths=[1, 1, 1],
        spectrum={"law": "power", "alpha": 1.0, "normalize": "spectral"},
        lag_spectrum={"law": "geometric", "decay": 0.5, "normalize": "none"},
    )
    norms = [float(torch.linalg.matrix_norm(A.detach(), ord=2)) for A in t._get_weights()]
    assert norms == pytest.approx([1.0, 0.5, 0.25], abs=1e-5)


def test_inner_spectrum_shape_survives_the_outer_scaling():
    t = LinearARTeacher.from_parameters(
        dim=8,
        window=2,
        span_lengths=[1, 1],
        scale=3.0,
        spectrum={"law": "power", "alpha": 1.0},
        lag_spectrum={"law": "geometric", "decay": 2.0, "normalize": "none"},
    )
    for lag, A in enumerate(t._get_weights()):
        sv = torch.linalg.svdvals(A)
        assert torch.allclose(sv / sv.max(), t.singular_values[lag], atol=1e-5)


def test_frobenius_normalized_inner_law_makes_outer_law_the_frobenius_norm():
    t = LinearARTeacher.from_parameters(
        dim=8,
        window=2,
        span_lengths=[1, 1],
        spectrum={"law": "power", "alpha": 1.0, "normalize": "frobenius"},
        lag_spectrum={"law": "geometric", "decay": 0.5, "normalize": "none"},
    )
    norms = [float(torch.linalg.matrix_norm(A.detach())) for A in t._get_weights()]
    assert norms == pytest.approx([1.0, 0.5], abs=1e-5)


def test_per_lag_rank_produces_different_effective_dimensions():
    t = LinearARTeacher.from_parameters(
        dim=10, window=3, span_lengths=[1, 1, 1], spectrum={"rank": [6, 3, 1]}
    )
    ranks = [int(torch.linalg.matrix_rank(A)) for A in t._get_weights()]
    assert ranks == [6, 3, 1]
    assert t.rank == 6


def test_teacher_rejects_rank_above_dim():
    with pytest.raises(ValueError, match="must be <= dim"):
        LinearARTeacher.from_parameters(
            dim=4, window=1, span_lengths=[1], spectrum={"rank": 5}
        )


def test_lag_restriction_keeps_the_heaviest_end():
    # decay > 1 with reverse=False puts the mass on the most recent spans.
    t = LinearARTeacher.from_parameters(
        dim=6,
        window=3,
        span_lengths=[1, 2, 3],
        lag_spectrum={"law": "geometric", "decay": 2.0, "normalize": "none"},
    )
    r = t.with_lag_restriction(2)
    assert r.window == 2
    assert r.span_lengths == [2, 3]
    assert torch.allclose(r._get_weights(), t._get_weights()[-2:])

    # reverse=True flips which end carries the weight.
    t2 = LinearARTeacher.from_parameters(
        dim=6,
        window=3,
        span_lengths=[1, 2, 3],
        lag_spectrum={"law": "geometric", "decay": 2.0, "normalize": "none", "reverse": True},
    )
    r2 = t2.with_lag_restriction(2)
    assert r2.span_lengths == [1, 2]
    assert torch.allclose(r2._get_weights(), t2._get_weights()[:2])


def test_lag_restriction_carries_spectrum_metadata():
    t = LinearARTeacher.from_parameters(
        dim=6, window=3, span_lengths=[1, 1, 1], spectrum={"rank": [4, 3, 2]}
    )
    r = t.with_lag_restriction(2)
    assert [s.rank for s in r.spectrum_specs] == [3, 2]
    assert r.singular_values.shape == (2, 6)


def test_shared_bases_teacher_shares_features_across_lags():
    t = LinearARTeacher.from_parameters(
        dim=8,
        window=2,
        span_lengths=[1, 1],
        spectrum={"law": "power", "alpha": [0.5, 2.0]},
        shared_bases=True,
    )
    A, B = t._get_weights()
    ua = torch.linalg.svd(A)[0][:, :3].abs()
    ub = torch.linalg.svd(B)[0][:, :3].abs()
    assert torch.allclose(ua, ub, atol=1e-5)


def test_disjoint_teacher_lags_are_frobenius_orthogonal():
    t = LinearARTeacher.from_parameters(
        dim=12,
        window=3,
        span_lengths=[1, 1, 1],
        spectrum={"law": "power", "alpha": 1.0, "rank": 4},
        orthogonality="disjoint",
    )
    A, B, C = t._get_weights()
    for X, Y in ((A, B), (A, C), (B, C)):
        assert float((X * Y).sum().abs()) < 1e-4


def test_teacher_still_produces_normalized_log_probs_with_a_power_spectrum():
    t = LinearARTeacher.from_parameters(
        dim=6,
        window=2,
        span_lengths=[1, 1],
        scale=4.0,
        spectrum={"law": "power", "alpha": 1.5},
    )
    ctx = torch.zeros(3, t.context_length, t.dim)
    ctx[..., 0] = 1.0
    lp = t.next_token_log_probs(ctx)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(3), atol=1e-5)


# ---- AttentionARTeacher wiring -----------------------------------------------


def test_attention_teacher_default_is_untouched_by_the_spectrum_hook():
    a = AttentionARTeacher(dim=6, hidden_dim=8, seed=0)
    b = AttentionARTeacher(dim=6, hidden_dim=8, seed=0, spectrum=None)
    assert torch.allclose(a.encoder.weight, b.encoder.weight)
    assert a.spectrum_spec is None


def test_attention_teacher_applies_spectrum_to_targets():
    t = AttentionARTeacher(
        dim=6,
        hidden_dim=8,
        seed=0,
        spectrum={"law": "power", "alpha": 1.0},
        spectrum_targets=["encoder", "readout"],
    )
    expected = spectra.spectrum(SpectrumSpec(law="power", alpha=1.0), 6)
    assert torch.allclose(torch.linalg.svdvals(t.encoder.weight), expected, atol=1e-5)
    assert torch.allclose(torch.linalg.svdvals(t.readout.weight), expected, atol=1e-5)


def test_attention_teacher_spectrum_targets_are_validated():
    with pytest.raises(ValueError, match="unknown spectrum target"):
        AttentionARTeacher(dim=6, spectrum={"law": "flat"}, spectrum_targets=["bogus"])


def test_attention_teacher_with_spectrum_is_seed_reproducible():
    kw = dict(dim=6, hidden_dim=8, seed=3, spectrum={"law": "power", "alpha": 2.0})
    a, b = AttentionARTeacher(**kw), AttentionARTeacher(**kw)
    assert torch.allclose(a.encoder.weight, b.encoder.weight)
    assert torch.allclose(
        a.block.self_attention.value_proj.weight, b.block.self_attention.value_proj.weight
    )

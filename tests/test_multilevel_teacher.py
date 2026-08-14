"""Correctness suite for MultiLevelHierarchicalTeacher + ChunkCode.

Covers:
  1. L=1 equivalence with HierarchicalTeacher (backward-compat guarantee).
  2. Multi-level decode round-trip at every level.
  3. Normalization of unroll surface log-probs.
  4. AR (predict_next) vs unrolled consistency at arbitrary — incl. mid-chunk,
     mid higher-level-chunk — positions.
  5. Fold-vs-brute-force: predict_next equals a direct enumeration of the joint
     over (base token, per-level tuples). Certifies the nested marginalization.
  6. Adaptive (attention) base path.
"""
from math import prod

import pytest
import torch
import torch.nn.functional as F

from src.teachers import (
    AttentionARTeacher,
    ChunkCode,
    HierarchicalTeacher,
    LinearARTeacher,
    MultiLevelHierarchicalTeacher,
)


def _linear_base(dim, window, scale=10.0, seed=0):
    torch.manual_seed(seed)
    return LinearARTeacher.from_parameters(
        dim=dim,
        span_lengths=[1] * window,
        spectrum={'rank': dim},
        window=window,
        lag_spectrum={'law': 'geometric', 'decay': 1.7},
        scale=scale,
    )


def _build_surface(teacher, base_ids, seed=0):
    """Top-down expand `base_ids` (n_base,) into surface + per-level id streams.

    Returns (surface (n_base*total, dim), id_streams) where id_streams[l] is the
    id stream over the input alphabet of level l (id_streams[0] == base_ids),
    and id_streams[num_levels] is the surface id stream.
    """
    torch.manual_seed(seed)
    ids = base_ids
    id_streams = [ids]
    surface = None
    for l, level in enumerate(teacher.levels):
        tuple_ids = torch.randint(0, level.num_tuples, ids.shape)
        chunks = level.sample(ids, tuple_ids=tuple_ids)  # (count, size, out_dim)
        child_ids = chunks.argmax(dim=-1).reshape(-1)
        id_streams.append(child_ids)
        if l == teacher.num_levels - 1:
            surface = chunks.reshape(-1, level.out_dim)
        ids = child_ids
    return surface, id_streams


def test_l1_equivalence():
    """A one-level MultiLevel matches HierarchicalTeacher on the same table."""
    hidden_dim, chunk_dim, chunk_size, M, window = 8, 8, 2, 2, 3
    base = _linear_base(hidden_dim, window)
    ht = HierarchicalTeacher(
        base_teacher=base, chunk_dim=chunk_dim, chunk_size=chunk_size,
        num_tuples=M, chunk_seed=0,
    )
    cc = ChunkCode(
        in_dim=hidden_dim, out_dim=chunk_dim, size=chunk_size, num_tuples=M,
        chunk_table=ht._chunk_table.clone(),
    )
    ml = MultiLevelHierarchicalTeacher(base_teacher=base, levels=[cc])

    assert ml.dim == ht.dim
    assert ml.context_length == ht.context_length
    assert ml.burn_in == ht.burn_in
    assert ml.total == chunk_size
    assert list(ml.span_lengths) == list(ht.span_lengths)

    # Valid surface batch.
    torch.manual_seed(3)
    B, L_h = 3, 12
    base_ids = torch.randint(0, hidden_dim, (B * L_h,))
    surf, _ = _build_surface(ml, base_ids)
    surf = surf.reshape(B, L_h * chunk_size, chunk_dim)

    with torch.no_grad():
        lp_ht = ht.unroll(surf)
        lp_ml = ml.unroll(surf)
    assert torch.allclose(lp_ht, lp_ml, atol=1e-5), (
        f"unroll mismatch: {(lp_ht - lp_ml).abs().max()}"
    )

    # next_token_log_probs on a chunk-aligned context.
    ctx = surf[:, : ht.context_length, :]
    with torch.no_grad():
        assert torch.allclose(
            ht.next_token_log_probs(ctx), ml.next_token_log_probs(ctx), atol=1e-5
        )

    # predict_next across chunk-aligned and mid-chunk positions.
    for T in range(ml.burn_in, ml.burn_in + 2 * chunk_size):
        pref = surf[:, :T, :]
        with torch.no_grad():
            assert torch.allclose(
                ht.predict_next(pref), ml.predict_next(pref), atol=1e-5
            ), f"predict_next mismatch at T={T}"


def _multilevel(sizes, dims, tuples, base_window=2, seed=0):
    """Build an L-level teacher. `dims` = [base_dim, out0, out1, ...] (len L+1)."""
    base = _linear_base(dims[0], base_window, seed=seed)
    levels = [
        ChunkCode(
            in_dim=dims[l], out_dim=dims[l + 1], size=sizes[l],
            num_tuples=tuples[l], chunk_seed=100 + l,
        )
        for l in range(len(sizes))
    ]
    return MultiLevelHierarchicalTeacher(base_teacher=base, levels=levels)


def test_planning_beliefs_exact():
    """next_base_log_probs + the downward cascade match brute-force enumeration."""
    ml = _multilevel(sizes=[2, 3], dims=[5, 4, 7], tuples=[2, 2], base_window=2)
    torch.manual_seed(0)
    surf = ml.sample_surface_prefix((ml.base_teacher.burn_in + 5) * ml.total, batch_size=2)
    base = ml.base_teacher
    total, burn_in, w = ml.total, ml.burn_in, base.context_length

    # (1) p_next == current-base posterior pushed through one base transition.
    p_cur = ml.latent_beliefs(surf)[0]
    p_next = ml.next_base_log_probs(surf)
    base_hidden = ml._decode_levels(surf, stop_after=0)
    B, n_pred, bd = p_cur.shape
    for pos in range(0, n_pred, 3):
        b_idx = (burn_in + pos) // total
        prior = base_hidden[:, b_idx - w + 1 : b_idx, :]
        exp = torch.zeros(B, bd)
        for a in range(bd):
            cand = torch.eye(bd)[a].view(1, 1, bd).expand(B, 1, bd)
            nxt = base.next_token_log_probs(torch.cat([prior, cand], dim=1))
            exp += p_cur[:, pos, :].exp()[:, a : a + 1] * nxt.exp()
        assert (exp.clamp(min=1e-30).log() - p_next[:, pos, :]).abs().max() < 1e-4

    # (2) downward cascade surface marginal == enumeration over (a, m0, m1).
    pb = torch.randn(bd).log_softmax(0)
    s1 = ml.levels[1].size
    M0, M1 = ml.levels[0].num_tuples, ml.levels[1].num_tuples
    tbl0, tbl1 = ml.levels[0]._chunk_slot_indices, ml.levels[1]._chunk_slot_indices
    for local_pos in range(total):
        slot0, slot1 = local_pos // s1, local_pos % s1
        got = ml._downward_beliefs(pb.unsqueeze(0), local_pos)[-1].exp().squeeze(0)
        enum = torch.zeros(ml.dim)
        for a in range(bd):
            for m0 in range(M0):
                b = int(tbl0[a, m0, slot0])
                for m1 in range(M1):
                    enum[int(tbl1[b, m1, slot1])] += pb.exp()[a] / (M0 * M1)
        assert (got - enum / enum.sum()).abs().max() < 1e-5

    # (3) planning_beliefs: one entry per level (latent + surface), normalized,
    # with U_l = total // span[l] units of the next base token.
    plan = ml.planning_beliefs(surf)
    assert len(plan) == ml.num_levels + 1
    assert [x.shape[2] for x in plan] == [1, 2, 6]  # top / mid / surface
    for x in plan:
        sums = x.exp().sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_next_unit_beliefs_exact():
    """Offset-+1 beliefs match brute force for both sibling and crossing cases."""
    ml = _multilevel(sizes=[2, 3], dims=[5, 4, 7], tuples=[2, 2], base_window=2)
    torch.manual_seed(0)
    surf = ml.sample_surface_prefix((ml.base_teacher.burn_in + 4) * ml.total, batch_size=1)
    total, burn_in = ml.total, ml.burn_in
    s0, s1 = ml.levels[0].size, ml.levels[1].size
    M0, M1 = ml.levels[0].num_tuples, ml.levels[1].num_tuples

    nu = ml.next_unit_beliefs(surf)
    assert torch.allclose(nu[0], ml.next_base_log_probs(surf), atol=1e-5)  # base +1 = p_next

    base_lp = ml.base_teacher.unroll(ml._decode_levels(surf, stop_after=0))
    p_next = ml.next_base_log_probs(surf)
    tbl0, tbl1 = ml.levels[0]._chunk_slot_indices, ml.levels[1]._chunk_slot_indices
    for j in range(nu[1].shape[1]):
        i, lp = j // total, j % total
        slot0, r1 = lp // s1, lp % s1
        sb = surf[0, burn_in + i * total : burn_in + (i + 1) * total, :]
        p_base = base_lp[0, i, :].exp()
        obs_mid = [
            int(ml.levels[1].decode(sb[jj * s1 : (jj + 1) * s1].unsqueeze(0)).argmax(-1))
            for jj in range(slot0)
        ]
        obs_surf = [int(sb[slot0 * s1 + k].argmax()) for k in range(r1)]
        belief = torch.zeros(ml.levels[1].in_dim)
        if slot0 + 1 < s0:  # sibling within current base token
            for a in range(ml.base_teacher.dim):
                for m0 in range(M0):
                    t0 = tbl0[a, m0]
                    if any(int(t0[jj]) != obs_mid[jj] for jj in range(slot0)):
                        continue
                    cid = int(t0[slot0])
                    beta = sum(
                        1.0 / M1
                        for m1 in range(M1)
                        if all(int(tbl1[cid, m1, k]) == obs_surf[k] for k in range(r1))
                    )
                    belief[int(t0[slot0 + 1])] += p_base[a].item() / M0 * beta
        else:  # crossing to the next base token's first unit
            pnb = p_next[0, j, :].exp()
            for a2 in range(ml.base_teacher.dim):
                for m0 in range(M0):
                    belief[int(tbl0[a2, m0, 0])] += pnb[a2].item() / M0
        belief = belief / belief.sum().clamp(min=1e-30)
        assert (nu[1][0, j].exp() - belief).abs().max() < 1e-5


def test_decode_roundtrip():
    ml = _multilevel(sizes=[2, 3], dims=[6, 4, 8], tuples=[2, 2])
    torch.manual_seed(1)
    n_base = 5
    base_ids = torch.randint(0, ml.base_teacher.dim, (n_base,))
    surf, id_streams = _build_surface(ml, base_ids)
    surf = surf.unsqueeze(0)  # (1, L_surf, dim)

    # Decode to base tokens and to every intermediate level.
    for l in range(ml.num_levels):
        decoded = ml._decode_levels(surf, stop_after=l + 1).argmax(-1).squeeze(0)
        assert torch.equal(decoded, id_streams[l + 1]), f"level {l} decode mismatch"
    base_decoded = ml.decode_chunk_aligned(surf).argmax(-1).squeeze(0)
    assert torch.equal(base_decoded, id_streams[0])


def test_unroll_normalization():
    ml = _multilevel(sizes=[2, 3], dims=[6, 4, 8], tuples=[1, 2])
    torch.manual_seed(2)
    B = 4
    n_base = ml.base_teacher.burn_in + 6
    surf = ml.sample_surface_prefix(n_base * ml.total, batch_size=B)
    with torch.no_grad():
        lp = ml.unroll(surf)
    sums = lp.exp().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), (
        f"probs don't sum to 1: min={sums.min()} max={sums.max()}"
    )


def _check_ar_unroll(ml, seed=5):
    torch.manual_seed(seed)
    B = 3
    n_base = ml.base_teacher.burn_in + 5
    L_surf = n_base * ml.total
    surf = ml.sample_surface_prefix(L_surf, batch_size=B)
    with torch.no_grad():
        lp = ml.unroll(surf)  # (B, L_surf - burn_in, dim)
    # Compare predict_next at each output position to the unrolled value.
    for j in range(0, lp.shape[1], max(1, ml.total // 2)):
        T = ml.burn_in + j
        pref = surf[:, :T, :]
        with torch.no_grad():
            ar = ml.predict_next(pref)
        diff = (ar - lp[:, j, :]).abs().max().item()
        assert diff < 1e-4, f"AR/unroll mismatch at j={j} (T={T}): {diff}"


def test_ar_unroll_consistency():
    _check_ar_unroll(_multilevel(sizes=[2, 3], dims=[6, 4, 8], tuples=[2, 2]))
    _check_ar_unroll(_multilevel(sizes=[3, 2, 2], dims=[5, 6, 4, 8], tuples=[1, 2, 1]))


def _brute_force_l2(ml, prefix):
    """Independent P(surface_T | prefix) for a 2-level teacher via enumeration
    over (base token a, tuple m0, tuple m1)."""
    assert ml.num_levels == 2
    s0, s1 = ml.levels[0].size, ml.levels[1].size
    total = s0 * s1
    M0, M1 = ml.levels[0].num_tuples, ml.levels[1].num_tuples
    base_dim, surf_dim = ml.base_teacher.dim, ml.dim

    T = prefix.shape[1]
    r = T % total
    aligned = T - r
    slot0, r1 = r // s1, r % s1
    slot1 = r1  # span[2] == 1

    base_hidden = ml._decode_levels(prefix[:, :aligned, :], stop_after=0)
    p_base = ml._base_next_log_probs(base_hidden).exp().squeeze(0)  # (base_dim,)

    tbl0 = ml.levels[0]._chunk_slot_indices  # (base_dim, M0, s0)
    tbl1 = ml.levels[1]._chunk_slot_indices  # (out0, M1, s1)

    # Decoded values of the completed level-0 children (over out_dim0).
    obs0 = []
    for j in range(slot0):
        child = prefix[:, aligned + j * s1 : aligned + (j + 1) * s1, :]
        obs0.append(int(ml.levels[1].decode(child).argmax(-1)))
    # Observed surface tokens of the completed level-1 slots.
    obs1 = [
        int(prefix[0, aligned + slot0 * s1 + j, :].argmax())
        for j in range(slot1)
    ]

    probs = torch.zeros(surf_dim)
    for a in range(base_dim):
        for m0 in range(M0):
            tup0 = tbl0[a, m0]  # (s0,)
            if any(int(tup0[j]) != obs0[j] for j in range(slot0)):
                continue
            b = int(tup0[slot0])
            for m1 in range(M1):
                tup1 = tbl1[b, m1]  # (s1,)
                if any(int(tup1[j]) != obs1[j] for j in range(slot1)):
                    continue
                pred = int(tup1[slot1])
                probs[pred] += p_base[a].item() / (M0 * M1)
    probs /= probs.sum().clamp(min=1e-30)
    return probs


def test_fold_vs_brute_force():
    ml = _multilevel(sizes=[2, 3], dims=[5, 4, 7], tuples=[2, 2], base_window=2)
    torch.manual_seed(7)
    n_base = ml.base_teacher.burn_in + 4
    surf = ml.sample_surface_prefix(n_base * ml.total, batch_size=1)
    # Probe a spread of positions: chunk-aligned, mid-level-1, mid-level-0.
    for T in range(ml.burn_in, ml.burn_in + 2 * ml.total):
        pref = surf[:, :T, :]
        with torch.no_grad():
            fold = ml.predict_next(pref).exp().squeeze(0)
        brute = _brute_force_l2(ml, pref)
        diff = (fold - brute).abs().max().item()
        assert diff < 1e-5, f"fold != brute force at T={T}: {diff}"


def _brute_beliefs_l2(ml, prefix):
    """Independent per-level posteriors P(base token | prefix), P(mid token |
    prefix) for a 2-level teacher, by enumerating (a, m0, m1)."""
    s0, s1 = ml.levels[0].size, ml.levels[1].size
    total = s0 * s1
    M0, M1 = ml.levels[0].num_tuples, ml.levels[1].num_tuples
    base_dim, out0 = ml.base_teacher.dim, ml.levels[0].out_dim

    T = prefix.shape[1]
    r = T % total
    aligned = T - r
    slot0, r1 = r // s1, r % s1
    slot1 = r1

    base_hidden = ml._decode_levels(prefix[:, :aligned, :], stop_after=0)
    p_base = ml._base_next_log_probs(base_hidden).exp().squeeze(0)
    tbl0 = ml.levels[0]._chunk_slot_indices
    tbl1 = ml.levels[1]._chunk_slot_indices

    obs0 = [
        int(ml.levels[1].decode(prefix[:, aligned + j * s1 : aligned + (j + 1) * s1, :]).argmax(-1))
        for j in range(slot0)
    ]
    obs1 = [int(prefix[0, aligned + slot0 * s1 + j, :].argmax()) for j in range(slot1)]

    post_base = torch.zeros(base_dim)
    post_mid = torch.zeros(out0)
    for a in range(base_dim):
        for m0 in range(M0):
            tup0 = tbl0[a, m0]
            if any(int(tup0[j]) != obs0[j] for j in range(slot0)):
                continue
            b = int(tup0[slot0])
            for m1 in range(M1):
                tup1 = tbl1[b, m1]
                if any(int(tup1[j]) != obs1[j] for j in range(slot1)):
                    continue
                w = p_base[a].item() / (M0 * M1)
                post_base[a] += w
                post_mid[b] += w
    return post_base / post_base.sum().clamp(min=1e-30), post_mid / post_mid.sum().clamp(min=1e-30)


def test_latent_beliefs():
    ml = _multilevel(sizes=[2, 3], dims=[5, 4, 7], tuples=[2, 2], base_window=2)
    torch.manual_seed(9)
    n_base = ml.base_teacher.burn_in + 4
    surf = ml.sample_surface_prefix(n_base * ml.total, batch_size=1)

    beliefs = ml.latent_beliefs(surf)
    assert len(beliefs) == ml.num_levels
    n_out = surf.shape[1] - ml.burn_in
    for l in range(ml.num_levels):
        assert beliefs[l].shape == (1, n_out, ml.levels[l].in_dim)
        sums = beliefs[l].exp().sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    # Base-level belief at each base-token start is exactly the base next-token dist.
    base_lp = ml.base_teacher.unroll(ml._decode_levels(surf, stop_after=0))
    starts = torch.arange(0, n_out, ml.total)
    assert torch.allclose(beliefs[0][:, starts, :], base_lp, atol=1e-5)

    # Rigorous: per-level beliefs match brute-force enumeration at every position.
    for T in range(ml.burn_in, ml.burn_in + 2 * ml.total):
        j = T - ml.burn_in
        pb, pm = _brute_beliefs_l2(ml, surf[:, :T, :])
        assert (beliefs[0][0, j].exp() - pb).abs().max() < 1e-5, f"base belief @T={T}"
        assert (beliefs[1][0, j].exp() - pm).abs().max() < 1e-5, f"mid belief @T={T}"


def _bottom_chunks_valid(ml, surface) -> bool:
    """Every complete bottom-level chunk in `surface` matches some tuple."""
    level = ml.levels[-1]
    n = surface.shape[-2] // level.size
    slot_ids = surface[..., : n * level.size, :].reshape(-1, level.size, level.out_dim).argmax(-1)
    valid_set = {tuple(r) for r in level._chunk_slot_indices.reshape(-1, level.size).tolist()}
    return all(tuple(r) in valid_set for r in slot_ids.tolist())


def test_predictor_routing_and_burn_in():
    from src.data import ARDataset
    from src.predictors import MultiLevelHierarchicalPredictor, build_predictor

    ml = _multilevel(sizes=[2, 3], dims=[6, 4, 8], tuples=[2, 2])

    pred = build_predictor(ml, kind="classification")
    assert isinstance(pred, MultiLevelHierarchicalPredictor)
    with pytest.raises(TypeError, match="MultiLevelHierarchicalTeacher"):
        MultiLevelHierarchicalPredictor(_linear_base(4, 2))

    # Burn-in prefixes are valid chunk-composed surface (single + batched).
    burn = pred.random_burn_in(ml.burn_in)
    assert burn.shape == (ml.burn_in, ml.dim)
    assert _bottom_chunks_valid(ml, burn)
    burn_b = pred.random_burn_in_batch(5, ml.burn_in)
    assert burn_b.shape == (5, ml.burn_in, ml.dim)
    assert _bottom_chunks_valid(ml, burn_b)

    # End-to-end AR generation yields valid chunks (length a multiple of total).
    ds = ARDataset(
        predictor=pred, window=ml.window, dim=ml.dim, number=4,
        length=4 * ml.total, prefix_length=ml.burn_in, unroll_sequences=False,
    )
    data = ds.data  # (4, burn_in + 4*total, dim)
    assert data.shape == (4, ml.burn_in + 4 * ml.total, ml.dim)
    assert _bottom_chunks_valid(ml, data)


def test_adaptive_base():
    """Attention (unbounded) base: build, unroll, normalize, AR/unroll consistency."""
    torch.manual_seed(0)
    base = AttentionARTeacher(dim=6, unbounded=True, burn_in=2, scale=5.0, seed=0)
    levels = [
        ChunkCode(in_dim=6, out_dim=4, size=2, num_tuples=1, chunk_seed=1),
        ChunkCode(in_dim=4, out_dim=8, size=2, num_tuples=2, chunk_seed=2),
    ]
    ml = MultiLevelHierarchicalTeacher(base_teacher=base, levels=levels)
    assert ml.is_adaptive
    assert ml.burn_in == base.burn_in * ml.total

    B, n_base = 2, base.burn_in + 5
    surf = ml.sample_surface_prefix(n_base * ml.total, batch_size=B)
    with torch.no_grad():
        lp = ml.unroll(surf)
    sums = lp.exp().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    _check_ar_unroll(ml, seed=11)

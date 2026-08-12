"""Post-training spelling-generalization eval.

The abstraction test: each latent unit at level `l` has `M = num_tuples`
interchangeable surface *spellings* (tuples), all decoding to the same id via
globally disjoint supports. A student that formed the *abstract* unit decodes its
id from the residual regardless of which spelling realized it; one that took a
surface shortcut only works on spellings it has seen.

For each (layer, level with M>1, slot, offset) we fit a linear probe on positions
whose TARGET unit used a *train* subset of tuples and evaluate on positions whose
target unit used a *held-out* tuple (leave-one-tuple-out). We report:

    seen_acc    : probe eval on held-out positions of the *train* tuples
    heldout_acc : probe eval on *held-out-tuple* positions (novel spellings)
    gap         : seen_acc - heldout_acc  (the shortcut signature)
    bayes_acc   : the spelling-invariant Bayes-optimal accuracy (anchor)

heldout_acc near bayes_acc ⇒ genuine, spelling-invariant abstraction; heldout_acc
collapsing toward chance with a large gap ⇒ surface memorization.

This is a heavy, one-off checkpoint-time analysis (many probe fits, per-tuple
splits), deliberately kept out of the training loop. It reuses the teacher's
`decode_level` (unit ids + realized tuple), `_span`, and `latent_beliefs`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.teachers import MultiLevelHierarchicalTeacher


@dataclass
class SpellingResult:
    layer: int
    level: int
    slot: int
    offset: int
    seen_acc: float
    heldout_acc: float
    gap: float
    bayes_acc: Optional[float]
    n_heldout: int


def _capture_residuals(
    student: nn.Module, data: torch.Tensor
) -> Dict[int, torch.Tensor]:
    """Run one student forward and grab the residual after the pos-encoder
    (layer 0) and after each transformer block (layers 1..N)."""
    caps: Dict[int, torch.Tensor] = {}
    handles = []

    def mk(i: int):
        def hook(module, inputs, output):
            caps[i] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    handles.append(student.pos_encoder.register_forward_hook(mk(0)))
    for i, block in enumerate(student.transformer_blocks):
        handles.append(block.register_forward_hook(mk(i + 1)))

    was_training = student.training
    student.eval()
    with torch.no_grad():
        student(data[:, :-1, :])
    if was_training:
        student.train()
    for h in handles:
        h.remove()
    return caps


def _fit_probe(
    X: torch.Tensor, y: torch.Tensor, num_classes: int, max_iters: int, l2: float
) -> nn.Linear:
    probe = nn.Linear(X.shape[-1], num_classes).to(X.device)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    opt = torch.optim.LBFGS(
        probe.parameters(),
        max_iter=max_iters,
        tolerance_grad=1e-4,
        tolerance_change=1e-6,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(probe(X), y) + l2 * probe.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return probe


def _acc(probe: nn.Linear, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return (probe(X).argmax(dim=-1) == y).float().mean().item()


def analyze_residuals(
    teacher: MultiLevelHierarchicalTeacher,
    residuals: Dict[int, torch.Tensor],
    data: torch.Tensor,
    offsets: Optional[List[int]] = None,
    max_iters: int = 50,
    l2: float = 1e-4,
    fit_frac: float = 0.7,
    seed: int = 0,
    min_rows: int = 8,
) -> List[SpellingResult]:
    """Core analysis over pre-captured residuals.

    `residuals[layer]` is (N, T, D). `data` is the full (N, L, dim) surface
    batch the residuals came from (T == L - 1). Only levels with num_tuples > 1
    are analyzed; only offsets <= 0 (retention/refinement), where a spelling is
    actually observed, are meaningful (planning is skipped by default).
    """
    if offsets is None:
        base_ctx = teacher.base_teacher.burn_in
        offsets = list(range(-base_ctx, 1))  # retention + refinement, no planning

    span = teacher._span
    burn_in = teacher.burn_in
    beliefs = teacher.latent_beliefs(data) if 0 in offsets else None
    gen = torch.Generator().manual_seed(seed)

    results: List[SpellingResult] = []
    for level in range(teacher.num_levels):
        M = teacher.levels[level].num_tuples
        if M < 2:
            continue  # no spellings to hold out
        in_dim = teacher.levels[level].in_dim
        span_l, span_child = span[level], span[level + 1]
        unit_oh, tup = teacher.decode_level(data, level, return_tuple=True)
        labels = unit_oh.argmax(dim=-1)  # (N, L_level)
        L_level = labels.shape[1]

        for layer, R in residuals.items():
            N, T, D = R.shape
            t = torch.arange(T, device=R.device)
            for slot in range(teacher.levels[level].size):
                sel = t[(t % span_l) // span_child == slot]
                for offset in offsets:
                    target_c = sel // span_l + offset
                    valid = (target_c >= 0) & (target_c < L_level)
                    if not valid.any():
                        continue
                    tsel = sel[valid]
                    tc = target_c[valid]

                    X = R[:, tsel, :].reshape(-1, D)
                    y = labels[:, tc].reshape(-1)
                    mtup = tup[:, tc].reshape(-1)  # target unit's realized tuple

                    res = _tuple_split_eval(
                        X, y, mtup, in_dim, M, max_iters, l2, fit_frac, gen, min_rows
                    )
                    if res is None:
                        continue
                    seen_acc, heldout_acc, n_heldout = res
                    bayes_acc = _bayes_acc(
                        offset, beliefs, level, tsel, y, burn_in, N
                    )
                    results.append(
                        SpellingResult(
                            layer=layer,
                            level=level,
                            slot=slot,
                            offset=offset,
                            seen_acc=seen_acc,
                            heldout_acc=heldout_acc,
                            gap=seen_acc - heldout_acc,
                            bayes_acc=bayes_acc,
                            n_heldout=n_heldout,
                        )
                    )
    return results


def _tuple_split_eval(X, y, mtup, in_dim, M, max_iters, l2, fit_frac, gen, min_rows):
    """Leave-one-tuple-out: fit on train tuples, eval on the held-out tuple.
    Returns (mean_seen_acc, mean_heldout_acc, total_heldout_rows) or None."""
    seen_accs: List[float] = []
    heldout_accs: List[float] = []
    n_heldout = 0
    for m in range(M):
        ho = mtup == m
        tr = ~ho
        if int(ho.sum()) < min_rows or int(tr.sum()) < min_rows:
            continue
        tr_idx = tr.nonzero(as_tuple=True)[0]
        perm = torch.randperm(tr_idx.numel(), generator=gen)
        n_fit = max(1, int(tr_idx.numel() * fit_frac))
        fit_idx = tr_idx[perm[:n_fit]]
        seen_idx = tr_idx[perm[n_fit:]]
        if seen_idx.numel() == 0:
            seen_idx = fit_idx
        probe = _fit_probe(X[fit_idx], y[fit_idx], in_dim, max_iters, l2)
        seen_accs.append(_acc(probe, X[seen_idx], y[seen_idx]))
        heldout_accs.append(_acc(probe, X[ho], y[ho]))
        n_heldout += int(ho.sum())
    if not heldout_accs:
        return None
    return (
        sum(seen_accs) / len(seen_accs),
        sum(heldout_accs) / len(heldout_accs),
        n_heldout,
    )


def _bayes_acc(offset, beliefs, level, tsel, y, burn_in, N) -> Optional[float]:
    """Spelling-invariant Bayes-optimal accuracy for these positions."""
    if offset < 0:
        return 1.0  # retention: target unit complete -> deterministic
    if offset != 0 or beliefs is None:
        return None
    idx = tsel - burn_in
    vb = idx >= 0
    if not bool(vb.any()):
        return None
    b = beliefs[level][:, idx.clamp(min=0), :]  # (N, P, C)
    belief = b.reshape(-1, b.shape[-1])
    valid = vb.unsqueeze(0).expand(N, -1).reshape(-1)
    with torch.no_grad():
        pred = belief[valid].argmax(dim=-1)
        return (pred == y[valid]).float().mean().item()


def spelling_generalization(
    teacher: MultiLevelHierarchicalTeacher,
    student: nn.Module,
    data: torch.Tensor,
    **kwargs,
) -> List[SpellingResult]:
    """Capture residuals from one student forward, then run `analyze_residuals`."""
    residuals = _capture_residuals(student, data)
    return analyze_residuals(teacher, residuals, data, **kwargs)


def summarize(results: List[SpellingResult]) -> Dict[tuple, Dict[str, float]]:
    """Aggregate per (layer, level): mean heldout_acc / seen_acc / gap / bayes_acc."""
    by_key: Dict[tuple, List[SpellingResult]] = {}
    for r in results:
        by_key.setdefault((r.layer, r.level), []).append(r)
    out: Dict[tuple, Dict[str, float]] = {}
    for key, rs in by_key.items():
        bayes = [r.bayes_acc for r in rs if r.bayes_acc is not None]
        out[key] = {
            "seen_acc": sum(r.seen_acc for r in rs) / len(rs),
            "heldout_acc": sum(r.heldout_acc for r in rs) / len(rs),
            "gap": sum(r.gap for r in rs) / len(rs),
            "bayes_acc": (sum(bayes) / len(bayes)) if bayes else float("nan"),
            "n": sum(r.n_heldout for r in rs),
        }
    return out

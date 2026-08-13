from math import prod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.teachers.base import ADAPTIVE, ARTeacher
from src.teachers.chunk_code import ChunkCode


class MultiLevelHierarchicalTeacher(ARTeacher):
    """A configurable multi-level chunk-composed teacher (hierarchical HMM).

    Wraps a base AR teacher over a top alphabet (`base_teacher.dim`) and a stack
    of `ChunkCode` levels ordered **top->bottom**: `levels[0]` is adjacent to the
    base, `levels[-1]` emits surface tokens. Each level maps its input alphabet
    to fixed length-`size` tuples over its output alphabet, with the alphabet
    chain `in_dim[0] == base.dim`, `in_dim[l] == out_dim[l-1]`, and surface vocab
    `out_dim[-1]`. One base token expands into `total == prod(size_l)` surface
    tokens.

    Context lives only in the base; the levels are fixed random codebooks with
    globally disjoint supports (so surface->...->base decoding is deterministic).

    Correctness at *any* surface position — including positions mid-way through
    several nested open chunks — comes from an exact **fold** of each level's
    `next_slot_logprobs`. At surface position T the open chunk at level `l` has
    `slot_l = (T mod span[l]) // span[l+1]` completed slots (whose decoded values
    are known), where `span[l] == prod_{j>=l} size[j]`. Folding a distribution
    over the base token down through the levels — conditioning each on its
    observed completed slots — yields the exact Bayes surface posterior, because
    `next_slot_logprobs` accepts an arbitrary input distribution and completed
    subtrees decode deterministically. `next_token_log_probs`, `predict_next`,
    and `unroll` all return **log surface probabilities**.

    With `L == 1` this reproduces `HierarchicalTeacher` exactly.
    """

    def __init__(
        self,
        base_teacher: ARTeacher,
        levels: List[ChunkCode],
    ) -> None:
        super().__init__()
        if len(levels) == 0:
            raise ValueError("levels must contain at least one ChunkCode.")

        # Validate the alphabet chain top->bottom.
        if levels[0].in_dim != base_teacher.dim:
            raise ValueError(
                f"levels[0].in_dim ({levels[0].in_dim}) must equal base_teacher.dim "
                f"({base_teacher.dim})."
            )
        for l in range(1, len(levels)):
            if levels[l].in_dim != levels[l - 1].out_dim:
                raise ValueError(
                    f"levels[{l}].in_dim ({levels[l].in_dim}) must equal "
                    f"levels[{l - 1}].out_dim ({levels[l - 1].out_dim})."
                )

        self.base_teacher = base_teacher
        self.levels = nn.ModuleList(levels)


        sizes = [lv.size for lv in levels]
        self.total = prod(sizes)
        # span[l] = surface tokens spanned by one input token at level l.
        # span[0] == total (one base token); span[L] == 1.
        span = [1] * (len(levels) + 1)
        for l in range(len(levels) - 1, -1, -1):
            span[l] = span[l + 1] * sizes[l]
        self._span = span

    @classmethod
    def from_level_specs(cls, base_teacher: ARTeacher, levels) -> "MultiLevelHierarchicalTeacher":
        """Build from lightweight per-level specs (the config entry point).

        `levels` is an ordered (top->bottom) list of mappings with keys
        `chunk_dim` (the level's output alphabet), `chunk_size`, and optional
        `num_tuples` (default 1) and `chunk_seed`. The input-alphabet chain is
        resolved here: `in_dim[0] == base_teacher.dim`, `in_dim[l] == chunk_dim[l-1]`.
        """
        codes: List[ChunkCode] = []
        in_dim = base_teacher.dim
        for spec in levels:
            out_dim = int(spec["chunk_dim"])
            codes.append(
                ChunkCode(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    size=int(spec["chunk_size"]),
                    num_tuples=int(spec.get("num_tuples", 1)),
                    chunk_seed=spec.get("chunk_seed", None),
                )
            )
            in_dim = out_dim
        return cls(base_teacher=base_teacher, levels=codes)

    # --- ARTeacher interface ---
    @property
    def dim(self) -> int:
        return self.levels[-1].out_dim

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def context_length(self) -> int:
        if self.base_teacher.is_adaptive:
            return ADAPTIVE
        return self.base_teacher.context_length * self.total

    @property
    def burn_in(self) -> int:
        # A whole number of base tokens, so sampled burn-in prefixes stay
        # chunk-aligned at every level.
        return self.base_teacher.burn_in * self.total

    @property
    def hidden_dim(self) -> int:
        """Top alphabet size (base vocab) — the outermost latent."""
        return self.base_teacher.dim

    @property
    def window(self) -> int:
        return self.base_teacher.window

    @property
    def span_lengths(self) -> List[int]:
        return [s * self.total for s in self.base_teacher.span_lengths]

    @property
    def stride(self) -> Optional[int]:
        base_stride = getattr(self.base_teacher, "stride", None)
        return base_stride * self.total if base_stride is not None else None

    def _get_weights(self) -> torch.Tensor:
        return self.base_teacher._get_weights()

    # --- decoding ---
    def _decode_levels(self, surface: torch.Tensor, stop_after: int) -> torch.Tensor:
        """Decode surface up the stack, applying levels L-1 .. stop_after.

        `stop_after == 0` -> base tokens (one-hot over `base.dim`).
        `stop_after == l+1` -> the completed-slot values at level `l` (one-hot
        over `out_dim[l]`); `stop_after == num_levels` returns surface unchanged.
        """
        x = surface
        for l in range(self.num_levels - 1, stop_after - 1, -1):
            x = self.levels[l].decode(x)
        return x

    def decode_chunk_aligned(self, surface: torch.Tensor) -> torch.Tensor:
        """Public surface -> base-token one-hots (for probes / analysis)."""
        return self._decode_levels(surface, stop_after=0)

    def decode_level(self, surface: torch.Tensor, level: int, return_tuple: bool = False):
        """Decode the level-`level` unit ids covering `surface`.

        Returns (..., L_level, in_dim[level]) one-hot. With `return_tuple`, also
        returns the realized tuple index per unit (..., L_level) — which of the
        M spellings produced each unit (used by the spelling-generalization
        analysis). Equivalent to `_decode_levels(surface, stop_after=level)` for
        the ids, but also exposes the tuple.
        """
        below = self._decode_levels(surface, stop_after=level + 1)
        return self.levels[level].decode(below, return_tuple=return_tuple)

    # --- the fold ---
    def _fold(
        self,
        p_base: torch.Tensor,
        base_surface: torch.Tensor,
        local_pos: int,
    ) -> torch.Tensor:
        """Exact Bayes surface log-probs at `local_pos` within a base token.

        Args:
            p_base: (..., base.dim) log-probs over the current base token.
            base_surface: (..., total, surface_dim) surface of the current base
                token. Only positions `< local_pos` are read (completed slots);
                the rest may be arbitrary (e.g. zero padding).
            local_pos: surface offset within the base token being predicted.

        Returns (..., surface_dim) log surface probabilities.
        """
        lead = base_surface.shape[:-2]
        p = p_base
        c = 0  # surface offset (within base token) of the current open chunk at level l
        for l in range(self.num_levels):
            span_child = self._span[l + 1]
            slot_l = (local_pos - c) // span_child
            size_l = self.levels[l].size
            observed = torch.zeros(
                *lead, size_l, dtype=torch.long, device=base_surface.device
            )
            if slot_l > 0:
                completed = base_surface[..., c : c + slot_l * span_child, :]
                vals = self._decode_levels(completed, stop_after=l + 1)  # (..., slot_l, out_dim[l])
                observed[..., :slot_l] = vals.argmax(dim=-1)
            p = self.levels[l].next_slot_logprobs(p, observed, slot_l, slot_l)
            c += slot_l * span_child
        return p

    def _fold_beliefs(
        self,
        p_base: torch.Tensor,
        base_surface: torch.Tensor,
        local_pos: int,
    ) -> List[torch.Tensor]:
        """Bayes-optimal belief over the *current* latent at every level.

        Exact posterior over each level-`l` unit covering `local_pos` given the
        base context and *all* observed surface of the current base token — via
        two-pass belief propagation on the open path:

          α_l : downward message (the fold's `p` entering level `l`) — evidence
                from the base prior and the completed slots of levels above.
          β_l : upward message — likelihood of the observed sub-tree under the
                level-`l` unit, i.e. this level's completed slots AND the
                partially-observed frontier below (which the student has seen).

        posterior_l ∝ α_l · β_l. Returns a length-L list, entry `l` shape
        (..., in_dim[l]) log-probs.
        """
        lead = base_surface.shape[:-2]

        # Downward pass: α_l entering each level, plus per-level observed slots.
        alphas: List[torch.Tensor] = []
        observeds: List[torch.Tensor] = []
        slots: List[int] = []
        p = p_base
        c = 0
        for l in range(self.num_levels):
            span_child = self._span[l + 1]
            slot_l = (local_pos - c) // span_child
            observed = torch.zeros(
                *lead, self.levels[l].size, dtype=torch.long, device=base_surface.device
            )
            if slot_l > 0:
                completed = base_surface[..., c : c + slot_l * span_child, :]
                vals = self._decode_levels(completed, stop_after=l + 1)
                observed[..., :slot_l] = vals.argmax(dim=-1)
            alphas.append(p)
            observeds.append(observed)
            slots.append(slot_l)
            p = self.levels[l].next_slot_logprobs(p, observed, slot_l, slot_l)
            c += slot_l * span_child

        # Upward pass: β_l = Σ_m (1/M) · compat(x, m) · β_{l+1}(child(x, m)),
        # bottoming out at the frontier (deepest level has no child factor).
        betas: List[Optional[torch.Tensor]] = [None] * self.num_levels
        beta_child_msg: Optional[torch.Tensor] = None
        for l in range(self.num_levels - 1, -1, -1):
            level = self.levels[l]
            compat = level._compat_mask(observeds[l], slots[l])  # (..., in_dim, M)
            if l == self.num_levels - 1:
                beta = compat.sum(dim=-1) / level.num_tuples
            else:
                child_idx = level._chunk_slot_indices[:, :, slots[l]]  # (in_dim, M)
                beta_child = beta_child_msg[..., child_idx]  # (..., in_dim, M)
                beta = (compat * beta_child).sum(dim=-1) / level.num_tuples
            betas[l] = beta
            beta_child_msg = beta

        beliefs: List[torch.Tensor] = []
        for l in range(self.num_levels):
            post = alphas[l].exp() * betas[l]
            post = post / post.sum(dim=-1, keepdim=True).clamp(min=1e-30)
            beliefs.append(post.clamp(min=1e-30).log())
        return beliefs

    def latent_beliefs(self, sequence: torch.Tensor) -> List[torch.Tensor]:
        """Per-level Bayes belief over the current unit at each predicted position.

        Mirrors `unroll` but returns, for each level `l`, the log-belief over the
        level-`l` latent (input alphabet `in_dim[l]`) covering every surface
        position in `[burn_in:]`. Returns a length-L list, entry `l` shape
        (B, L_surf - burn_in, in_dim[l]). Used as the offset-0 Bayes ceiling for
        the probes.
        """
        B, L_surf, D = sequence.shape
        if L_surf % self.total != 0:
            raise ValueError(
                f"sequence length {L_surf} must be a multiple of total={self.total}"
            )
        if L_surf <= self.burn_in:
            raise ValueError(
                f"sequence length {L_surf} must exceed burn_in {self.burn_in}"
            )

        base_hidden = self._decode_levels(sequence, stop_after=0)
        base_log_probs = self.base_teacher.unroll(base_hidden)  # (B, n_base_out, base_dim)
        n_base_out = base_log_probs.shape[1]
        pred_surface = sequence[:, self.burn_in :, :].reshape(B, n_base_out, self.total, D)

        p_base_flat = base_log_probs.reshape(B * n_base_out, -1)
        pred_surface_flat = pred_surface.reshape(B * n_base_out, self.total, D)

        per_level: List[List[torch.Tensor]] = [[] for _ in range(self.num_levels)]
        for local_pos in range(self.total):
            beliefs = self._fold_beliefs(p_base_flat, pred_surface_flat, local_pos)
            for l in range(self.num_levels):
                per_level[l].append(beliefs[l])

        out: List[torch.Tensor] = []
        for l in range(self.num_levels):
            stacked = torch.stack(per_level[l], dim=1)  # (B*n_base_out, total, in_dim[l])
            out.append(
                stacked.reshape(B, n_base_out * self.total, self.levels[l].in_dim)
            )
        return out

    def _base_next_log_probs(self, base_hidden: torch.Tensor) -> torch.Tensor:
        """Next base-token log-probs from a decoded base context, honoring the
        base teacher's memory regime (whole prefix if adaptive, else the trailing
        `context_length` tokens)."""
        if not self.base_teacher.is_adaptive:
            ctx = self.base_teacher.context_length
            if base_hidden.shape[-2] > ctx:
                base_hidden = base_hidden[..., -ctx:, :]
        return self.base_teacher.next_token_log_probs(base_hidden)

    # --- ARTeacher methods ---
    def next_token_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        """Predict surface slot 0 of the next base token from a chunk-aligned
        context. context: (..., n*total, surface_dim) -> (..., surface_dim)."""
        T = context.shape[-2]
        if self.is_adaptive:
            if T % self.total != 0:
                raise ValueError(
                    f"adaptive context length {T} is not aligned to total={self.total}"
                )
        elif T != self.context_length:
            raise ValueError(
                f"context has {T} tokens; expected {self.context_length}"
            )
        base_hidden = self._decode_levels(context, stop_after=0)
        p_base = self._base_next_log_probs(base_hidden)
        lead = context.shape[:-2]
        base_surface = context.new_zeros(*lead, self.total, self.dim)
        return self._fold(p_base, base_surface, local_pos=0)

    def predict_next(self, prefix: torch.Tensor) -> torch.Tensor:
        """Autoregressive single-step prediction at any (possibly mid-chunk)
        surface position."""
        T = prefix.shape[-2]
        if T < self.burn_in:
            raise ValueError(f"prefix length {T} < burn_in {self.burn_in}")

        r0 = T % self.total
        aligned_len = T - r0  # whole base tokens
        base_hidden = self._decode_levels(prefix[..., :aligned_len, :], stop_after=0)
        p_base = self._base_next_log_probs(base_hidden)

        lead = prefix.shape[:-2]
        base_surface = prefix.new_zeros(*lead, self.total, self.dim)
        if r0 > 0:
            base_surface[..., :r0, :] = prefix[..., aligned_len : aligned_len + r0, :]
        return self._fold(p_base, base_surface, local_pos=r0)

    def unroll(
        self,
        sequence: torch.Tensor,
        return_targets: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Per-position surface predictions over a chunk-aligned batch.

        sequence: (B, L_surf, surface_dim), L_surf a multiple of `total` and
        > burn_in. Returns log-probs (B, L_surf - burn_in, surface_dim).
        """
        B, L_surf, D = sequence.shape
        if L_surf % self.total != 0:
            raise ValueError(
                f"sequence length {L_surf} must be a multiple of total={self.total}"
            )
        if L_surf <= self.burn_in:
            raise ValueError(
                f"sequence length {L_surf} must exceed burn_in {self.burn_in}"
            )

        base_hidden = self._decode_levels(sequence, stop_after=0)  # (B, L_h, base_dim)
        base_log_probs = self.base_teacher.unroll(base_hidden)  # (B, n_base_out, base_dim)
        n_base_out = base_log_probs.shape[1]

        # Surface of each *predicted* base token (those starting at burn_in).
        pred_surface = sequence[:, self.burn_in :, :].reshape(B, n_base_out, self.total, D)

        p_base_flat = base_log_probs.reshape(B * n_base_out, -1)
        pred_surface_flat = pred_surface.reshape(B * n_base_out, self.total, D)

        slot_logprobs = [
            self._fold(p_base_flat, pred_surface_flat, local_pos)
            for local_pos in range(self.total)
        ]
        stacked = torch.stack(slot_logprobs, dim=1)  # (B*n_base_out, total, D)
        surface_logprobs = stacked.reshape(B, n_base_out * self.total, D)

        if return_targets:
            return surface_logprobs, sequence[:, self.burn_in :, :]
        return surface_logprobs

    def with_lag_restriction(self, k: int) -> "MultiLevelHierarchicalTeacher":
        """Restrict the base teacher's lags; keep the same level codebooks."""
        restricted_base = self.base_teacher.with_lag_restriction(k)
        levels = [
            ChunkCode(
                in_dim=lv.in_dim,
                out_dim=lv.out_dim,
                size=lv.size,
                num_tuples=lv.num_tuples,
                chunk_table=lv._chunk_table,
            )
            for lv in self.levels
        ]
        return MultiLevelHierarchicalTeacher(base_teacher=restricted_base, levels=levels)

    # --- generation ---
    def sample_surface_prefix(
        self,
        num_surface_tokens: int,
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Top-down tree sample of a valid chunk-composed surface prefix.

        Draws base ids, then expands through each level's chunk table. Returns
        (num_surface_tokens, surface_dim), or (batch_size, num_surface_tokens,
        surface_dim) if `batch_size` is given. Every surface token is a
        legitimate slot of some base token's nested expansion.
        """
        if num_surface_tokens % self.total != 0:
            raise ValueError(
                f"num_surface_tokens ({num_surface_tokens}) must be a multiple of "
                f"total ({self.total})"
            )
        n_base = num_surface_tokens // self.total
        dev = device if device is not None else next(self.parameters()).device
        lead = () if batch_size is None else (batch_size,)

        ids = torch.randint(0, self.base_teacher.dim, (*lead, n_base), device=dev)
        for l, level in enumerate(self.levels):
            chunks = level.sample(ids)  # (*lead, count, size_l, out_dim[l])
            if l == self.num_levels - 1:
                return chunks.reshape(*lead, num_surface_tokens, self.dim)
            ids = chunks.argmax(dim=-1).reshape(*lead, -1)  # next-level id stream
        raise AssertionError("unreachable")  # pragma: no cover

from typing import Optional

import torch

from src.teachers.base import ARTeacher
from src.teachers.chunk_code import ChunkCode
from src.teachers.multilevel import MultiLevelHierarchicalTeacher


class HierarchicalTeacher(MultiLevelHierarchicalTeacher):
    """Single-level chunk-composed teacher — the ``L=1`` case of
    :class:`MultiLevelHierarchicalTeacher`.

    Kept as a convenience wrapper with a flat constructor
    (``chunk_dim``/``chunk_size``/``num_tuples``) and surface<->hidden aliases,
    so existing configs and call sites keep working. All of the actual machinery
    (``next_token_log_probs``, ``predict_next``, ``unroll``, ``latent_beliefs``,
    ``sample_surface_prefix``, the Bayes fold) is inherited from the multi-level
    base with a single :class:`ChunkCode` level.

    The base teacher operates over a hidden vocabulary of size ``base_teacher.dim``.
    Each hidden token maps to ``num_tuples`` (M) length-``chunk_size`` surface
    tuples over ``chunk_dim`` symbols, with globally disjoint supports so that
    surface->hidden decoding stays deterministic even when M > 1.
    """

    def __init__(
        self,
        base_teacher: ARTeacher,
        chunk_dim: int,
        chunk_size: int,
        num_tuples: int = 1,
        chunk_seed: Optional[int] = None,
        chunk_table: Optional[torch.Tensor] = None,
    ) -> None:
        level = ChunkCode(
            in_dim=base_teacher.dim,
            out_dim=chunk_dim,
            size=chunk_size,
            num_tuples=num_tuples,
            chunk_seed=chunk_seed,
            chunk_table=chunk_table,
        )
        super().__init__(base_teacher=base_teacher, levels=[level])

    # --- single-level convenience aliases (back-compat) ---
    @property
    def _level(self) -> ChunkCode:
        return self.levels[0]

    @property
    def chunk_dim(self) -> int:
        return self._level.out_dim

    @property
    def chunk_size(self) -> int:
        return self._level.size

    @property
    def num_tuples(self) -> int:
        return self._level.num_tuples

    @property
    def _chunk_table(self) -> torch.Tensor:
        return self._level._chunk_table

    @property
    def _chunk_slot_indices(self) -> torch.Tensor:
        return self._level._chunk_slot_indices

    def _decode_chunk_aligned(self, surface: torch.Tensor) -> torch.Tensor:
        """(..., L_h * chunk_size, chunk_dim) -> (..., L_h, hidden_dim) one-hot."""
        return self._level.decode(surface)

    def _compat_mask(
        self, observed_slots: torch.Tensor, num_observed: int
    ) -> torch.Tensor:
        return self._level._compat_mask(observed_slots, num_observed)

    def _hidden_probs_to_surface_logprobs(
        self,
        hidden_log_probs: torch.Tensor,
        observed_slots: torch.Tensor,
        num_observed: int,
        slot_to_predict: int,
    ) -> torch.Tensor:
        return self._level.next_slot_logprobs(
            hidden_log_probs, observed_slots, num_observed, slot_to_predict
        )

    def with_lag_restriction(self, k: int) -> "HierarchicalTeacher":
        """Restrict the underlying base teacher's lags; keep the same chunk table."""
        restricted_base = self.base_teacher.with_lag_restriction(k)
        return HierarchicalTeacher(
            base_teacher=restricted_base,
            chunk_dim=self.chunk_dim,
            chunk_size=self.chunk_size,
            num_tuples=self.num_tuples,
            chunk_table=self._chunk_table,
        )

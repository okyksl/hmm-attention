from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkCode(nn.Module):
    """One level of a chunk-composed code: maps an input alphabet to fixed
    surface tuples over an output alphabet.

    Each input id (of `in_dim` symbols) maps to `num_tuples` (M) distinct
    length-`size` tuples over an output alphabet of `out_dim` symbols, via a
    chunk table with **globally disjoint supports** across (in_id, tuple_idx).
    Because supports are disjoint, any complete tuple uniquely identifies its
    input id, so output->input decoding is deterministic even with M > 1.

    This is the single-level machinery shared by `HierarchicalTeacher` (which
    holds one such level implicitly) and `MultiLevelHierarchicalTeacher` (which
    stacks a list of them). The naming generalizes `HierarchicalTeacher`'s
    single level: `in_dim` plays "hidden_dim", `out_dim` plays "chunk_dim",
    `size` plays "chunk_size".

    The forward direction (in->out) is stochastic when M > 1: each input id
    uniformly draws one of its M tuples. `next_slot_logprobs` therefore
    marginalizes over both an input posterior *and* the within-input tuple
    posterior conditioned on any observed slots of the current chunk. Crucially
    it accepts an **arbitrary input log-prob distribution**, not just a delta —
    which is what lets a multi-level stack chain levels by folding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        size: int,
        num_tuples: int = 1,
        chunk_seed: Optional[int] = None,
        chunk_table: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("size must be a positive integer.")
        if size > out_dim:
            raise ValueError("size cannot exceed out_dim.")
        if num_tuples <= 0:
            raise ValueError("num_tuples must be a positive integer.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = size
        self.num_tuples = num_tuples

        if chunk_table is None:
            generator = None
            if chunk_seed is not None:
                generator = torch.Generator().manual_seed(int(chunk_seed))
            chunk_table = self._generate_unique_chunks(generator=generator)
        else:
            expected = (in_dim, num_tuples, size, out_dim)
            if tuple(chunk_table.shape) != expected:
                raise ValueError(
                    f"chunk_table shape {tuple(chunk_table.shape)} != expected {expected}"
                )

        self.register_buffer("_chunk_table", chunk_table)
        self.register_buffer("_chunk_slot_indices", chunk_table.argmax(dim=-1))

    # --- construction ---
    def _generate_unique_chunks(
        self, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        needed = self.in_dim * self.num_tuples
        total_chunks = self.out_dim**self.size
        if total_chunks < needed:
            raise ValueError(
                "Not enough unique surface chunks to cover all input tokens "
                f"with {self.num_tuples} tuples each. "
                f"Need {needed}, but only {total_chunks} available."
            )

        # Treat a chunk as a base-`out_dim` integer. This samples from the
        # Cartesian product, so repeated surface symbols such as (3, 3) are
        # valid. Sampling encoded chunks without replacement guarantees global
        # uniqueness even when every possible chunk is used.
        encoded = torch.randperm(total_chunks, generator=generator)[:needed]
        slot_indices = torch.empty(needed, self.size, dtype=torch.long)
        remainder = encoded
        for slot in range(self.size - 1, -1, -1):
            slot_indices[:, slot] = remainder % self.out_dim
            remainder = remainder // self.out_dim

        chunks = F.one_hot(slot_indices, num_classes=self.out_dim).float()
        return chunks.reshape(self.in_dim, self.num_tuples, self.size, self.out_dim)

    # --- decoding (output-alphabet surface -> input-alphabet one-hots) ---
    def decode(self, surface: torch.Tensor, return_tuple: bool = False):
        """(..., L * size, out_dim) -> (..., L, in_dim) one-hot.

        With num_tuples > 1, matches the observed slot sequence against any of
        the M tuples per input id. Supports are globally disjoint, so exactly
        one (in_id, tuple_idx) can match a valid input. Invalid inputs fall back
        to input id 0 (argmax of an all-False match row).

        If `return_tuple`, also return the realized tuple index per position
        (..., L) — which of the M spellings produced each chunk. This is the
        label the spelling-generalization analysis holds out.
        """
        *lead, l_surf, cd = surface.shape
        if cd != self.out_dim:
            raise ValueError(f"Trailing dim {cd} != out_dim {self.out_dim}")
        if l_surf % self.size != 0:
            raise ValueError(
                f"Surface length {l_surf} is not chunk-aligned (size={self.size})"
            )
        l = l_surf // self.size
        chunks = surface.reshape(*lead, l, self.size, self.out_dim)
        slot_idx = chunks.argmax(dim=-1)  # (..., L, size)
        matches = (
            slot_idx.unsqueeze(-2).unsqueeze(-2) == self._chunk_slot_indices
        ).all(dim=-1)  # (..., L, in_dim, M)
        flat = matches.reshape(*matches.shape[:-2], self.in_dim * self.num_tuples)
        combined = flat.float().argmax(dim=-1)  # (..., L) index into in_dim * M
        in_ids = combined // self.num_tuples
        one_hot = F.one_hot(in_ids, num_classes=self.in_dim).to(surface.dtype)
        if return_tuple:
            return one_hot, combined % self.num_tuples
        return one_hot

    # --- marginalization (input distribution -> output-slot distribution) ---
    def _compat_mask(
        self, observed_slots: torch.Tensor, num_observed: int
    ) -> torch.Tensor:
        """Compatibility mask over (in_id, tuple_idx) given leading slots.

        Returns shape (..., in_dim, num_tuples). Entry (h, m) is 1 iff
        `_chunk_slot_indices[h, m, :num_observed] == observed_slots[..., :num_observed]`.
        For num_observed == 0, all entries are 1.
        """
        lead_shape = observed_slots.shape[:-1]
        if num_observed == 0:
            return torch.ones(
                (*lead_shape, self.in_dim, self.num_tuples),
                device=observed_slots.device,
                dtype=torch.float,
            )
        obs = observed_slots[..., :num_observed]  # (..., num_observed)
        table = self._chunk_slot_indices[:, :, :num_observed]  # (in_dim, M, num_observed)
        matches = (obs.unsqueeze(-2).unsqueeze(-2) == table).all(dim=-1)  # (..., in_dim, M)
        return matches.to(torch.float)

    def next_slot_logprobs(
        self,
        in_log_probs: torch.Tensor,
        observed_slots: torch.Tensor,
        num_observed: int,
        slot_to_predict: int,
    ) -> torch.Tensor:
        """Marginalize over (in_id, tuple_idx) to get output-slot log-probs.

        Uses a uniform 1/M prior over tuples per input id. Compatibility
        filtering conditions the posterior on the observed slot prefix; the
        result is mixed through the chunk table at `slot_to_predict`.

        `in_log_probs` may be *any* distribution over the input alphabet (not
        just a one-hot / delta) — the marginalization composes linearly, which
        is exactly what a multi-level fold needs.
        """
        p_h = in_log_probs.exp()  # (..., in_dim)
        # Uniform tuple prior 1/M; broadcast into (..., in_dim, M)
        p_hm = p_h.unsqueeze(-1) / self.num_tuples
        compat = self._compat_mask(observed_slots, num_observed)  # (..., in_dim, M)
        p_posterior = p_hm * compat
        norm = p_posterior.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-30)
        p_posterior = p_posterior / norm
        # chunk_table[:, :, slot_to_predict, :] -> (in_dim, M, out_dim)
        slot_table = self._chunk_table[:, :, slot_to_predict, :]
        out_probs = torch.einsum("...hm,hmd->...d", p_posterior, slot_table)
        return out_probs.clamp(min=1e-30).log()

    # --- generation (input ids -> output-alphabet surface tuples) ---
    def sample(
        self,
        in_ids: torch.Tensor,
        tuple_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Expand input ids into surface tuples. (...,) -> (..., size, out_dim).

        Draws a uniform tuple per input id unless `tuple_ids` is given.
        """
        table = self._chunk_table
        if tuple_ids is None:
            tuple_ids = torch.randint(
                0, self.num_tuples, in_ids.shape, device=table.device
            )
        return table[in_ids, tuple_ids]  # (..., size, out_dim)

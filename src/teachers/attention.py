from contextlib import contextmanager
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import spectra
from src.model.attention import generate_square_subsequent_mask
from src.model.decoder import DecoderBlock
from src.model.positional import PositionalEncoding
from src.spectra import SpectrumSpec
from src.teachers.base import ADAPTIVE, ARTeacher

# Linear maps of the teacher whose singular values can be shaped by `spectrum`.
SPECTRUM_TARGETS = ("encoder", "readout", "query", "key", "value", "output", "mlp")
DEFAULT_SPECTRUM_TARGETS = ("encoder", "readout", "query", "key", "value", "output")


@contextmanager
def _seeded(seed: Optional[int]):
    """Temporarily seed the global RNG so a random teacher is reproducible.

    Restores the previous RNG state on exit, so constructing a seeded teacher
    does not perturb the surrounding stream (dataset burn-in, student init, ...).
    """
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


class AttentionARTeacher(ARTeacher):
    """Attention-based autoregressive teacher: one self-attention layer + MLP.

    A fixed (random-init, frozen) data-generating process whose next-token
    distribution is a *content-dependent* mixture of the context — the upgrade
    over `LinearARTeacher`, whose span mixing is fixed by position. Concretely:

        one-hot (B, T, dim)
          -> encoder      : Linear(dim -> hidden_dim)          [frozen]
          -> pos encoder  : PositionalEncoding (pe_type flag)  [frozen]
          -> DecoderBlock : causal MultiHeadAttention + MLP     [frozen]
          -> readout      : Linear(hidden_dim -> dim)          [frozen]
          -> logits * scale -> log_softmax

    Memory range is a knob. With `unbounded=True` (default) the teacher attends
    over the *entire* prefix — the whole point of attention, and the only way a
    downstream emission can depend on arbitrarily far context; `context_length`
    reports `ADAPTIVE` and only `burn_in` (the minimum prefix) matters. With
    `unbounded=False` it behaves as a fixed order-`window` process (windowed
    like the other teachers), and `burn_in` defaults to `window`.

    Sharpness is controlled by `scale` (softmax temperature 1/scale), matching
    `LinearARTeacher.from_parameters(scale=...)`. Outputs are **log-probs**.

    Cost note: unbounded autoregressive *generation* is O(L^3) over a length-L
    sequence (O(L^2) per step); `unroll` evaluation stays a single O(L^2) forward.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_heads: int = 1,
        ff_hidden_dim: int = -1,
        window: int = 1,
        unbounded: bool = True,
        burn_in: Optional[int] = None,
        scale: float = 1.0,
        pe_type: str = "absolute",
        pe_learnable: bool = False,
        pe_max_sequence_length: int = 512,
        init_scale: float = 1.0,
        use_query_projection: bool = True,
        use_key_projection: bool = True,
        use_value_projection: bool = True,
        use_output_projection: bool = True,
        use_mlp: bool = True,
        skip_connection: bool = True,
        layer_normalization: bool = True,
        attention_disentanglement: bool = False,
        attention_bias: bool = True,
        spectrum: Union[SpectrumSpec, Mapping[str, Any], None] = None,
        spectrum_targets: Sequence[str] = DEFAULT_SPECTRUM_TARGETS,
        seed: Optional[int] = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if window <= 0:
            raise ValueError(f"window must be positive; got {window}")
        if pe_type == "one_hot":
            raise ValueError(
                "pe_type='one_hot' is not supported by AttentionARTeacher "
                "(it changes the embedding dimension); use 'absolute' or 'none'."
            )

        # Invariant: burn_in == context_length for non-adaptive teachers, so an
        # independent burn_in is only meaningful when unbounded. Bounded teachers
        # pin burn_in to `window` (which is their context_length).
        if unbounded:
            resolved_burn_in = burn_in if burn_in is not None else 1
            if resolved_burn_in <= 0:
                raise ValueError(f"burn_in must be positive; got {resolved_burn_in}")
        else:
            if burn_in is not None and burn_in != window:
                raise ValueError(
                    f"burn_in ({burn_in}) must equal window ({window}) for a bounded "
                    "teacher: burn_in == context_length when not adaptive."
                )
            resolved_burn_in = window

        self._dim = dim
        self._hidden_dim = hidden_dim if hidden_dim is not None else dim
        self._window = window
        self._burn_in = resolved_burn_in
        self.unbounded = unbounded
        self.scale = scale
        self.pe_type = pe_type

        ff = self._hidden_dim if ff_hidden_dim == -1 else ff_hidden_dim

        with _seeded(seed):
            self.encoder = nn.Linear(dim, self._hidden_dim, bias=False)
            nn.init.orthogonal_(self.encoder.weight)
            self.encoder.weight.data.mul_(init_scale)

            self.pos_encoder = PositionalEncoding.create_positional_encoder(
                pe_type=pe_type,
                pe_learnable=pe_learnable,
                pe_embedding_dim=self._hidden_dim,
                pe_max_sequence_length=pe_max_sequence_length,
                pe_dropout=0.0,
                pe_init_scale=init_scale,
            )

            self.block = DecoderBlock(
                d_model=self._hidden_dim,
                num_heads=num_heads,
                ff_hidden_layer=ff,
                dropout=0.0,
                pe_type=pe_type,
                pe_max_sequence_length=pe_max_sequence_length,
                init_scale=init_scale,
                skip_connection=skip_connection,
                layer_normalization=layer_normalization,
                use_query_projection=use_query_projection,
                use_key_projection=use_key_projection,
                use_value_projection=use_value_projection,
                use_output_projection=use_output_projection,
                use_mlp=use_mlp,
                attention_disentanglement=attention_disentanglement,
                attention_bias=attention_bias,
            )

            self.readout = nn.Linear(self._hidden_dim, dim, bias=False)
            nn.init.orthogonal_(self.readout.weight)
            self.readout.weight.data.mul_(init_scale)

            # Optional feature spectrum. Left off (`spectrum: null`) the teacher
            # keeps its default init exactly; supplied, it reshapes the singular
            # values of the selected linear maps, so the process has a graded
            # rather than flat feature importance profile.
            self.spectrum_spec = (
                SpectrumSpec.from_config(spectrum) if spectrum is not None else None
            )
            self.spectrum_targets = tuple(spectrum_targets)
            if self.spectrum_spec is not None:
                self._apply_spectrum(self.spectrum_spec, self.spectrum_targets, init_scale)

        # A teacher is a fixed process: freeze params and disable dropout/train
        # behaviour. Params are never handed to the optimizer regardless, but
        # freezing makes the "fixed data-generating process" contract explicit.
        if freeze:
            self.requires_grad_(False)
        self.eval()

    # --- spectrum shaping ---
    def _spectrum_modules(self, target: str) -> List[nn.Linear]:
        """The `nn.Linear`s behind a target name (may be empty if projections
        are disabled, or several when heads are disentangled)."""
        attn = self.block.self_attention
        lookup = {
            "encoder": self.encoder,
            "readout": self.readout,
            "query": attn.query_proj,
            "key": attn.key_proj,
            "value": attn.value_proj,
            "output": attn.out_proj,
            "mlp": nn.ModuleList([self.block.linear1, self.block.linear2]),
        }
        if target not in lookup:
            raise ValueError(
                f"unknown spectrum target {target!r}; expected one of {SPECTRUM_TARGETS}"
            )
        module = lookup[target]
        candidates = list(module) if isinstance(module, nn.ModuleList) else [module]
        return [m for m in candidates if isinstance(m, nn.Linear)]

    def _apply_spectrum(
        self, spec: SpectrumSpec, targets: Sequence[str], init_scale: float
    ) -> None:
        """Re-draw the selected weight matrices with singular values from `spec`.

        Each weight keeps its shape; the spectrum is materialized at
        `min(out_features, in_features)` and the matrix is scaled by
        `init_scale`, matching the default init's convention.
        """
        with torch.no_grad():
            for target in targets:
                for module in self._spectrum_modules(target):
                    out_f, in_f = module.weight.shape
                    s = spectra.spectrum(spec, min(out_f, in_f))
                    w = spectra.random_matrix_with_spectrum(s, out_f, in_f)
                    module.weight.copy_(w * init_scale)

    # --- ARTeacher interface ---
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def context_length(self) -> int:
        return ADAPTIVE if self.unbounded else self._window

    @property
    def burn_in(self) -> int:
        return self._burn_in

    @property
    def window(self) -> int:
        return self._window

    @property
    def span_lengths(self) -> List[int]:
        # Unit spans summing to burn_in, so downstream helpers that size
        # positional encodings from `calculate_context_length(span_lengths)`
        # recover the burn-in / prefix length.
        return [1] * self._burn_in

    @property
    def stride(self) -> Optional[int]:
        return None

    def _forward_seq(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, dim) one-hot -> (B, T, dim) scaled logits (causal).

        Position t's logits are a function of tokens 0..t only (causal mask),
        so they predict token t+1 under the AR factorization.
        """
        if tokens.shape[-1] != self._dim:
            raise ValueError(
                f"Trailing dim {tokens.shape[-1]} != dim {self._dim}"
            )
        h = self.encoder(tokens)
        h = self.pos_encoder(h)
        mask = generate_square_subsequent_mask(h.size(1), device=h.device)
        h, _ = self.block(h, mask)
        return self.readout(h) * self.scale

    def sequence_log_probs(self, sequence: torch.Tensor) -> torch.Tensor:
        """Causal per-position next-token log-probs, (B, L, dim).

        One forward: position `t` predicts token `t+1` from tokens `0..t`. The
        base `unroll` slices this for the adaptive path, and `next_token_log_probs`
        reads out the last position.
        """
        return F.log_softmax(self._forward_seq(sequence), dim=-1)

    def next_token_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        """Predict the next token from a context block. context: (B, T, dim).

        Attends over all T provided tokens and reads out the last position.
        Returns (B, dim) log-probs.
        """
        return self.sequence_log_probs(context)[:, -1, :]

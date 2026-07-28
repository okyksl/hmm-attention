import warnings
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import spectra
from src.spectra import SpectrumSpec
from src.teachers.base import ARTeacher

SpectrumConfig = Union[SpectrumSpec, Mapping[str, Any], None]


class LinearARTeacher(ARTeacher):
    """Linear autoregressive teacher over one-hot / dense token spaces.

    The teacher looks at a fixed context of `context_length` tokens, splits it
    into `window` variable-length spans (`span_lengths[i]` tokens per span),
    aggregates each span (sum, or weighted sum via `span_position_weights`),
    and computes a linear combination of the per-span aggregates using
    per-lag weight matrices `_params[i]` shape `(dim, dim)`. Optionally, a
    `stride` schedule places span start positions with overlap.

    Each weight matrix is generated as `A_i = scale · w_i · U_i diag(ŝ_i) V_iᵀ`
    (see `src.spectra`): an **inner** spectrum `ŝ_i` — normalized, so it only
    sets the *shape* of the feature importances — scaled by an **outer**
    per-lag weight `w_i` that sets the matrix norm.

    `next_token_log_probs` returns log-probabilities (log-softmax of the raw
    linear output). Distribution sharpness is controlled by the `scale`
    argument in `from_parameters`, which multiplies the weight matrices —
    equivalent to a softmax temperature of `1/scale`.
    """

    def __init__(
        self,
        params: torch.Tensor,
        span_lengths: List[int],
        stride: Optional[int] = None,
        span_position_weights: Optional[List[float]] = None,
        scale: float = 1.0,
        singular_values: Optional[torch.Tensor] = None,
        lag_weights: Optional[torch.Tensor] = None,
        spectrum_specs: Optional[Sequence[SpectrumSpec]] = None,
        lag_spectrum_spec: Optional[SpectrumSpec] = None,
        shared_bases: bool = False,
        orthogonality: str = "none",
    ) -> None:
        super().__init__()
        window, dim, dim2 = params.shape
        if dim != dim2:
            raise ValueError(f"params must be (window, dim, dim); got {tuple(params.shape)}")
        if len(span_lengths) != window:
            raise ValueError(
                f"span_lengths length {len(span_lengths)} must equal window {window}"
            )

        self._params = nn.Parameter(data=params.detach().clone())
        self._dim = dim
        self._window = window
        self._span_lengths = list(span_lengths)
        self._stride = stride
        self._span_position_weights = (
            list(span_position_weights) if span_position_weights is not None else None
        )

        # Metadata (preserved for verbose logging / analysis; not used in forward).
        self.scale = scale
        self.spectrum_specs = list(spectrum_specs) if spectrum_specs else [SpectrumSpec()] * window
        self.lag_spectrum_spec = lag_spectrum_spec or SpectrumSpec(normalize="none")
        self.shared_bases = shared_bases
        self.orthogonality = orthogonality
        self.register_buffer(
            "singular_values",
            singular_values if singular_values is not None else torch.ones(window, dim),
            persistent=False,
        )
        self.register_buffer(
            "lag_weights",
            lag_weights if lag_weights is not None else torch.ones(window),
            persistent=False,
        )

    # --- ARTeacher interface ---
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def rank(self) -> int:
        """Largest number of non-zero singular values across the lag matrices."""
        return int((self.singular_values > 0).sum(dim=-1).max())

    @property
    def context_length(self) -> int:
        if self._stride is not None:
            return (self._window - 1) * self._stride + self._span_lengths[-1]
        return sum(self._span_lengths)

    @property
    def window(self) -> int:
        return self._window

    @property
    def span_lengths(self) -> List[int]:
        return list(self._span_lengths)

    @property
    def stride(self) -> Optional[int]:
        return self._stride

    @property
    def span_position_weights(self) -> Optional[List[float]]:
        return list(self._span_position_weights) if self._span_position_weights else None

    def _get_weights(self) -> torch.Tensor:
        return self._params

    def next_token_log_probs(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, context_length, dim). Returns (B, dim) log-probs."""
        if context.shape[-2] != self.context_length:
            raise ValueError(
                f"context has {context.shape[-2]} tokens; expected exactly {self.context_length}"
            )

        # Aggregate each span into a single (dim,)-vector per batch.
        aggregates = []
        for lag_idx, span_len in enumerate(self._span_lengths):
            if self._stride is not None:
                start_idx = lag_idx * self._stride
            else:
                start_idx = sum(self._span_lengths[:lag_idx])
            end_idx = start_idx + span_len
            span_tokens = context[..., start_idx:end_idx, :]  # (B, span_len, dim)

            if self._span_position_weights is not None:
                w = torch.tensor(
                    self._span_position_weights,
                    device=span_tokens.device,
                    dtype=span_tokens.dtype,
                ).view(1, span_len, 1)
                aggregates.append((span_tokens * w).sum(dim=-2))
            else:
                aggregates.append(span_tokens.sum(dim=-2))
        # (B, window, dim)
        x_agg = torch.stack(aggregates, dim=-2)

        # (window, dim, dim) x (B, window, dim, 1) -> (B, window, dim, 1) -> sum -> (B, dim)
        weights = self._params.unsqueeze(0)  # (1, window, dim, dim)
        out = torch.matmul(weights, x_agg.unsqueeze(-1)).squeeze(-1).sum(dim=-2)
        return F.log_softmax(out, dim=-1)

    # --- Lag restriction ---
    def with_lag_restriction(self, k: int) -> "LinearARTeacher":
        """Return a shallow-copy teacher restricted to k of the `window` lags."""
        if k <= 0 or k > self._window:
            raise ValueError(f"k={k} must be in [1, window={self._window}]")

        # Keep the contiguous end carrying the most outer weight — the lags the
        # process actually relies on. For a monotone outer law this is just
        # "drop the tail of the decay".
        # Ties (a flat outer law) fall through to the most recent spans, the
        # usual order-k Markov restriction.
        w = self.lag_weights
        if float(w[:k].sum()) > float(w[self._window - k :].sum()):
            sl = slice(None, k)
        else:
            sl = slice(self._window - k, None)

        return LinearARTeacher(
            params=self._params[sl],
            span_lengths=self._span_lengths[sl],
            stride=self._stride,
            span_position_weights=self._span_position_weights,
            scale=self.scale,
            singular_values=self.singular_values[sl],
            lag_weights=self.lag_weights[sl],
            spectrum_specs=self.spectrum_specs[sl],
            lag_spectrum_spec=self.lag_spectrum_spec,
            shared_bases=self.shared_bases,
            orthogonality=self.orthogonality,
        )

    # --- Constructor helpers ---
    @classmethod
    def from_parameters(
        cls,
        dim: int,
        span_lengths: List[int],
        window: int = 1,
        scale: float = 1.0,
        spectrum: SpectrumConfig = None,
        lag_spectrum: SpectrumConfig = None,
        shared_bases: bool = False,
        orthogonality: str = "none",
        stride: Optional[int] = None,
        span_position_weights: Optional[List[float]] = None,
    ) -> "LinearARTeacher":
        """Build a teacher from an inner (feature) and outer (lag) spectrum.

        Args:
            spectrum: inner singular-value law, applied within each lag matrix.
                Normalized (`normalize='spectral'` by default), so it sets the
                shape of the feature importances, not the norm. `law`, `decay`,
                `alpha`, and `rank` may each be a length-`window` list to vary
                the spectrum per lag.
            lag_spectrum: outer law over lags. Un-normalized by default, so
                `w_i` literally sets lag `i`'s matrix norm. `reverse=True` puts
                the largest weight on the last (most recent) span.
            shared_bases: reuse one (U, V) pair across lags, so the matrices
                differ only through their singular values.
            orthogonality: `none`, `disjoint` (spectrum-preserving mutual
                Frobenius-orthogonality), or `gram_schmidt`. See `src.spectra`.
        """
        assert dim > 0, f"Dimension {dim} must be positive"
        assert window > 0, f"Window {window} must be positive"
        assert scale > 0, f"Scale {scale} must be positive"

        if stride is not None:
            assert stride > 0, f"Stride {stride} must be positive"
            min_span = min(span_lengths)
            if stride > min_span:
                warnings.warn(
                    f"Stride ({stride}) > min span_length ({min_span}) creates gaps between intervals"
                )

        if span_position_weights is not None:
            if len(set(span_lengths)) != 1:
                raise ValueError(
                    f"All span_lengths must be equal when using span_position_weights. "
                    f"Got: {span_lengths}"
                )
            span_len = span_lengths[0]
            if len(span_position_weights) != span_len:
                raise ValueError(
                    f"span_position_weights length ({len(span_position_weights)}) "
                    f"must match span_length ({span_len})"
                )
            weight_sum = sum(span_position_weights)
            if weight_sum <= 0:
                raise ValueError(
                    f"span_position_weights must sum to a positive value. Got sum: {weight_sum}"
                )
            span_position_weights = [w / weight_sum for w in span_position_weights]

        # Inner: one normalized singular-value profile per lag.
        specs = spectra.expand_specs(spectrum, window)
        for spec in specs:
            if spec.rank != -1 and spec.rank > dim:
                raise ValueError(f"spectrum rank {spec.rank} must be <= dim {dim}")
        matrices = spectra.random_matrices(
            n=window,
            dim=dim,
            specs=specs,
            shared_bases=shared_bases,
            orthogonality=orthogonality,
        )

        # Outer: one scalar per lag, setting each matrix's norm.
        lag_spec = spectra.SpectrumSpec.from_config(
            lag_spectrum if lag_spectrum is not None else {"normalize": "none"}
        )
        lag_weights = spectra.spectrum(lag_spec, window)

        A = torch.stack([m * w for m, w in zip(matrices, lag_weights)], dim=0)
        A *= scale

        return cls(
            params=A,
            span_lengths=span_lengths,
            stride=stride,
            span_position_weights=span_position_weights,
            scale=scale,
            singular_values=torch.stack([spectra.spectrum(s, dim) for s in specs]),
            lag_weights=lag_weights,
            spectrum_specs=specs,
            lag_spectrum_spec=lag_spec,
            shared_bases=shared_bases,
            orthogonality=orthogonality,
        )

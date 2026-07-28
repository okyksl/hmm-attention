"""Spectrum-controlled random matrix generation.

A matrix in this codebase is generated as

    A = w · U diag(s) Vᵀ

where `s` is a **normalized** singular-value profile drawn from a *law* (the
inner / feature spectrum) and `w` is a scalar drawn from a second law over
whatever the outer index is (lags, for `LinearARTeacher`). The inner law sets
the *shape* of the feature importances; the outer law sets the *norm*.

Both are described by the same `SpectrumSpec`, so "geometric decay across lags"
and "power-law decay across features" are one mechanism used twice.
"""

import warnings
from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Sequence, Union

import torch

LAWS = ("flat", "geometric", "power")
NORMALIZATIONS = ("spectral", "frobenius", "sum", "none")
ORTHOGONALITIES = ("none", "disjoint", "gram_schmidt")

# Fields that may be given per-index (a list of length n) instead of as a scalar.
_BROADCASTABLE = ("law", "decay", "alpha", "rank")


@dataclass(frozen=True)
class SpectrumSpec:
    """A decreasing non-negative profile of length `n`.

    Attributes:
        law: `flat` (all ones), `geometric` (`decay**k`), or `power`
            (`(k+1)**-alpha`).
        decay: base of the geometric law. `1.0` makes it flat.
        alpha: exponent of the power law. `0.0` makes it flat.
        rank: keep the top `rank` entries and zero the rest; `-1` keeps all.
        normalize: `spectral` (max entry = 1), `frobenius` (‖·‖₂ = 1), `sum`
            (entries sum to 1), or `none`.
        reverse: flip the profile, so the *last* index carries the most mass.
    """

    law: str = "flat"
    decay: float = 1.0
    alpha: float = 1.0
    rank: int = -1
    normalize: str = "spectral"
    reverse: bool = False

    def __post_init__(self) -> None:
        if self.law not in LAWS:
            raise ValueError(f"law must be one of {LAWS}; got {self.law!r}")
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {NORMALIZATIONS}; got {self.normalize!r}"
            )
        if self.law == "geometric" and self.decay <= 0:
            raise ValueError(f"decay must be positive; got {self.decay}")
        if self.rank != -1 and self.rank <= 0:
            raise ValueError(f"rank must be positive or -1; got {self.rank}")

    @classmethod
    def from_config(cls, cfg: Union["SpectrumSpec", Mapping[str, Any], None]) -> "SpectrumSpec":
        """Coerce `None` / a mapping (incl. an OmegaConf node) into a spec."""
        if cfg is None:
            return cls()
        if isinstance(cfg, SpectrumSpec):
            return cfg
        fields = {k: cfg[k] for k in cfg.keys()}
        unknown = set(fields) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown spectrum fields: {sorted(unknown)}")
        return cls(**fields)


def expand_specs(
    cfg: Union[SpectrumSpec, Mapping[str, Any], None], n: int
) -> List[SpectrumSpec]:
    """Resolve a possibly per-index spectrum config into `n` scalar specs.

    Any of `law`, `decay`, `alpha`, `rank` may be given as a length-`n` list to
    vary that field across the outer index — e.g. `rank: [8, 4, 2]` gives each
    lag its own effective dimensionality while sharing everything else.
    """
    if cfg is None or isinstance(cfg, SpectrumSpec):
        return [SpectrumSpec.from_config(cfg)] * n

    scalar = {}
    per_index = {}
    for key in cfg.keys():
        value = cfg[key]
        if key in _BROADCASTABLE and isinstance(value, (list, tuple, Sequence)) and not isinstance(value, str):
            value = list(value)
            if len(value) != n:
                raise ValueError(
                    f"per-index spectrum field {key!r} has length {len(value)}; expected {n}"
                )
            per_index[key] = value
        else:
            scalar[key] = value

    base = SpectrumSpec.from_config(scalar)
    if not per_index:
        return [base] * n
    return [replace(base, **{k: v[i] for k, v in per_index.items()}) for i in range(n)]


def spectrum(
    spec: SpectrumSpec, n: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Materialize `spec` as a length-`n` non-negative tensor."""
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    k = torch.arange(n, dtype=dtype)
    if spec.law == "flat":
        s = torch.ones(n, dtype=dtype)
    elif spec.law == "geometric":
        s = torch.full((n,), float(spec.decay), dtype=dtype) ** k
    else:  # power
        s = (k + 1.0) ** (-float(spec.alpha))

    rank = n if spec.rank == -1 else min(spec.rank, n)
    if rank < n:
        s[rank:] = 0.0

    # Truncate first, then flip: `reverse` with a rank cutoff puts the surviving
    # entries at the *end*, which is what "the last index matters most" means.
    if spec.reverse:
        s = torch.flip(s, dims=(0,))

    if spec.normalize == "spectral":
        denom = s.max()
    elif spec.normalize == "frobenius":
        denom = torch.linalg.norm(s)
    elif spec.normalize == "sum":
        denom = s.sum()
    else:
        denom = torch.ones((), dtype=dtype)
    if denom <= 0:
        raise ValueError(f"spectrum from {spec} cannot be normalized (norm is 0)")
    return s / denom


def effective_rank(s: torch.Tensor) -> float:
    """Participation ratio (‖s‖₁² / ‖s‖₂²) — a soft count of active directions."""
    l1 = s.abs().sum()
    l2 = torch.linalg.norm(s)
    if l2 <= 0:
        return 0.0
    return float(l1**2 / l2**2)


def random_orthogonal(d: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A Haar-random d×d orthogonal matrix."""
    return torch.linalg.svd(torch.randn(d, d, dtype=dtype))[0]


def random_matrix_with_spectrum(
    s: torch.Tensor, rows: int, cols: Optional[int] = None
) -> torch.Tensor:
    """A random `rows × cols` matrix whose singular values are `s`.

    `s` is truncated / zero-padded to `min(rows, cols)`. The left and right
    singular bases are Haar-random and independent.
    """
    cols = rows if cols is None else cols
    k = min(rows, cols)
    U, _, Vh = torch.linalg.svd(torch.randn(rows, cols), full_matrices=False)
    sv = torch.zeros(k, dtype=U.dtype)
    sv[: min(k, s.numel())] = s[:k].to(U.dtype)
    return (U * sv) @ Vh


def random_matrices(
    n: int,
    dim: int,
    specs: Union[SpectrumSpec, Sequence[SpectrumSpec]],
    shared_bases: bool = False,
    orthogonality: str = "none",
) -> List[torch.Tensor]:
    """Generate `n` square `dim × dim` matrices with prescribed spectra.

    Args:
        specs: one spec, or one per matrix (see `expand_specs`).
        shared_bases: reuse a single (U, V) pair across all matrices, so the
            matrices differ only in their singular values — "same features,
            different importance".
        orthogonality: how the matrices relate to each other.
            * `none` — independent draws.
            * `disjoint` — each matrix gets a disjoint block of columns from a
              shared random orthogonal basis on both sides. Gives exact
              Frobenius-orthogonality (⟨A_i, A_j⟩_F = 0, since the column spaces
              are orthogonal) *and* exact prescribed spectra. Requires the
              effective ranks to sum to at most `dim`.
            * `gram_schmidt` — orthogonalize the flattened matrices in ℝ^{d²}.
              Exactly Frobenius-orthogonal but it does **not** preserve the
              prescribed singular values, so it is only faithful for a flat
              spectrum.
    """
    if orthogonality not in ORTHOGONALITIES:
        raise ValueError(
            f"orthogonality must be one of {ORTHOGONALITIES}; got {orthogonality!r}"
        )
    if isinstance(specs, SpectrumSpec):
        specs = [specs] * n
    if len(specs) != n:
        raise ValueError(f"got {len(specs)} specs for {n} matrices")

    svals = [spectrum(spec, dim) for spec in specs]

    if orthogonality == "disjoint":
        if shared_bases:
            raise ValueError(
                "orthogonality='disjoint' assigns each matrix its own singular "
                "subspace, which is incompatible with shared_bases=True."
            )
        ranks = [int((s > 0).sum()) for s in svals]
        if sum(ranks) > dim:
            raise ValueError(
                f"orthogonality='disjoint' needs the effective ranks {ranks} to sum to "
                f"at most dim={dim} (got {sum(ranks)}); lower the rank or raise dim."
            )
        left = random_orthogonal(dim)
        right = random_orthogonal(dim)
        matrices = []
        offset = 0
        for s, r in zip(svals, ranks):
            cols = slice(offset, offset + r)
            matrices.append((left[:, cols] * s[:r]) @ right[:, cols].T)
            offset += r
        return matrices

    if shared_bases:
        U, _, Vh = torch.linalg.svd(torch.randn(dim, dim))
        matrices = [(U * s) @ Vh for s in svals]
    else:
        matrices = [random_matrix_with_spectrum(s, dim) for s in svals]

    if orthogonality == "gram_schmidt":
        # A flat, full-rank spectrum is the only one Gram-Schmidt leaves intact
        # (it is the isotropic case); anything shaped or truncated gets mangled.
        if any(spec.law != "flat" or spec.rank != -1 for spec in specs):
            warnings.warn(
                "orthogonality='gram_schmidt' orthogonalizes in R^{d*d} and does not "
                "preserve the prescribed singular values; use 'disjoint' to keep both."
            )
        matrices = _gram_schmidt_matrices(matrices)

    return matrices


def _gram_schmidt_matrices(matrices: List[torch.Tensor]) -> List[torch.Tensor]:
    """Make matrices mutually Frobenius-orthogonal, restoring the first's norm."""
    d = matrices[0].shape[0]
    if len(matrices) > d * d:
        raise ValueError(
            f"cannot generate {len(matrices)} orthogonal matrices in R^{d}x{d}"
        )
    target_norm = matrices[0].norm()

    ortho_vecs: List[torch.Tensor] = []
    for m in matrices:
        v = m.flatten()
        for u in ortho_vecs:
            v = v - torch.dot(v, u) * u
        norm = v.norm()
        if norm > 1e-10:
            v = v / norm
        ortho_vecs.append(v)
    return [v.reshape(d, d) * target_norm for v in ortho_vecs]

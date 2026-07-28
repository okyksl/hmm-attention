from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


def calculate_context_length(span_lengths: List[int], stride: Optional[int] = None) -> int:
    """Total context length accounting for stride (last-span end - first-span start)."""
    if stride is not None:
        return (len(span_lengths) - 1) * stride + span_lengths[-1]
    return sum(span_lengths)


def _resolve_sentinel(value: int, default: int) -> int:
    """Resolve the `-1` sentinel used across configs to a concrete default."""
    return default if value == -1 else value


def preprocess_cfg(cfg: DictConfig) -> DictConfig:
    """Normalize teacher/student/ngram configs (mutates + returns cfg).

    Resolves `-1` sentinels (dim, rank, window, hidden_dim) against the
    corresponding upstream default, and merges ngram configs on top of the
    student config so they inherit its architecture.
    """
    # teacher. Spectrum `rank: -1` needs no resolution here — `src.spectra`
    # resolves it against the spectrum length at construction.
    if "dim" in cfg.teacher:
        cfg.teacher.dim = _resolve_sentinel(cfg.teacher.dim, cfg.dataset.dim)
    if "window" in cfg.teacher:
        cfg.teacher.window = _resolve_sentinel(cfg.teacher.window, cfg.dataset.window)
    if "hidden_dim" in cfg.teacher:
        cfg.teacher.hidden_dim = _resolve_sentinel(
            cfg.teacher.hidden_dim, cfg.teacher.dim
        )
    # MultiLevelHierarchicalTeacher: resolve each level's `chunk_dim` sentinel
    # against the surface vocab (dataset.dim). Intermediate alphabets should be
    # set explicitly; -1 defaults to the surface dim.
    if "levels" in cfg.teacher:
        for level in cfg.teacher.levels:
            if "chunk_dim" in level:
                level.chunk_dim = _resolve_sentinel(level.chunk_dim, cfg.dataset.dim)

    # student
    if "student" in cfg:
        cfg.student.dim = _resolve_sentinel(cfg.student.dim, cfg.dataset.dim)
        if "window" in cfg.student:
            cfg.student.window = _resolve_sentinel(
                cfg.student.window, cfg.teacher.window
            )
        if "hidden_dim" in cfg.student:
            cfg.student.hidden_dim = _resolve_sentinel(
                cfg.student.hidden_dim, cfg.student.dim
            )

    # ngrams inherit student's architecture, then override _target_ + ngram.
    if "ngrams" in cfg:
        for name, ngram_cfg in cfg.ngrams.items():
            merged = OmegaConf.create(cfg.student)
            merged._target_ = ngram_cfg._target_
            merged.ngram = ngram_cfg.ngram
            cfg.ngrams[name] = merged

    return cfg


def configure_positional_encoding(cfg: DictConfig, teacher) -> None:
    """Set student/ngram PE hidden + embedding dims from the *instantiated* teacher.

    Must run after teacher instantiation because `HierarchicalTeacher` exposes
    surface-scaled `span_lengths` and `stride` only once constructed.
    """
    if "student" not in cfg or "pe_type" not in cfg.student:
        return

    pe_type = cfg.student["pe_type"]
    span_lengths = list(teacher.span_lengths)
    stride = getattr(teacher, "stride", None)

    if pe_type == "one_hot":
        prefix_length = calculate_context_length(span_lengths, stride)
        embedding_dim = prefix_length + cfg.dataset.length - 1
        cfg.student.hidden_dim = cfg.dataset.dim + embedding_dim
        cfg.student.pe_embedding_dim = embedding_dim
    if pe_type == "absolute":
        cfg.student.pe_embedding_dim = cfg.student.hidden_dim

    if "ngrams" not in cfg:
        return

    for name, ngram_cfg in cfg.ngrams.items():
        if pe_type == "one_hot":
            n = ngram_cfg.ngram
            if stride is not None:
                embedding_dim = (n - 1) * stride + span_lengths[n - 1]
            else:
                embedding_dim = sum(span_lengths[:n])
            hidden_dim = cfg.dataset.dim + embedding_dim
        else:
            embedding_dim = cfg.student.pe_embedding_dim
            hidden_dim = cfg.student.hidden_dim
        cfg.ngrams[name].pe_embedding_dim = embedding_dim
        cfg.ngrams[name].hidden_dim = hidden_dim

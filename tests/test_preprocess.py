from omegaconf import OmegaConf

from src.runner.preprocess import (
    calculate_context_length,
    configure_positional_encoding,
    preprocess_cfg,
)


# ---- calculate_context_length -----------------------------------------------


def test_calculate_context_length_no_stride():
    assert calculate_context_length([2, 3, 1], stride=None) == 6


def test_calculate_context_length_with_stride():
    # (n-1)*stride + last_span = 2*2 + 3 = 7
    assert calculate_context_length([1, 1, 3], stride=2) == 7


# ---- preprocess_cfg (sentinel resolution) -----------------------------------


def _base_cfg():
    return OmegaConf.create(
        {
            "dataset": {"dim": 16, "window": 3, "length": 20},
            "teacher": {"dim": -1, "window": -1},
            "student": {"dim": -1, "window": -1, "hidden_dim": -1},
        }
    )


def test_preprocess_cfg_resolves_teacher_sentinels():
    cfg = _base_cfg()
    preprocess_cfg(cfg)
    assert cfg.teacher.dim == cfg.dataset.dim  # -1 → dataset.dim
    assert cfg.teacher.window == cfg.dataset.window


def test_preprocess_cfg_resolves_student_sentinels():
    cfg = _base_cfg()
    preprocess_cfg(cfg)
    assert cfg.student.dim == cfg.dataset.dim
    assert cfg.student.window == cfg.teacher.window
    assert cfg.student.hidden_dim == cfg.student.dim


def test_preprocess_cfg_preserves_explicit_values():
    cfg = _base_cfg()
    cfg.teacher.dim = 8
    preprocess_cfg(cfg)
    assert cfg.teacher.dim == 8


def test_preprocess_cfg_leaves_spectrum_rank_sentinel_to_src_spectra():
    # `rank: -1` resolves against the spectrum length at construction, not here.
    cfg = _base_cfg()
    cfg.teacher.spectrum = {"rank": -1}
    preprocess_cfg(cfg)
    assert cfg.teacher.spectrum.rank == -1


def test_preprocess_cfg_merges_ngram_on_top_of_student():
    cfg = _base_cfg()
    cfg.ngrams = {
        "bigram": OmegaConf.create({"_target_": "path.To.Ngram", "ngram": 2}),
    }
    preprocess_cfg(cfg)
    merged = cfg.ngrams.bigram
    assert merged.hidden_dim == cfg.student.hidden_dim
    assert merged.dim == cfg.student.dim
    assert merged._target_ == "path.To.Ngram"
    assert merged.ngram == 2


# ---- configure_positional_encoding ------------------------------------------


class _StubTeacher:
    """Minimal duck-typed teacher for PE configuration tests."""

    def __init__(self, span_lengths, stride=None):
        self.span_lengths = span_lengths
        self.stride = stride


def test_configure_pe_absolute_sets_embedding_to_hidden():
    cfg = OmegaConf.create(
        {
            "dataset": {"dim": 16, "length": 20},
            "student": {"pe_type": "absolute", "hidden_dim": 32, "pe_embedding_dim": -1},
            "ngrams": {},
        }
    )
    configure_positional_encoding(cfg, _StubTeacher([1, 1, 1]))
    assert cfg.student.pe_embedding_dim == cfg.student.hidden_dim


def test_configure_pe_one_hot_sets_dims_from_prefix_length():
    cfg = OmegaConf.create(
        {
            "dataset": {"dim": 16, "length": 20},
            "student": {"pe_type": "one_hot", "hidden_dim": -1, "pe_embedding_dim": -1},
            "ngrams": {},
        }
    )
    teacher = _StubTeacher([1, 1, 1])  # prefix_length = 3
    configure_positional_encoding(cfg, teacher)
    embedding_dim = 3 + cfg.dataset.length - 1  # prefix + length - 1
    assert cfg.student.pe_embedding_dim == embedding_dim
    assert cfg.student.hidden_dim == cfg.dataset.dim + embedding_dim

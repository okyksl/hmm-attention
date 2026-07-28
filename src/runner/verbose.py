import logging

import torch

from src.spectra import effective_rank
from src.teachers import LinearARTeacher


def log_teacher_summary(teacher: torch.nn.Module) -> None:
    """Log teacher weight stats. No-op for teachers that aren't `LinearARTeacher`."""
    if not isinstance(teacher, LinearARTeacher):
        return
    _log_summary("Teacher", teacher, teacher._params)


def log_student_summary(student: torch.nn.Module) -> None:
    """Log student weight stats if the student happens to be a `LinearARTeacher`."""
    if not isinstance(student, LinearARTeacher):
        return
    _log_summary("Student", student, student._get_weights())


def _log_summary(role: str, model: LinearARTeacher, params: torch.Tensor) -> None:
    logger = logging.getLogger()
    logger.info(f"===== {role} =====")
    logger.info(f"{role} rank: {model.rank}")
    logger.info(f"{role} dim: {model.dim}")
    logger.info(f"{role} window: {model.window}")
    logger.info(f"{role} scale: {model.scale}")
    logger.info(f"{role} weights: {model._get_weights().shape}")

    flat = params.view(-1, params.size(-1))
    logger.info(
        f"Frobenius norm/norm^2: {torch.linalg.norm(flat)}, "
        f"{torch.linalg.norm(flat) ** 2}"
    )
    logger.info(
        f"Operator norm/norm^2: {torch.linalg.norm(flat, ord=2)}, "
        f"{torch.linalg.norm(flat, ord=2) ** 2}"
    )

    # Spectrum structure: the outer law's per-lag weights and the inner law's
    # realized singular values (top-8) plus their participation ratio.
    logger.info(f"{role} lag spectrum: {model.lag_spectrum_spec}")
    logger.info(f"{role} lag weights: {model.lag_weights.tolist()}")
    logger.info(f"{role} feature spectrum: {model.spectrum_specs[0]}")
    for lag_idx, s in enumerate(model.singular_values):
        logger.info(
            f"  lag {lag_idx}: eff. rank {effective_rank(s):.2f}, "
            f"top-8 singular values {[round(v, 4) for v in s[:8].tolist()]}"
        )

from typing import Optional

from src.predictors.base import Predictor
from src.predictors.classification import (
    ClassificationPredictor,
    HierarchicalPredictor,
    MultiLevelHierarchicalPredictor,
)
from src.predictors.regression import RegressionPredictor

__all__ = [
    "Predictor",
    "ClassificationPredictor",
    "HierarchicalPredictor",
    "MultiLevelHierarchicalPredictor",
    "RegressionPredictor",
    "build_predictor",
]


def build_predictor(
    teacher,
    kind: str = "classification",
    argmax: bool = False,
    noise_std: float = 0.0,
) -> Predictor:
    """Wrap a teacher in a Predictor.

    `kind` selects between classification (discrete one-hot) and regression
    (continuous). For classification, `HierarchicalTeacher` gets a
    `HierarchicalPredictor` so burn-in emits valid chunk-composed prefixes;
    every other teacher uses the plain `ClassificationPredictor`.
    """
    # Local import keeps the package importable without materializing the whole
    # teacher hierarchy at module load.
    from src.teachers import HierarchicalTeacher, MultiLevelHierarchicalTeacher

    if kind == "classification":
        # HierarchicalTeacher is an L=1 subclass of MultiLevelHierarchicalTeacher,
        # so check the more specific type first.
        if isinstance(teacher, HierarchicalTeacher):
            return HierarchicalPredictor(teacher, argmax=argmax)
        if isinstance(teacher, MultiLevelHierarchicalTeacher):
            return MultiLevelHierarchicalPredictor(teacher, argmax=argmax)
        return ClassificationPredictor(teacher, argmax=argmax)
    if kind == "regression":
        return RegressionPredictor(teacher, noise_std=noise_std)
    raise ValueError(f"Unknown predictor kind: {kind!r}")

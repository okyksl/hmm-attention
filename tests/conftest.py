"""Shared pytest fixtures for the trainer/model/data test suite.

Fixtures are intentionally tiny so unit tests can inspect outputs directly and
so the end-to-end smoke test runs in well under a second on CPU.
"""

from typing import Tuple

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import ARDataset, ar_batch_collate
from src.predictors import (
    ClassificationPredictor,
    HierarchicalPredictor,
    build_predictor,
)
from src.model import TransformerDecoder
from src.teachers import HierarchicalTeacher, LinearARTeacher


# ---- global determinism ------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_everything():
    """Reset RNG state before every test for repeatable generation."""
    torch.manual_seed(0)


# ---- device ------------------------------------------------------------------


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


# ---- teachers ----------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_teacher() -> LinearARTeacher:
    return LinearARTeacher.from_parameters(
        dim=4,
        window=2,
        span_lengths=[1, 1],
        spectrum={'rank': 4},
        scale=5.0,
    )


@pytest.fixture(scope="session")
def tiny_hier_teacher(tiny_teacher: LinearARTeacher) -> HierarchicalTeacher:
    return HierarchicalTeacher(
        base_teacher=tiny_teacher,
        chunk_dim=4,
        chunk_size=2,
        chunk_seed=0,
    )


# ---- predictors --------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_predictor(tiny_teacher: LinearARTeacher) -> ClassificationPredictor:
    return build_predictor(tiny_teacher, kind="classification")


@pytest.fixture(scope="session")
def tiny_hier_predictor(tiny_hier_teacher: HierarchicalTeacher) -> HierarchicalPredictor:
    return build_predictor(tiny_hier_teacher, kind="classification")


# ---- dataset + loaders -------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_dataset(tiny_predictor: ClassificationPredictor) -> ARDataset:
    # window=2 matches teacher; prefix_length=2 matches teacher.context_length.
    return ARDataset(
        predictor=tiny_predictor,
        window=2,
        dim=4,
        number=8,
        length=6,
        prefix_length=2,
        unroll_sequences=False,
    )


@pytest.fixture()
def tiny_loaders(tiny_dataset: ARDataset) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(tiny_dataset, batch_size=4, collate_fn=ar_batch_collate)
    val_loader = DataLoader(tiny_dataset, batch_size=4, collate_fn=ar_batch_collate)
    return train_loader, val_loader


# ---- student -----------------------------------------------------------------


@pytest.fixture()
def tiny_student() -> TransformerDecoder:
    return TransformerDecoder(
        dim=4,
        hidden_dim=8,
        num_heads=2,
        ff_hidden_dim=8,
        num_blocks=1,
        dropout=0.0,
        pe_type="absolute",
        pe_learnable=True,
        pe_embedding_dim=8,
        pe_max_sequence_length=32,
    )

import numpy as np
import torch

from src.model import TransformerDecoder
from src.trainer.attention_logger import AttentionLogger


class _FakeWriter:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((payload, step))


def test_attention_logger_averages_uneven_batches_once(
    tiny_teacher, tiny_student
):
    writer = _FakeWriter()
    logger = AttentionLogger(writer, tiny_teacher, tiny_student, frequency=1)
    first = torch.stack(
        [torch.full((2, 3, 3), 1.0), torch.full((2, 3, 3), 3.0)], dim=0
    ).unsqueeze(0)
    second = torch.full((1, 1, 2, 3, 3), 5.0)

    logger.log(step=4, split="val", attn_weight_batches=[first, second])

    assert len(writer.calls) == 1
    payload, step = writer.calls[0]
    assert step == 4
    table = payload["attn/L1/weights/val"]
    weights = np.asarray([row[-1] for row in table.data])
    assert np.allclose(weights, 3.0)


def test_attention_logger_emits_tables_and_scalars_but_no_images(
    tiny_teacher, tiny_student
):
    writer = _FakeWriter()
    logger = AttentionLogger(writer, tiny_teacher, tiny_student, frequency=1)
    attention = torch.softmax(torch.randn(1, 3, 2, 4, 4), dim=-1)

    logger.log(step=0, split="train", attn_weight_batches=[attention])

    payload, _ = writer.calls[0]
    assert "attn/L1/weights/train" in payload
    assert "attn/L1/align_cos_sim_head1_span1/train" in payload
    assert "attn/L1/span_mass_head1_span1/train" in payload
    assert not any(
        name.endswith(("/heatmaps/train", "/offset_charts/train"))
        or name in {
            "attn/L1/align_cos_sim/train",
            "attn/L1/align_proj_norm/train",
            "attn/L1/value_cos_sim/train",
            "attn/L1/value_proj_norm/train",
        }
        for name in payload
    )


def test_attention_logger_respects_frequency(tiny_teacher, tiny_student):
    writer = _FakeWriter()
    logger = AttentionLogger(writer, tiny_teacher, tiny_student, frequency=5)
    attention = torch.ones(1, 1, 2, 2, 2)

    logger.log(step=3, split="val", attn_weight_batches=[attention])

    assert writer.calls == []


def test_attention_logger_rejects_incompatible_batch_shapes(
    tiny_teacher, tiny_student
):
    writer = _FakeWriter()
    logger = AttentionLogger(writer, tiny_teacher, tiny_student, frequency=1)
    batches = [torch.ones(1, 2, 2, 3, 3), torch.ones(1, 1, 2, 4, 4)]

    with np.testing.assert_raises_regex(ValueError, "shapes differ"):
        logger.log(step=0, split="val", attn_weight_batches=batches)


def test_attention_logger_preserves_disentangled_value_scalars(tiny_teacher):
    student = TransformerDecoder(
        dim=4,
        hidden_dim=8,
        num_heads=2,
        ff_hidden_dim=8,
        num_blocks=1,
        dropout=0.0,
        pe_type="none",
        attention_disentanglement=True,
    )
    writer = _FakeWriter()
    logger = AttentionLogger(writer, tiny_teacher, student, frequency=1)
    attention = torch.softmax(torch.randn(1, 2, 2, 3, 3), dim=-1)

    logger.log(step=0, split="val", attn_weight_batches=[attention])

    payload, _ = writer.calls[0]
    assert "attn/L1/value_norm_head1/val" in payload
    assert "attn/L1/value_cos_head1_teacher1/val" in payload
    assert "attn/L1/value_inner_head2_teacher2/val" in payload

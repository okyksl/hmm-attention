from typing import List, Optional

import numpy as np
import torch

from src.model import TransformerDecoder
from src.profiling import get_profiler
from src.teachers import LinearARTeacher
from src.visualizer import (
    attention_alignment_scalars,
    attention_span_mass_scalars,
    build_attention_table,
    compute_attention_alignment,
    compute_value_alignment,
    value_alignment_scalars,
)


class AttentionLogger:
    """Log numerical attention snapshots and derived scalars.

    The tables contain batch-averaged attention activations, not learned model
    parameters.  Rendering is deliberately deferred to post-training analysis
    so the requested numerical cadence does not stall the training loop.
    """

    def __init__(
        self,
        writer,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        frequency: int,
    ) -> None:
        self.writer = writer
        self.teacher = teacher
        self.student = student
        self.frequency = frequency

    @staticmethod
    def _average_batches(attn_weight_batches: List[torch.Tensor]) -> np.ndarray:
        """Weighted batch mean as ``(layer, head, query, key)`` on CPU.

        Reduction happens on the source device and only the small mean is
        transferred.  Accumulating sums avoids concatenating validation batches
        and remains correct when the final batch is smaller.
        """
        expected_shape = attn_weight_batches[0].shape
        if len(expected_shape) != 5:
            raise ValueError(
                "attention batches must have shape "
                "(layer, batch, head, query, key)"
            )

        total: Optional[torch.Tensor] = None
        count = 0
        reference = expected_shape[:1] + expected_shape[2:]
        for batch in attn_weight_batches:
            if len(batch.shape) != 5 or batch.shape[:1] + batch.shape[2:] != reference:
                raise ValueError("attention batch shapes differ outside the batch axis")
            batch_sum = batch.detach().sum(dim=1, dtype=torch.float32)
            total = batch_sum if total is None else total + batch_sum
            count += batch.shape[1]

        if total is None or count == 0:
            raise ValueError("cannot average empty attention batches")
        return (total / count).cpu().numpy()

    def log(
        self,
        step: int,
        split: str,
        attn_weight_batches: List[torch.Tensor],
    ) -> None:
        if not isinstance(self.student, TransformerDecoder):
            return
        if self.writer is None or step % self.frequency != 0:
            return
        if not attn_weight_batches:
            return

        with get_profiler().cpu(f"attention_log_{split}"):
            attn_avg = self._average_batches(attn_weight_batches)
            payload = {}

            for layer, layer_attn in enumerate(attn_avg):
                # Keep the historical W&B key.  Here "weights" means attention
                # activations; it does not refer to learned parameters.
                layer_name = f"L{layer + 1}"
                payload[f"attn/{layer_name}/weights/{split}"] = (
                    build_attention_table(layer_attn)
                )

                if not isinstance(self.teacher, LinearARTeacher):
                    continue

                stride: Optional[int] = getattr(self.teacher, "stride", None)
                context_length = getattr(
                    self.teacher,
                    "context_length",
                    sum(self.teacher.span_lengths),
                )
                alignment = compute_attention_alignment(
                    layer_attn,
                    span_lengths=self.teacher.span_lengths,
                    context_length=context_length,
                    stride=stride,
                )
                payload.update(
                    attention_alignment_scalars(alignment, split, layer_name)
                )
                payload.update(
                    attention_span_mass_scalars(
                        layer_attn,
                        span_lengths=self.teacher.span_lengths,
                        context_length=context_length,
                        split=split,
                        layer_name=layer_name,
                        stride=stride,
                    )
                )

                value_alignment = compute_value_alignment(
                    teacher_matrices=self.teacher._params,
                    student=self.student,
                    dim=self.teacher.dim,
                    layer=layer,
                )
                if value_alignment is not None:
                    payload.update(
                        value_alignment_scalars(
                            value_alignment,
                            split=split,
                            layer_name=layer_name,
                        )
                    )

            # One history row / queue operation per split and attention step.
            self.writer.log(payload, step=step)

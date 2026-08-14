from typing import Dict, List, Tuple

import torch

from src.hierarchy_slots import hierarchy_slot_counts, hierarchy_slot_ids
from src.loss import KLDivergenceLoss
from src.teachers import ARTeacher, MultiLevelHierarchicalTeacher


class TeacherEvaluator:
    """Runs the teacher (optionally lag-restricted) and updates its metrics.

    Owns the precomputed `teacher.with_lag_restriction(k)` cache so train/val
    loops don't rebuild them each step. Also handles the "teacher-generated
    data has leading prefix tokens" alignment — a lag-restricted teacher
    sees a shorter context, so we trim data from the front so both produce
    the same number of output positions.
    """

    def __init__(
        self,
        teacher: torch.nn.Module,
        device: torch.device,
        slot_mode: str = "surface",
    ) -> None:
        self.teacher = teacher
        self.is_ar = isinstance(teacher, ARTeacher)
        self.is_hierarchical = isinstance(teacher, MultiLevelHierarchicalTeacher)
        self.slot_mode = slot_mode
        self.slot_counts = (
            hierarchy_slot_counts(teacher, slot_mode)
            if self.is_hierarchical
            else []
        )
        self._teacher_by_k: Dict[int, ARTeacher] = {}
        self.prefix_ks: List[int] = []
        # Lag-restricted variants only exist for teachers with an explicit
        # multi-lag window (LinearARTeacher / HierarchicalTeacher, which override
        # `with_lag_restriction`). Attention / adaptive teachers have no lag
        # structure, so they get only the full-teacher metric.
        supports_lags = (
            self.is_ar
            and type(teacher).with_lag_restriction is not ARTeacher.with_lag_restriction
        )
        if supports_lags:
            # k == window is a no-op (same as self.teacher); skip to avoid
            # a redundant params clone.
            for k in range(1, teacher.window):
                self._teacher_by_k[k] = teacher.with_lag_restriction(k).to(device)
            self.prefix_ks = list(range(1, teacher.window + 1))

    def metric_keys(self) -> List[str]:
        if not self.is_ar:
            return []
        return [
            f"{context}/kl/{split}"
            for context in self.context_names()
            for split in ("train", "val")
        ]

    def context_names(self) -> List[str]:
        if not self.is_ar:
            return []
        return ["teacher"] + [f"teacher_k{k}" for k in self.prefix_ks]

    def loss_metric_keys(self) -> List[str]:
        keys = [
            f"{context}/loss/{split}"
            for context in self.context_names()
            for split in ("train", "val")
        ]
        keys.extend(self._location_metric_keys("loss"))
        return keys

    def acc_metric_keys(self) -> List[str]:
        keys = [
            f"{context}/acc/{split}"
            for context in self.context_names()
            for split in ("train", "val")
        ]
        keys.extend(self._location_metric_keys("acc"))
        return keys

    def location_names(self) -> List[str]:
        """Hierarchical surface locations, ordered top-to-bottom.

        They use the same configured slot layout as the probe logger. The
        terminal surface-token level is included with one slot, so the same
        surface prediction contributes to one slot metric at every level of
        the full generative path.
        """
        if not self.is_hierarchical:
            return []
        return [
            f"level{level}/slot{slot}"
            for level, count in enumerate(self.slot_counts)
            for slot in range(count)
        ]

    def _location_metric_keys(
        self,
        metric: str,
        contexts: List[str] | None = None,
    ) -> List[str]:
        contexts = self.context_names() if contexts is None else contexts
        return [
            f"{context}/{location}/{metric}/{split}"
            for context in contexts
            for location in self.location_names()
            for split in ("train", "val")
        ]

    def student_loss_metric_keys(self) -> List[str]:
        return self._location_metric_keys("loss", contexts=["student"])

    def student_acc_metric_keys(self) -> List[str]:
        return self._location_metric_keys("acc", contexts=["student"])

    @staticmethod
    def context_name(prefix: int) -> str:
        return "teacher" if prefix < 0 else f"teacher_k{prefix}"

    def _resolve(self, prefix: int):
        if prefix > 0 and prefix in self._teacher_by_k:
            return self._teacher_by_k[prefix]
        return self.teacher

    def _align_data(self, data: torch.Tensor, model) -> torch.Tensor:
        # `unroll` drops `burn_in` leading positions (== context_length for
        # bounded teachers, so this matches the old alignment). A lag-restricted
        # teacher has a smaller burn_in and would otherwise emit more positions;
        # trim the front so both produce the same count. Uses burn_in rather
        # than context_length so an adaptive teacher (context_length == ADAPTIVE)
        # aligns correctly (no trim when burn_ins match).
        if self.teacher.burn_in != model.burn_in:
            return data[:, self.teacher.burn_in - model.burn_in :, :]
        return data

    def run(
        self,
        data: torch.Tensor,
        prefix: int = -1,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the (optionally lag-restricted) teacher on `data`.

        Returns `(out, log_probs, targets)`. `log_probs` is always the raw
        teacher log-probs; `out` is `exp(log_probs)` (probs) when
        `normalize=True`, else `log_probs`.
        """
        model = self._resolve(prefix)
        data = self._align_data(data, model)
        log_probs, targets = model.unroll(data, return_targets=True)
        out = log_probs.exp() if normalize else log_probs
        return out, log_probs, targets

    def update_loss_acc_metrics(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        prefix: int,
        split: str,
        metrics: Dict[str, "LossMetric"],
        loss_fn: torch.nn.Module,
    ) -> None:
        """Update aggregate and hierarchical-location teacher metrics.

        Location metrics score the same surface predictions as the aggregate
        metrics, partitioned by the configured slot layout at each hierarchy
        level. Non-hierarchical teachers only update the aggregates.
        """
        context = self.context_name(prefix)
        batch_size = targets.shape[0]
        metrics[f"{context}/loss/{split}"].update(
            loss_fn(log_probs, targets).item(), batch_size
        )
        metrics[f"{context}/acc/{split}"].update(log_probs, targets)

        self.update_location_metrics(
            out=log_probs,
            targets=targets,
            context=context,
            split=split,
            metrics=metrics,
            loss_fn=loss_fn,
        )

    def update_location_metrics(
        self,
        out: torch.Tensor,
        targets: torch.Tensor,
        context: str,
        split: str,
        metrics: Dict[str, "LossMetric"],
        loss_fn: torch.nn.Module,
        position_offset: int = 0,
    ) -> None:
        """Update one model's loss/accuracy on every hierarchy location.

        ``position_offset`` is the absolute surface index predicted by
        ``out[:, 0]``. Teacher unrolls start at an aligned hierarchy boundary;
        student outputs start at the dataset prefix length.
        """

        if not self.is_hierarchical:
            return

        batch_size = targets.shape[0]
        positions = torch.arange(out.shape[1], device=out.device) + position_offset
        for level, count in enumerate(self.slot_counts):
            slots = hierarchy_slot_ids(
                positions, self.teacher, level, self.slot_mode
            )
            for slot in range(count):
                mask = slots == slot
                if not bool(mask.any()):
                    continue
                stub = f"{context}/level{level}/slot{slot}"
                location_out = out[:, mask, :]
                location_targets = targets[:, mask, :]
                metrics[f"{stub}/loss/{split}"].update(
                    loss_fn(location_out, location_targets).item(), batch_size
                )
                metrics[f"{stub}/acc/{split}"].update(
                    location_out, location_targets
                )

    def update_kl_metrics(
        self,
        student_out: torch.Tensor,
        data: torch.Tensor,
        split: str,
        metrics: Dict[str, "LossMetric"],
    ) -> None:
        """KL(teacher || student) at full context and each lag restriction."""
        if not self.is_ar:
            return
        kl = KLDivergenceLoss(reduction="mean")
        probs, _, _ = self.run(data, prefix=-1, normalize=True)
        metrics[f"teacher/kl/{split}"].update(
            kl(student_out, probs).item(), data.size(0)
        )
        for k in self.prefix_ks:
            probs_k, _, _ = self.run(data, prefix=k, normalize=True)
            metrics[f"teacher_k{k}/kl/{split}"].update(
                kl(student_out, probs_k).item(), data.size(0)
            )

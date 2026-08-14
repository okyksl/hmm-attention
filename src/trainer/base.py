import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import tqdm

from src.loss import CrossentropyLoss
from src.profiling import format_report, get_profiler
from src.metrics import (
    AccuracyMetric,
    ConstantAccuracyMetric,
    ConstantLossMetric,
    LossMetric,
    MinMetric,
    RelativeMetric,
)
from src.teachers import ARTeacher
from src.trainer.attention_logger import AttentionLogger
from src.trainer.checkpoint import (
    CHECKPOINT_FILENAME,
    is_stub_payload,
    load_checkpoint,
    restore_into_trainer,
    save_checkpoint,
)
from src.trainer.config import LoggingConfig, NgramConfig, SchedulerConfig
from src.trainer.ngram_eval import NgramEvaluator
from src.trainer.probe_logger import ProbeLogger
from src.trainer.registry import MetricRegistry
from src.trainer.teacher_eval import TeacherEvaluator


class Trainer(ABC):
    """Abstract trainer with shared step / eval / logging plumbing.

    Concrete subclasses only need to define how the student is optimized —
    typically by overriding `_train_loop` (per-step SGD, custom schedules, etc.)
    and `_train_ngram` (auxiliary-model training phase).
    """

    def __init__(
        self,
        steps: int,
        device: torch.device,
        teacher: torch.nn.Module,
        student: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_cfg: SchedulerConfig,
        ngram_cfg: NgramConfig,
        logging_cfg: LoggingConfig,
        max_grad_norm: Optional[float] = None,
        checkpoint_path: Optional[Path] = None,
        resume_from: Optional[Path] = None,
        wandb_run_id: Optional[str] = None,
        config_hash: Optional[str] = None,
        checkpoint_frequency: int = 1,
    ) -> None:
        self.steps = steps
        self.current_step = 0

        self.device = device
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler_cfg = scheduler_cfg
        self.ngram_cfg = ngram_cfg
        self.logging_cfg = logging_cfg
        self.max_grad_norm = max_grad_norm
        self.logger = logging.getLogger()

        # Checkpointing: `checkpoint_path` is where we WRITE (None disables
        # saves); `resume_from` is where we READ on startup (None = fresh run).
        # `wandb_run_id`/`config_hash` are metadata the runner injects; stored
        # in each checkpoint so a future resume can reattach to the same wandb
        # run and reject config drift.
        self.checkpoint_path = checkpoint_path
        self.resume_from = resume_from
        self.wandb_run_id = wandb_run_id
        self.config_hash = config_hash
        # Save every N calls to `_save_checkpoint` (= every N log steps).
        # Trade-off: higher N cuts I/O but on resume you'd re-run the (N-1)
        # log-steps' worth of work that already got logged to wandb — wandb
        # drops non-monotonic step logs, so those re-logs become no-ops and
        # you'd see a discontinuity in the curves. Default 1 keeps the saved
        # step identical to the last wandb-logged step (no gap possible).
        self.checkpoint_frequency = max(1, int(checkpoint_frequency))
        self._save_counter = 0

        # Populated in _init_loop after models are moved to device.
        self.teacher_eval: Optional[TeacherEvaluator] = None
        self.ngram_evals: Dict[str, NgramEvaluator] = {}
        self.attention_logger: Optional[AttentionLogger] = None
        self.probe_logger: Optional[ProbeLogger] = None

    # --- Entry point ---
    def train(self) -> None:
        self._init_loop()
        resumed = self._maybe_resume()
        # On resume, constant teacher metrics ride in the loaded metrics
        # snapshot — re-running the dry loop would double-count them.
        if not resumed and isinstance(self.teacher, ARTeacher):
            self._dry_loop()  # TODO: adapt for TransformerDecoder

        self.logger.info("Beginning training")
        start = time.time()
        while self.current_step < self.steps:
            # Two phases: first ngram models, then the student.
            if self.current_step < self.ngram_cfg.steps:
                self._train_ngram()
            else:
                self._train_loop()
        self.logger.info(
            f"Finished training! Total time: {(time.time() - start) / 3600:.2f}h"
        )
        prof = get_profiler()
        if prof.enabled and self.logging_cfg.writer is None:
            # Terminal dump of the full run when wandb is off. With wandb on,
            # per-window means are already streamed via `profile/*_ms` and
            # `_write_to_wandb` resets the accumulator each log step.
            self.logger.info("Profiler report:\n" + format_report(prof.report()))

    # --- LR scheduler ---
    def _call_lr_sched(self, metric: torch.Tensor) -> None:
        if self.scheduler_cfg.pass_metric:
            self.scheduler_cfg.scheduler.step(metric)
        else:
            self.scheduler_cfg.scheduler.step()

    def _update_lr_sched(self, metric: torch.Tensor, epoch_end: bool) -> None:
        if self.scheduler_cfg.update_on_iter and not epoch_end:
            self._call_lr_sched(metric)
        if not self.scheduler_cfg.update_on_iter and epoch_end:
            self._call_lr_sched(metric)

    # --- Student forward + per-batch metrics ---
    def _run_student(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive student forward, with the leading prefix trimmed off.

        Returns `(out, target, attn_weights)`. The student is not scored on the
        first `prefix_length - 1` output positions (those live inside the
        burn-in region).
        """
        out, attn_weights = self.student(data[:, :-1, :])
        target = data[:, 1:, :]
        ctx = self.train_loader.dataset.prefix_length
        return (
            out[:, ctx - 1 :, :],
            target[:, ctx - 1 :, :],
            attn_weights[:, :, :, ctx - 1 :, ctx - 1 :],
        )

    def _forward_and_metrics(
        self,
        data: torch.Tensor,
        split: str,
        run_teacher_metrics: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Student forward plus loss/accuracy and teacher/ngram KL metrics.

        Returns `(out, target, loss, attn_weights)`. Shared by train and val —
        the training loop only adds `backward` + `optimizer.step` around this.

        `run_teacher_metrics=False` skips the teacher and ngram KL computations.
        Student loss/accuracy
        are always updated because they are cheap and drive the LR scheduler.
        """
        prof = get_profiler()
        with prof.cuda(f"run_student_{split}"):
            out, target, attn_weights = self._run_student(data)
            loss = self.loss_fn(out, target)

        self.metrics[f"student/loss/{split}"].update(loss.item(), data.size(0))
        self.metrics[f"student/acc/{split}"].update(out, target)
        self.teacher_eval.update_location_metrics(
            out=out,
            targets=target,
            context="student",
            split=split,
            metrics=self.metrics,
            loss_fn=self.loss_fn,
            position_offset=self.train_loader.dataset.prefix_length,
        )

        if run_teacher_metrics and isinstance(self.teacher, ARTeacher):
            with prof.cuda(f"teacher_kl_{split}"):
                self.teacher_eval.update_kl_metrics(out, data, split, self.metrics)
            with prof.cuda(f"ngram_kl_{split}"):
                for ne in self.ngram_evals.values():
                    ne.update_kl(out, data, split, self.metrics)

        return out, target, loss, attn_weights

    # --- Init + dry ---
    def _init_loop(self) -> None:
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        self.teacher_eval = TeacherEvaluator(
            self.teacher,
            self.device,
            slot_mode=self.logging_cfg.hierarchy_slot_mode,
        )
        # When the ngram training phase is disabled, skip ngram construction
        # entirely — otherwise ngram KL runs on every log step even though the
        # ngram models are never trained or trainable.
        if self.ngram_cfg.steps > 0:
            for m in self.ngram_cfg.models.values():
                m.to(self.device)
            self.ngram_evals = {
                name: NgramEvaluator(
                    name=name,
                    model=model,
                    optimizer=self.ngram_cfg.optimizers[name],
                    teacher=self.teacher,
                    teacher_evaluator=self.teacher_eval,
                )
                for name, model in self.ngram_cfg.models.items()
            }
        else:
            self.ngram_evals = {}
        self.attention_logger = AttentionLogger(
            writer=self.logging_cfg.writer,
            teacher=self.teacher,
            student=self.student,
            frequency=self.logging_cfg.attention_frequency,
        )
        self.probe_logger = ProbeLogger(
            writer=self.logging_cfg.writer,
            teacher=self.teacher,
            student=self.student,
            cfg=self.logging_cfg,
        )

        self.history: Dict[str, List[float]] = {}
        self.metrics = MetricRegistry()
        for key, metric in (
            ("student/loss/train", LossMetric()),
            ("student/acc/train", AccuracyMetric(k=1)),
            ("student/loss/val", LossMetric()),
            ("student/acc/val", AccuracyMetric(k=1)),
            ("student/loss/val_best", MinMetric()),
            ("student/grad_norm", LossMetric()),
        ):
            self.history[key] = []
            self.metrics.register(key, metric)

        self._register_metrics(
            self.teacher_eval.student_loss_metric_keys(), LossMetric
        )
        self._register_metrics(
            self.teacher_eval.student_acc_metric_keys(), AccuracyMetric, k=1
        )

        if isinstance(self.teacher, ARTeacher):
            self._register_metrics(self.teacher_eval.metric_keys(), LossMetric)
            self._register_metrics(
                self.teacher_eval.loss_metric_keys(), ConstantLossMetric
            )
            self._register_metrics(
                self.teacher_eval.acc_metric_keys(), ConstantAccuracyMetric, k=1
            )
            for ne in self.ngram_evals.values():
                self._register_metrics(ne.kl_metric_keys(), LossMetric)
                self._register_metrics(ne.loss_metric_keys(), LossMetric)
                self._register_metrics(ne.acc_metric_keys(), AccuracyMetric, k=1)

        self.lr_metric = (
            self.metrics["student/loss/train"]
            if self.scheduler_cfg.update_on_iter
            else self.metrics["student/loss/val"]
        )

    def _register_metrics(
        self, metrics: List[str], metric_class, **constructor_kwargs
    ) -> None:
        for key in metrics:
            self.history[key] = []
            self.metrics.register(key, metric_class(**constructor_kwargs))

    # --- Checkpointing ---
    def _maybe_resume(self) -> bool:
        """Load `self.resume_from` into this trainer. Returns True on resume.

        Stub payloads (wandb-id-only, written by the runner before training
        starts) return False — the wandb id is already picked up in the
        runner; the trainer needs a full state snapshot to actually resume.
        """
        if self.resume_from is None or not self.resume_from.exists():
            return False
        payload = load_checkpoint(self.resume_from, self.device)
        if is_stub_payload(payload):
            return False
        restore_into_trainer(self, payload)
        return True

    def _save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return
        # Skip based on modulo, count both taken + skipped calls so the cadence
        # is stable regardless of when training started.
        should_save = (self._save_counter % self.checkpoint_frequency) == 0
        self._save_counter += 1
        if not should_save:
            return
        prof = get_profiler()
        with prof.cuda("checkpoint_save"):
            try:
                save_checkpoint(
                    self,
                    path=self.checkpoint_path,
                    wandb_run_id=self.wandb_run_id,
                    cfg_hash=self.config_hash,
                )
            except OSError as e:
                # Disk full / permission denied — keep training rather than
                # killing a long run over a save failure.
                self.logger.warning(f"Checkpoint save failed: {e}")

    def _dry_loop(self) -> None:
        """One-shot pass over train + val to populate the constant teacher metrics."""
        self.teacher.eval()
        for split, loader in (("train", self.train_loader), ("val", self.val_loader)):
            pbar = tqdm.tqdm(total=len(loader), leave=False)
            pbar.set_description(f"Dry run | {split.capitalize()}")
            for data in loader:
                data = data.to(self.device)
                for prefix in [-1] + self.teacher_eval.prefix_ks:
                    _, log_probs, target = self.teacher_eval.run(
                        data, prefix=prefix, normalize=False
                    )
                    # CE and KL both apply log_softmax internally, which is a
                    # near-identity on log-probs (see src/loss.py:55).
                    self.teacher_eval.update_loss_acc_metrics(
                        log_probs=log_probs,
                        targets=target,
                        prefix=prefix,
                        split=split,
                        metrics=self.metrics,
                        loss_fn=self.loss_fn,
                    )
                pbar.update()
            pbar.close()

    # --- Loops that subclasses may override ---
    @abstractmethod
    def _train_loop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train_ngram(self) -> None:
        raise NotImplementedError

    def _val_loop(self, step: int) -> None:
        self.student.eval()
        prof = get_profiler()
        attn_batches: List[torch.Tensor] = []
        for data in self.val_loader:
            with torch.no_grad():
                with prof.cuda("data_to_device_val"):
                    data = data.to(self.device)
                self.probe_logger.before_forward("val")
                try:
                    _, _, _, attn_weights = self._forward_and_metrics(
                        data, split="val"
                    )
                finally:
                    self.probe_logger.after_forward(data)
                self.probe_logger.collect_val_batch()
                attn_batches.append(attn_weights)
        self.attention_logger.log(step, "val", attn_batches)
        # Measure end-to-end wall time, including probe computation, device
        # synchronization from scalar extraction, and W&B enqueueing.
        with prof.cpu("probe_log"):
            self.probe_logger.log(step, "val")

    def _val_ngram(self) -> None:
        for ne in self.ngram_evals.values():
            ne.model.eval()
        self.teacher.eval()
        for data in self.val_loader:
            with torch.no_grad():
                data = data.to(self.device)
                for ne in self.ngram_evals.values():
                    logits, target, loss = ne.eval_step(
                        data, self.loss_fn, self.ngram_cfg.use_teacher_target
                    )
                    self.metrics[f"ngram_{ne.name}/loss/val"].update(
                        loss.item(), data.size(0)
                    )
                    self.metrics[f"ngram_{ne.name}/acc/val"].update(logits, target)

    # --- End-of-step book-keeping ---
    def _end_step(
        self, step: int, step_time: Optional[float], ngram: bool = False
    ) -> None:
        self.logger.info(self._step_str(step, step_time, ngram))

        if self.val_loader is not None:
            loss_avg = self.metrics["student/loss/val"].compute()
            if loss_avg < self.metrics["student/loss/val_best"].compute():
                self.metrics["student/loss/val_best"].update(loss_avg)

        if ngram:
            ngram_metrics = {
                key: self.metrics[key]
                for ne in self.ngram_evals.values()
                for key in ne.loss_metric_keys() + ne.acc_metric_keys()
            }
        else:
            ngram_metrics = None

        if self.logging_cfg.writer is not None:
            self._write_to_wandb(step, ngram_metrics)

        # Persist BEFORE resetting metrics: the saved snapshot then matches
        # the last-logged wandb step exactly, so a resumed run's next log at
        # `step + log_frequency` satisfies wandb's monotonic-step rule.
        self._save_checkpoint()

        metrics = ngram_metrics or self.metrics
        for metric in metrics.values():
            if isinstance(
                metric,
                (RelativeMetric, MinMetric, ConstantLossMetric, ConstantAccuracyMetric),
            ):
                continue
            metric.reset()

    def _step_str(
        self, step: int, step_time: Optional[float], ngram: bool = False
    ) -> str:
        s = f"Step {step} "
        if step_time:
            s += f"| Step time: {step_time:.1f}s"
        if ngram:
            for ne in self.ngram_evals.values():
                s += (
                    f"| {ne.name} Train loss: {self.metrics[f'ngram_{ne.name}/loss/train'].compute():.3f} "
                    f"| {ne.name} Train acc: {self.metrics[f'ngram_{ne.name}/acc/train'].compute():.3f} "
                )
        else:
            s += f"| Train loss: {self.metrics['student/loss/train'].compute():.3f} "
            if isinstance(self.loss_fn, CrossentropyLoss):
                s += f"| Train acc: {self.metrics['student/acc/train'].compute():.3f} "
                if isinstance(self.teacher, ARTeacher):
                    s += f"| Train teacher acc: {self.metrics['teacher/acc/train'].compute():.3f} "
        return s

    def _write_to_wandb(self, step: int, ngram_metrics=None) -> None:
        metrics = ngram_metrics or self.metrics
        log_metrics = {name: metric.compute() for name, metric in metrics.items()}
        if self.scheduler_cfg.scheduler is None:
            raise ValueError("Scheduler is None, cannot log learning rate.")
        log_metrics["system/learning_rate"] = (
            self.scheduler_cfg.scheduler.get_last_lr()[0]
        )
        # Profiler snapshot — per-window stats since the last log, then reset.
        # `_calls` matters because mean alone hides "cheap-but-frequent" vs
        # "expensive-but-rare"; `_total_ms` is the direct contribution to wall
        # time (mean × calls).
        prof = get_profiler()
        if prof.enabled:
            for section, stats in prof.report(reset=True).items():
                log_metrics[f"profile/{section}/ms"] = stats["mean_ms"]
                log_metrics[f"profile/{section}/calls"] = stats["count"]
                log_metrics[f"profile/{section}/total_ms"] = stats["total_ms"]
        self.logging_cfg.writer.log(log_metrics, step=step)

    def _write_to_json(self, path: str, epoch: int) -> None:
        for key, metric in self.metrics.items():
            self.history[key].append(metric.compute())
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)

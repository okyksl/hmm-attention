"""Utility helpers for logging transformer self-attention during training.

Provides two independent representations:
* `log_attention_table` – structured numeric weights in a wandb.Table for analysis
* `log_attention_heatmap` – static per-head heatmaps logged as wandb.Images
* `compute_gt_attention_row` – ground-truth uniform attention distribution per head
* `log_value_matrix_alignment` – per-head cosine similarity of value weights vs. teacher
* `log_value_alignment_scalars` – per-(head, teacher) value alignment as wandb scalars
    for time-series plotting of cooperative offset dynamics (Section 3.3)
* `log_attention_alignment` – per-head attention alignment scalars and bar charts
* `log_attention_span_mass` – per-(head, span) attention mass as wandb scalars
    for time-series plotting of collaborative head specialization phases

Use `log_attention` as a wrapper when you want either or both attention functions.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import wandb


def _get_attention(
    attn_weights: torch.Tensor, layer: int = 0, batch_idx: int = -1
) -> np.ndarray:
    """Extract attention weights of shape (heads, seq_len, seq_len).

    Args:
        attn_weights: Full attention tensor from model
        layer: Which layer to extract (default: 0)
        batch_idx: Which batch item (-1 for batch average)

    Returns:
        Attention array of shape (heads, seq_len, seq_len)
    """
    try:
        attn = attn_weights[layer].detach().cpu().numpy()
    except IndexError as exc:
        raise IndexError(
            f"Layer index {layer} out of range for attn_weights with "
            f"{len(attn_weights)} layers"
        ) from exc

    if batch_idx == -1:  # average over batch
        return attn.mean(axis=0)

    if batch_idx >= attn.shape[0]:
        raise IndexError(
            f"batch_idx {batch_idx} out of range for batch size {attn.shape[0]}"
        )
    return attn[batch_idx]


def _default_token_seq(seq_len: int) -> List[str]:
    """Generate default token labels as 0-based indices."""
    return [str(i) for i in range(seq_len)]


def build_attention_table(
    attn: np.ndarray, token_seq: Optional[List[str]] = None
) -> wandb.Table:
    """Convert (heads, seq_len, seq_len) attention into structured wandb.Table.

    Args:
        attn: Attention weights of shape (heads, seq_len, seq_len)
        token_seq: Token labels (defaults to indices if None)

    Returns:
        wandb.Table with columns: head, query_idx, key_idx, query_token, key_token, weight
    """
    num_heads, seq_len, _ = attn.shape
    if token_seq is None:
        token_seq = _default_token_seq(seq_len)

    cols = ["head", "query_idx", "key_idx", "query_token", "key_token", "weight"]
    rows: List[List] = []

    for h in range(num_heads):
        for q in range(seq_len):
            for k in range(seq_len):
                rows.append([
                    h, q, k, token_seq[q], token_seq[k], float(attn[h, q, k])
                ])

    return wandb.Table(data=rows, columns=cols)


def log_attention_table(
    run: Optional["wandb.run"],
    attn_weights: torch.Tensor,
    token_seq: Optional[List[str]] = None,
    layer: int = 0,
    batch_idx: int = -1,
    step: Optional[int] = None,
    table_key: str = "attention_table",
) -> None:
    """Log structured attention weights as a wandb.Table.

    Args:
        run: Active wandb.run (skipped if None)
        attn_weights: Full attention tensor from model
        token_seq: Human-readable tokens (defaults to indices)
        layer: Which layer to visualize
        batch_idx: Which batch item (-1 for average)
        step: Training step for versioning
        table_key: Dashboard key for table versions
    """
    if run is None or step is None:
        return

    attn = _get_attention(attn_weights, layer, batch_idx)
    table = build_attention_table(attn, token_seq)

    # Log table with versioning at this step
    run.log({table_key: table}, step=step)


def log_attention_heatmap(
    run: Optional["wandb.run"],
    attn_weights: np.ndarray,
    log_key: str,
    token_seq: Optional[List[str]] = None,
    layer: int = 0,
    batch_idx: int = -1,
    step: Optional[int] = None,
) -> None:
    """Log per-head heatmaps plus an averaged heatmap.

    Args:
        run: wandb run instance (skipped if None)
        attn_weights: attention weights of shape (heads, seq_len, seq_len)
        log_key: key for logging
        token_seq: token labels (defaults to indices)
        layer: layer index (for consistency with table function)
        batch_idx: batch index (for consistency with table function)
        step: training step
    """
    if run is None or step is None:
        return

    # Handle both numpy arrays (current usage) and torch tensors (for consistency)
    if isinstance(attn_weights, torch.Tensor):
        attn = _get_attention(attn_weights, layer, batch_idx)
    else:
        attn = attn_weights  # Already processed numpy array

    num_heads, seq_len, _ = attn.shape
    if token_seq is None:
        token_seq = _default_token_seq(seq_len)

    images: List[wandb.Image] = []

    # Average heatmap first (index 0).
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(
        attn.mean(axis=0),
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        xticklabels=token_seq,
        yticklabels=token_seq,
        cbar=True,
    )
    plt.title("Average Heads")
    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.xticks(rotation=45)
    plt.tight_layout()
    images.append(wandb.Image(fig, caption="Average"))
    plt.close(fig)

    # Per-head heatmaps (1-indexed).
    for h in range(num_heads):
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(
            attn[h],
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            xticklabels=token_seq,
            yticklabels=token_seq,
            cbar=True,
        )
        plt.title(f"Head {h + 1}")
        plt.xlabel("Position")
        plt.ylabel("Position")
        plt.xticks(rotation=45)
        plt.tight_layout()
        images.append(wandb.Image(fig, caption=f"Head {h + 1}"))
        plt.close(fig)

    run.log({log_key: images}, step=step)


def _span_column_ranges(
    span_lengths: List[int],
    context_length: int,
    seq_len: int,
    stride: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Absolute `(start, end)` column ranges of each span in the trimmed attention axis.

    The context window occupies the last `context_length` columns. Within it,
    span `k` starts at `k * stride` (with stride) or `sum(span_lengths[:k])`
    (without). Ranges are clipped to `[0, seq_len)` so callers can slice
    directly. Returned as a list of half-open `(start, end)` pairs — one per
    span.
    """
    context_start = seq_len - context_length
    ranges: List[Tuple[int, int]] = []
    for k, span_len in enumerate(span_lengths):
        if stride is not None:
            start_in_context = k * stride
        else:
            start_in_context = sum(span_lengths[:k])
        abs_start = max(0, min(context_start + start_in_context, seq_len))
        abs_end = max(0, min(abs_start + span_len, seq_len))
        ranges.append((abs_start, abs_end))
    return ranges


def compute_gt_attention_row(
    span_lengths: List[int],
    context_length: int,
    seq_len: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """Ground-truth last-query attention row: uniform over each span's positions.

    For each head `h`, produces a distribution supported uniformly on the
    columns that belong to span `h` inside the trimmed attention axis. Row
    sums to 1.0 for non-degenerate spans (0 otherwise).

    Returns shape `(num_heads, seq_len)`.
    """
    ranges = _span_column_ranges(span_lengths, context_length, seq_len, stride)
    gt = np.zeros((len(span_lengths), seq_len), dtype=np.float32)
    for h, (start, end) in enumerate(ranges):
        if end > start:
            gt[h, start:end] = 1.0 / (end - start)
    return gt


def _heatmap_image(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    xlabel: str = "",
    ylabel: str = "Student head",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> wandb.Image:
    """Render a 2-D numpy array as a seaborn heatmap and return a wandb.Image."""
    fig, ax = plt.subplots(figsize=(max(4, len(col_labels)), max(3, len(row_labels))))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def log_value_matrix_alignment(
    run: Optional["wandb.run"],
    teacher_matrices: torch.Tensor,
    student: torch.nn.Module,
    dim: int,
    step: int,
    split: str,
    layer: int = 0,
    layer_name: Optional[str] = None,
) -> None:
    """Log all-pairs alignment between student value projections and teacher matrices.

    For every (student head h, teacher matrix k) pair this computes:
      - cos_sim(h, k): Frobenius cosine similarity between the effective (dim, dim)
        sub-block of value_proj[h].weight and teacher_matrices[k].
      - proj_norm(h, k): norm of the student matrix projected onto the teacher
        direction = ||W_h||_F * cos_sim(h, k).  This captures how much of the
        student's capacity is aligned with each teacher matrix.
      - student_norm(h): ||W_h||_F, the overall scale of the student value matrix.

    Logs:
      - Scalars {split}_value_student_norm_head{h+1} for each head.
      - Heatmap image {split}_value_cosine_sim: (num_heads × num_teacher), range [-1, 1].
      - Heatmap image {split}_value_proj_norm: (num_heads × num_teacher).

    Only operates when attention_disentanglement=True (each head has its own nn.Linear
    value projection).

    Args:
        run: Active wandb run (skipped if None)
        teacher_matrices: shape (window, dim, dim) — teacher._params
        student: TransformerDecoder instance
        dim: data / vocabulary dimensionality (not embed_dim)
        step: training step
        split: "train" or "val"
        layer: which transformer block to inspect (default 0)
    """
    if run is None:
        return

    block = student.transformer_blocks[layer]
    mha = block.self_attention

    if not getattr(mha, "attention_disentanglement", False):
        return
    if not isinstance(mha.value_proj, nn.ModuleList):
        return

    teacher_np = teacher_matrices.detach().cpu().numpy()  # (num_teacher, dim, dim)
    num_teacher = teacher_np.shape[0]
    num_heads = len(mha.value_proj)

    # Pre-compute flattened teacher vectors and their norms.
    teacher_flat = [teacher_np[k].ravel() for k in range(num_teacher)]
    teacher_norms = [float(np.linalg.norm(a)) for a in teacher_flat]

    # Collect student matrices and their norms.
    student_flat: List[Optional[np.ndarray]] = []
    student_norms: List[float] = []
    for h in range(num_heads):
        vp = mha.value_proj[h]
        if not isinstance(vp, nn.Linear):
            student_flat.append(None)
            student_norms.append(0.0)
            continue
        # Effective (dim, dim) block: identity decoder keeps first `dim` rows;
        # token embedding occupies first `dim` columns.
        W = vp.weight.detach().cpu().numpy()[:dim, :dim].ravel()
        student_flat.append(W)
        student_norms.append(float(np.linalg.norm(W)))

    # Build all-pairs matrices.
    cos_sim_mat = np.zeros((num_heads, num_teacher), dtype=np.float32)
    proj_norm_mat = np.zeros((num_heads, num_teacher), dtype=np.float32)

    for h in range(num_heads):
        w = student_flat[h]
        w_norm = student_norms[h]
        if w is None or w_norm == 0:
            continue
        for k in range(num_teacher):
            a = teacher_flat[k]
            a_norm = teacher_norms[k]
            if a_norm == 0:
                continue
            cos = float(np.dot(w, a) / (w_norm * a_norm))
            cos_sim_mat[h, k] = cos
            proj_norm_mat[h, k] = w_norm * cos  # signed projection norm

    row_labels = [f"Head {h + 1}" for h in range(num_heads)]
    col_labels = [f"Teacher {k + 1}" for k in range(num_teacher)]

    log_dict: dict = {}
    scope = layer_name or f"L{layer + 1}"

    # Per-head student norms as scalars.
    for h, norm in enumerate(student_norms):
        log_dict[f"attn/{scope}/value_norm_head{h + 1}/{split}"] = norm

    # Heatmap: cosine similarity (all pairs).
    log_dict[f"attn/{scope}/value_cos_sim/{split}"] = _heatmap_image(
        cos_sim_mat,
        row_labels,
        col_labels,
        title=f"Value matrix cosine similarity ({split}, step {step})",
        xlabel="Teacher matrix",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    # Heatmap: projected norm (all pairs).
    abs_max = float(np.abs(proj_norm_mat).max()) or 1.0
    log_dict[f"attn/{scope}/value_proj_norm/{split}"] = _heatmap_image(
        proj_norm_mat,
        row_labels,
        col_labels,
        title=f"Value matrix projected norm ({split}, step {step})",
        xlabel="Teacher matrix",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
    )

    run.log(log_dict, step=step)


def log_value_alignment_scalars(
    run: Optional["wandb.run"],
    teacher_matrices: torch.Tensor,
    student: torch.nn.Module,
    dim: int,
    step: int,
    split: str,
    layer: int = 0,
    layer_name: Optional[str] = None,
) -> None:
    """Log per-(head, teacher) value matrix alignment as wandb scalars.

    Goal: Track the cooperative offset dynamics described in Section 3.3.
    When one head starts aligning its value matrix with a teacher feature,
    other heads may temporarily develop *negative* alignment along that same
    direction to cancel cross-terms — a cooperative correction mechanism.

    For every (student head h, teacher matrix k) pair, logs:
      - ``{split}_value_cosine_head{h+1}_teacher{k+1}``:
            Frobenius cosine similarity (direction only, range [-1, 1]).
      - ``{split}_value_inner_head{h+1}_teacher{k+1}``:
            Raw Frobenius inner product <V_h, A*_k>.  Scales with dim²;
            use the cosine variant for normalized comparison.

    wandb visualization: group by *teacher* (feature) to get one panel per
    feature direction with lines per head — matching the paper's figure.
      - Feature (j) panel: ``{split}_value_inner_head*_teacher{j+1}``

    Only operates when attention_disentanglement=True.

    Args:
        run: Active wandb run (skipped if None)
        teacher_matrices: shape (window, dim, dim) — teacher._params (A*_k)
        student: TransformerDecoder instance
        dim: data / vocabulary dimensionality (not embed_dim)
        step: training step
        split: "train" or "val"
        layer: which transformer block to inspect (default 0)
    """
    if run is None:
        return

    block = student.transformer_blocks[layer]
    mha = block.self_attention

    if not getattr(mha, "attention_disentanglement", False):
        return
    if not isinstance(mha.value_proj, nn.ModuleList):
        return

    teacher_np = teacher_matrices.detach().cpu().numpy()  # (num_teacher, dim, dim)
    num_teacher = teacher_np.shape[0]
    num_heads = len(mha.value_proj)

    # Flatten each teacher matrix A*_k to a vector for dot-product computation.
    teacher_flat = [teacher_np[k].ravel() for k in range(num_teacher)]
    teacher_norms = [float(np.linalg.norm(a)) for a in teacher_flat]

    # Extract the effective (dim, dim) sub-block of each head's value projection.
    student_flat: List[Optional[np.ndarray]] = []
    student_norms: List[float] = []
    for h in range(num_heads):
        vp = mha.value_proj[h]
        if not isinstance(vp, nn.Linear):
            student_flat.append(None)
            student_norms.append(0.0)
            continue
        W = vp.weight.detach().cpu().numpy()[:dim, :dim].ravel()
        student_flat.append(W)
        student_norms.append(float(np.linalg.norm(W)))

    # Compute all-pairs alignment and log as individual scalars.
    log_dict: dict = {}
    scope = layer_name or f"L{layer + 1}"
    for h in range(num_heads):
        w = student_flat[h]
        w_norm = student_norms[h]
        if w is None or w_norm == 0:
            continue
        for k in range(num_teacher):
            a = teacher_flat[k]
            a_norm = teacher_norms[k]
            # Raw Frobenius inner product: <V_h, A*_k> = sum_ij V_h[i,j] * A*_k[i,j]
            inner = float(np.dot(w, a))
            # Cosine similarity: normalized to [-1, 1]
            cos = inner / (w_norm * a_norm) if a_norm > 0 else 0.0
            log_dict[f"attn/{scope}/value_cos_head{h + 1}_teacher{k + 1}/{split}"] = cos
            log_dict[f"attn/{scope}/value_inner_head{h + 1}_teacher{k + 1}/{split}"] = inner

    run.log(log_dict, step=step)


def log_attention_alignment(
    run: Optional["wandb.run"],
    attn_avg: np.ndarray,
    span_lengths: List[int],
    context_length: int,
    step: int,
    split: str,
    stride: Optional[int] = None,
    layer_name: str = "L1",
) -> None:
    """Log all-pairs alignment between student attention rows and GT span distributions.

    For every (student head h, GT span k) pair this computes:
      - cos_sim(h, k): cosine similarity between the last-query attention row of head h
        and the ground-truth uniform distribution over span k.
      - proj_norm(h, k): norm of the student attention row projected onto the GT span
        direction = ||attn_h||_2 * cos_sim(h, k).

    Logs:
      - Scalars {split}_attn_student_norm_head{h+1} for each head.
      - Heatmap image {split}_attn_cosine_sim: (num_heads × num_spans).
      - Heatmap image {split}_attn_proj_norm: (num_heads × num_spans).
      - Per-head bar charts {split}_attn_offset_charts comparing the last-query
        attention row of each head against every GT span (zoomed to context window).

    Args:
        run: Active wandb run (skipped if None)
        attn_avg: batch-averaged attention, shape (num_heads, seq_len, seq_len)
        span_lengths: per-span lengths (length == num GT spans)
        context_length: total context window length
        step: training step
        split: "train" or "val"
        stride: stride between spans (None = non-overlapping)
    """
    if run is None:
        return

    num_heads, seq_len, _ = attn_avg.shape
    num_spans = len(span_lengths)

    # Ground-truth rows: shape (num_spans, seq_len).
    gt = compute_gt_attention_row(span_lengths, context_length, seq_len, stride=stride)

    # Last-query attention row per head: shape (num_heads, seq_len).
    last_rows = attn_avg[:, -1, :]
    student_norms = [float(np.linalg.norm(last_rows[h])) for h in range(num_heads)]

    gt_flat = [gt[k] for k in range(num_spans)]
    gt_norms = [float(np.linalg.norm(g)) for g in gt_flat]

    # All-pairs cosine similarity and projected norm.
    cos_sim_mat = np.zeros((num_heads, num_spans), dtype=np.float32)
    proj_norm_mat = np.zeros((num_heads, num_spans), dtype=np.float32)

    for h in range(num_heads):
        pred = last_rows[h]
        p_norm = student_norms[h]
        if p_norm == 0:
            continue
        for k in range(num_spans):
            g_norm = gt_norms[k]
            if g_norm == 0:
                continue
            cos = float(np.dot(pred, gt_flat[k]) / (p_norm * g_norm))
            cos_sim_mat[h, k] = cos
            proj_norm_mat[h, k] = p_norm * cos  # signed projection norm

    row_labels = [f"Head {h + 1}" for h in range(num_heads)]
    col_labels = [f"Span {k + 1}" for k in range(num_spans)]

    log_dict: dict = {}

    # Per-head student norms as scalars.
    for h, norm in enumerate(student_norms):
        log_dict[f"attn/{layer_name}/align_norm_head{h + 1}/{split}"] = norm

    # All-pairs cosine similarity as *scalars* too, not just the heatmap image.
    # This is the scale-free counterpart of `span_mass`, and the series that
    # `src/analysis/head_phases.py` reads to infer which head owns which span.
    for h in range(num_heads):
        for k in range(num_spans):
            log_dict[
                f"attn/{layer_name}/align_cos_sim_head{h + 1}_span{k + 1}/{split}"
            ] = float(cos_sim_mat[h, k])

    # Heatmap: cosine similarity (all pairs).
    log_dict[f"attn/{layer_name}/align_cos_sim/{split}"] = _heatmap_image(
        cos_sim_mat,
        row_labels,
        col_labels,
        title=f"Attention cosine similarity ({split}, step {step})",
        xlabel="GT span",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
    )

    # Heatmap: projected norm (all pairs).
    abs_max = float(np.abs(proj_norm_mat).max()) or 1.0
    log_dict[f"attn/{layer_name}/align_proj_norm/{split}"] = _heatmap_image(
        proj_norm_mat,
        row_labels,
        col_labels,
        title=f"Attention projected norm ({split}, step {step})",
        xlabel="GT span",
        cmap="RdBu_r",
        vmin=-abs_max,
        vmax=abs_max,
    )

    # Per-head bar charts: head h's last-query row vs every GT span.
    context_start = max(0, seq_len - context_length)
    positions = np.arange(context_start, seq_len)
    bar_images: List[wandb.Image] = []

    for h in range(num_heads):
        pred_ctx = last_rows[h, context_start:]
        n_cols = num_spans + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), sharey=True)
        if n_cols == 1:
            axes = [axes]

        # Student attention.
        axes[0].bar(positions, pred_ctx, color="steelblue")
        axes[0].set_title(f"Head {h + 1}\n(student)", fontsize=9)
        axes[0].set_xlabel("Key pos")
        axes[0].set_ylabel("Weight")

        # One subplot per GT span.
        for k in range(num_spans):
            gt_ctx = gt[k, context_start:]
            axes[k + 1].bar(positions, gt_ctx, color="coral", alpha=0.8)
            axes[k + 1].set_title(
                f"GT span {k + 1}\ncos={cos_sim_mat[h, k]:.2f}", fontsize=9
            )
            axes[k + 1].set_xlabel("Key pos")

        fig.suptitle(f"Head {h + 1} — last-query attention ({split}, step {step})", fontsize=10)
        plt.tight_layout()
        bar_images.append(wandb.Image(fig, caption=f"Head {h + 1}"))
        plt.close(fig)

    log_dict[f"attn/{layer_name}/offset_charts/{split}"] = bar_images
    run.log(log_dict, step=step)


def log_attention_span_mass(
    run: Optional["wandb.run"],
    attn_avg: np.ndarray,
    span_lengths: List[int],
    context_length: int,
    step: int,
    split: str,
    stride: Optional[int] = None,
    layer_name: str = "L1",
) -> None:
    """Log the total attention mass each head places on each span's positions.

    Goal: Track the collaborative phases of head specialization.  Early in
    training all heads converge on the most statistically important position
    group; then heads sequentially break away to cover the remaining groups.
    Plotting these scalars over training steps (with log-scale axes) reveals
    the stage transitions clearly.

    For each (head h, span k) pair, computes:
        sum of attn_avg[h, last_query, positions_in_span_k]
    i.e. the total attention the last token places on span k's positions,
    averaged across all sequences in the batch.

    Logged as wandb scalars ``{split}_attn_mass_head{h+1}_span{k+1}``.

    wandb visualization: group by *span* to get one panel per position group
    with lines per head — matching the paper's attention position weight figure.
      - Position (j) panel: ``{split}_attn_mass_head*_span{j}``
    Use log scale on both axes to see the stage separation.

    Args:
        run: Active wandb run (skipped if None)
        attn_avg: batch-averaged attention, shape (num_heads, seq_len, seq_len)
        span_lengths: per-span lengths (length == num GT spans)
        context_length: total context window length
        step: training step
        split: "train" or "val"
        stride: stride between spans (None = non-overlapping)
    """
    if run is None:
        return

    num_heads, seq_len, _ = attn_avg.shape
    last_rows = attn_avg[:, -1, :]  # (num_heads, seq_len)
    ranges = _span_column_ranges(span_lengths, context_length, seq_len, stride)

    log_dict: dict = {}
    for k, (start, end) in enumerate(ranges):
        for h in range(num_heads):
            mass = float(last_rows[h, start:end].sum())
            log_dict[f"attn/{layer_name}/span_mass_head{h + 1}_span{k + 1}/{split}"] = mass

    run.log(log_dict, step=step)

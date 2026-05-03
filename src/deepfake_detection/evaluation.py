"""Evaluation harness.

Scores the trained CNN and the classical-forensics pipeline against the same
CIFAKE test split so they can be compared head-to-head.

The forensics pipeline is run in ``forensics_only`` mode here. The pretrained
HuggingFace classifier expects normal-resolution photos and was trained on a
different distribution (mostly faces, not generic CIFAR-style images), so
running it on 32×32 CIFAKE samples isn't a fair test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .cifake import CIFAKEDataset, make_loaders
from .model import SmallResNet, best_device
from .pipeline import DetectionPipeline


@dataclass(slots=True)
class Metrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    n: int
    extra: dict[str, float] = field(default_factory=dict)


def evaluate_model(
    model: SmallResNet,
    test_ds: CIFAKEDataset,
    device: str | None = None,
    batch_size: int = 256,
    console: Console | None = None,
) -> Metrics:
    console = console or Console()
    device = device or best_device()
    model = model.to(device).eval()
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    columns = (
        TextColumn("[bold]model eval[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    # Use no_grad rather than inference_mode — the latter has produced wrong
    # outputs on MPS in this codebase (model trained correctly, but predictions
    # under inference_mode collapsed to a single class).
    with torch.no_grad(), Progress(*columns, console=console, transient=True) as p:
        task_id = p.add_task("eval", total=len(loader))
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(fake)
            probs_all.append(probs)
            targets_all.append(y.numpy())
            p.update(task_id, advance=1)

    probs = np.concatenate(probs_all)
    targets = np.concatenate(targets_all)
    preds = (probs >= 0.5).astype(int)
    return _metrics("Trained CNN", targets, preds, probs)


def evaluate_pipeline(
    pipeline: DetectionPipeline,
    test_ds: CIFAKEDataset,
    limit: int | None = None,
    console: Console | None = None,
    seed: int = 0,
) -> Metrics:
    """Run the forensics pipeline image-by-image. Slow — pass ``limit`` to subsample."""
    console = console or Console()
    total = len(test_ds)
    n = total if limit is None else min(limit, total)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)[:n]

    probs: list[float] = []
    targets: list[int] = []
    columns = (
        TextColumn("[bold]forensics eval[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as p:
        task_id = p.add_task("eval", total=n)
        for i in indices:
            sample = test_ds.get_pil(int(i))
            try:
                report = pipeline.analyze(sample.image)
                probs.append(report.fake_score)
            except Exception:
                probs.append(0.5)  # neutral on failure
            targets.append(sample.target)
            p.update(task_id, advance=1)

    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)
    preds = (probs_arr >= 0.5).astype(int)
    return _metrics(
        "Forensics pipeline", targets_arr, preds, probs_arr,
        extra={"n_evaluated": float(n)},
    )


def run_full_evaluation(
    checkpoint: Path,
    pipeline_limit: int = 1000,
    batch_size: int = 256,
    device: str | None = None,
    console: Console | None = None,
) -> list[Metrics]:
    """Score the trained model on the full test set; the pipeline on a subsample."""
    from .training import load_checkpoint

    console = console or Console()
    _, _, _, test_ds = make_loaders(batch_size=batch_size, num_workers=0, augment=False)

    model = load_checkpoint(checkpoint, device=device)
    model_metrics = evaluate_model(model, test_ds, device=device, batch_size=batch_size, console=console)

    pipeline = DetectionPipeline.forensics_only()
    pipeline_metrics = evaluate_pipeline(pipeline, test_ds, limit=pipeline_limit, console=console)

    return [model_metrics, pipeline_metrics]


def _metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    extra: dict[str, float] | None = None,
) -> Metrics:
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = None
    return Metrics(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=auc,
        n=int(len(y_true)),
        extra=extra or {},
    )

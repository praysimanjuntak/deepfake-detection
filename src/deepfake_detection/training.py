"""Training loop for the small CNN on CIFAKE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.utils.data import DataLoader

from .cifake import make_loaders
from .model import SmallResNet, best_device

DEFAULT_CHECKPOINT = Path("checkpoints/smallresnet_cifake.pt")


@dataclass(slots=True)
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


@dataclass(slots=True)
class TrainResult:
    history: list[EpochStats] = field(default_factory=list)
    best_test_acc: float = 0.0
    checkpoint_path: Path | None = None


def train(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
    num_workers: int = 2,
    subset: int | None = None,
    device: str | None = None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    blocks_per_stage: int = 3,
    console: Console | None = None,
) -> TrainResult:
    console = console or Console()
    device = device or best_device()
    console.print(f"[bold]Device:[/bold] {device}")

    train_loader, test_loader, _, _ = make_loaders(
        batch_size=batch_size, num_workers=num_workers, subset=subset,
    )
    console.print(
        f"[bold]CIFAKE:[/bold] {len(train_loader.dataset)} train / "
        f"{len(test_loader.dataset)} test"
    )

    model = SmallResNet(num_classes=2, blocks_per_stage=blocks_per_stage).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Params:[/bold] {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    result = TrainResult()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, device, loss_fn, opt, console,
            label=f"epoch {epoch}/{epochs} train", train=True,
        )
        test_loss, test_acc = _run_epoch(
            model, test_loader, device, loss_fn, None, console,
            label=f"epoch {epoch}/{epochs}  test", train=False,
        )
        sched.step()
        stats = EpochStats(epoch, train_loss, train_acc, test_loss, test_acc)
        result.history.append(stats)
        console.print(
            f"[green]epoch {epoch}[/green] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > result.best_test_acc:
            result.best_test_acc = test_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "blocks_per_stage": blocks_per_stage,
                    "epoch": epoch,
                    "test_acc": test_acc,
                },
                checkpoint_path,
            )
            result.checkpoint_path = checkpoint_path
            console.print(f"  [dim]→ saved checkpoint to {checkpoint_path}[/dim]")

    return result


def _run_epoch(
    model: SmallResNet,
    loader: DataLoader,
    device: str,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    console: Console,
    label: str,
    train: bool,
) -> tuple[float, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[loss]:.4f} loss"),
        TextColumn("{task.fields[acc]:.4f} acc"),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task_id = progress.add_task(label, total=len(loader), loss=0.0, acc=0.0)
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = loss_fn(logits, y)
                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            batch_n = y.size(0)
            total_loss += loss.item() * batch_n
            total_correct += int((logits.argmax(1) == y).sum().item())
            total_seen += batch_n
            progress.update(
                task_id, advance=1,
                loss=total_loss / total_seen,
                acc=total_correct / total_seen,
            )

    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def load_checkpoint(path: Path, device: str | None = None) -> SmallResNet:
    device = device or best_device()
    payload = torch.load(path, map_location=device, weights_only=True)
    model = SmallResNet(num_classes=2, blocks_per_stage=payload.get("blocks_per_stage", 3))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model

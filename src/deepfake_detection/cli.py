"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .pipeline import DetectionPipeline, DetectionReport

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@click.group()
def cli() -> None:
    """Detect deepfakes and image forgery."""


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--no-model", is_flag=True, help="Skip the ML classifier (forensics only).")
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON report instead of a table.")
def detect(path: Path, no_model: bool, as_json: bool) -> None:
    """Analyze a single image."""
    pipeline = (
        DetectionPipeline.forensics_only() if no_model else DetectionPipeline()
    )
    console = Console()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task(f"Analyzing {path.name}", total=None)
        report = pipeline.analyze(path)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
        return

    _render(console, report)


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--no-model", is_flag=True, help="Skip the ML classifier (forensics only).")
@click.option(
    "--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
    help="Write a JSON-lines report to this file.",
)
def batch(folder: Path, no_model: bool, out_path: Path | None) -> None:
    """Analyze every image in a folder."""
    pipeline = (
        DetectionPipeline.forensics_only() if no_model else DetectionPipeline()
    )
    console = Console()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in _IMG_SUFFIXES)
    if not files:
        console.print(f"[yellow]No images under {folder}.[/yellow]")
        return

    table = Table(title=f"Batch: {folder}", show_lines=False)
    table.add_column("File", overflow="fold")
    table.add_column("Verdict")
    table.add_column("Fake score", justify="right")
    table.add_column("Confidence", justify="right")

    out_fh = out_path.open("w") if out_path else None
    try:
        for f in files:
            try:
                report = pipeline.analyze(f)
            except Exception as exc:  # noqa: BLE001 — surface per-file failures
                table.add_row(str(f.relative_to(folder)), "[red]error[/red]", "-", str(exc))
                continue
            color = {"fake": "red", "real": "green", "uncertain": "yellow"}[report.verdict]
            table.add_row(
                str(f.relative_to(folder)),
                f"[{color}]{report.verdict}[/{color}]",
                f"{report.fake_score:.2f}",
                f"{report.confidence:.2f}",
            )
            if out_fh:
                out_fh.write(json.dumps(report.to_dict(), default=str) + "\n")
    finally:
        if out_fh:
            out_fh.close()

    console.print(table)


def _render(console: Console, report: DetectionReport) -> None:
    color = {"fake": "red", "real": "green", "uncertain": "yellow"}[report.verdict]
    headline = (
        f"[bold {color}]{report.verdict.upper()}[/bold {color}]   "
        f"fake_score={report.fake_score:.3f}   confidence={report.confidence:.3f}"
    )
    console.print(Panel(headline, title=report.source or "image", border_style=color))

    sig_table = Table(title="Per-signal suspicion", show_header=True)
    sig_table.add_column("Signal")
    sig_table.add_column("Score", justify="right")
    sig_table.add_column("Detail")

    if report.classifier:
        cls = report.classifier
        sig_table.add_row(
            "ML classifier",
            f"{cls.fake_probability:.3f}",
            f"{cls.label} (model: {cls.model_id})",
        )
    sig_table.add_row(
        "ELA", f"{report.ela.suspicion:.3f}",
        f"mean_err={report.ela.mean_error:.2f} max_err={report.ela.max_error:.0f}",
    )
    sig_table.add_row(
        "Frequency", f"{report.frequency.suspicion:.3f}",
        f"hf_ratio={report.frequency.high_freq_energy:.2f} flatness={report.frequency.spectral_flatness:.3f}",
    )
    sig_table.add_row(
        "Noise residual", f"{report.noise.suspicion:.3f}",
        f"dispersion={report.noise.variance_dispersion:.3f}",
    )

    meta_detail = []
    if report.metadata.software:
        meta_detail.append(f"software={report.metadata.software}")
    if report.metadata.camera_model:
        meta_detail.append(f"camera={report.metadata.camera_model}")
    if report.metadata.flags:
        meta_detail.append("flags=" + "; ".join(report.metadata.flags))
    sig_table.add_row(
        "Metadata", f"{report.metadata.suspicion:.3f}",
        ", ".join(meta_detail) or "clean",
    )
    console.print(sig_table)


@cli.command()
@click.option("--epochs", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=128, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--workers", type=int, default=2, show_default=True, help="DataLoader workers.")
@click.option("--subset", type=int, default=None, help="Truncate each split (smoke-test mode).")
@click.option("--device", type=str, default=None, help="cpu / cuda / mps. Auto-detected if omitted.")
@click.option(
    "--checkpoint", type=click.Path(dir_okay=False, path_type=Path),
    default=Path("checkpoints/smallresnet_cifake.pt"), show_default=True,
)
def train(
    epochs: int, batch_size: int, lr: float, workers: int,
    subset: int | None, device: str | None, checkpoint: Path,
) -> None:
    """Train the small CNN on CIFAKE."""
    from .training import train as _train

    console = Console()
    result = _train(
        epochs=epochs, batch_size=batch_size, lr=lr,
        num_workers=workers, subset=subset, device=device,
        checkpoint_path=checkpoint, console=console,
    )
    console.print(
        Panel(
            f"best test accuracy: [bold green]{result.best_test_acc:.4f}[/bold green]\n"
            f"checkpoint: {result.checkpoint_path}",
            title="training complete", border_style="green",
        )
    )


@cli.command()
@click.option(
    "--checkpoint", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("checkpoints/smallresnet_cifake.pt"), show_default=True,
)
@click.option(
    "--pipeline-limit", type=int, default=1000, show_default=True,
    help="Subsample for the (slow) classical-forensics pipeline.",
)
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--device", type=str, default=None)
def evaluate(checkpoint: Path, pipeline_limit: int, batch_size: int, device: str | None) -> None:
    """Score the trained model + forensics pipeline on the CIFAKE test split."""
    from .evaluation import run_full_evaluation

    console = Console()
    metrics_list = run_full_evaluation(
        checkpoint=checkpoint,
        pipeline_limit=pipeline_limit,
        batch_size=batch_size,
        device=device,
        console=console,
    )

    table = Table(title="CIFAKE test results", show_lines=True)
    table.add_column("Method")
    table.add_column("N", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("AUC", justify="right")
    for m in metrics_list:
        table.add_row(
            m.name, str(m.n),
            f"{m.accuracy:.4f}", f"{m.precision:.4f}",
            f"{m.recall:.4f}", f"{m.f1:.4f}",
            "n/a" if m.auc is None else f"{m.auc:.4f}",
        )
    console.print(table)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()

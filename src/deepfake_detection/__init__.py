"""Deepfake & image-forgery detection toolkit."""

from .classifier import ClassifierResult, DeepfakeClassifier
from .forensics import (
    ELAResult,
    FrequencyResult,
    NoiseResult,
    error_level_analysis,
    frequency_analysis,
    noise_residual,
)
from .metadata import MetadataResult, inspect_metadata
from .pipeline import DetectionPipeline, DetectionReport

__all__ = [
    "ClassifierResult",
    "DeepfakeClassifier",
    "DetectionPipeline",
    "DetectionReport",
    "ELAResult",
    "FrequencyResult",
    "MetadataResult",
    "NoiseResult",
    "error_level_analysis",
    "frequency_analysis",
    "inspect_metadata",
    "noise_residual",
]


def main() -> None:
    """Entry point used by the ``deepfake-detection`` console script."""
    from .cli import main as _cli

    _cli()

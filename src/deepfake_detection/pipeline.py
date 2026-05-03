"""End-to-end detection pipeline.

Combines the pretrained classifier with classical forensics signals (ELA, FFT,
noise residual) and EXIF metadata into a single ``DetectionReport``. The
classical signals act as evidence even when the ML model is uncertain or
unavailable, and the per-signal breakdown is what makes the output explainable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

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

# How much each signal counts in the combined verdict.
_WEIGHTS = {
    "classifier": 0.55,
    "ela": 0.15,
    "frequency": 0.15,
    "noise": 0.10,
    "metadata": 0.05,
}


@dataclass(slots=True)
class DetectionReport:
    verdict: str  # "fake", "real", or "uncertain"
    fake_score: float  # 0..1, higher = more likely fake
    confidence: float  # 0..1, distance from 0.5
    classifier: ClassifierResult | None
    ela: ELAResult
    frequency: FrequencyResult
    noise: NoiseResult
    metadata: MetadataResult
    signal_scores: dict[str, float] = field(default_factory=dict)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "verdict": self.verdict,
            "fake_score": self.fake_score,
            "confidence": self.confidence,
            "signal_scores": self.signal_scores,
            "source": self.source,
        }
        if self.classifier is not None:
            out["classifier"] = {
                k: v for k, v in asdict(self.classifier).items() if k != "raw_scores"
            } | {"raw_scores": self.classifier.raw_scores}
        out["forensics"] = {
            "ela": {
                "mean_error": self.ela.mean_error,
                "max_error": self.ela.max_error,
                "suspicion": self.ela.suspicion,
            },
            "frequency": {
                "high_freq_energy": self.frequency.high_freq_energy,
                "spectral_flatness": self.frequency.spectral_flatness,
                "suspicion": self.frequency.suspicion,
            },
            "noise": {
                "variance_dispersion": self.noise.variance_dispersion,
                "suspicion": self.noise.suspicion,
            },
        }
        out["metadata"] = {
            "has_exif": self.metadata.has_exif,
            "camera_make": self.metadata.camera_make,
            "camera_model": self.metadata.camera_model,
            "software": self.metadata.software,
            "flags": self.metadata.flags,
            "suspicion": self.metadata.suspicion,
        }
        return out


class DetectionPipeline:
    def __init__(
        self,
        classifier: DeepfakeClassifier | None = None,
        use_classifier: bool = True,
    ) -> None:
        self.use_classifier = use_classifier
        self.classifier = classifier if classifier is not None else DeepfakeClassifier()

    @classmethod
    def forensics_only(cls) -> "DetectionPipeline":
        """Pipeline that skips the ML model — useful for tests or offline use."""
        return cls(classifier=None, use_classifier=False)

    def analyze(self, source: str | Path | Image.Image) -> DetectionReport:
        image, source_str = _load(source)

        cls_result: ClassifierResult | None = None
        if self.use_classifier and self.classifier is not None:
            cls_result = self.classifier.predict(image)

        ela = error_level_analysis(image)
        freq = frequency_analysis(image)
        noise = noise_residual(image)
        meta = inspect_metadata(image)

        signal_scores: dict[str, float] = {
            "ela": ela.suspicion,
            "frequency": freq.suspicion,
            "noise": noise.suspicion,
            "metadata": meta.suspicion,
        }
        if cls_result is not None:
            signal_scores["classifier"] = cls_result.fake_probability

        fake_score = _weighted_score(signal_scores)
        confidence = abs(fake_score - 0.5) * 2
        if confidence < 0.2:
            verdict = "uncertain"
        else:
            verdict = "fake" if fake_score >= 0.5 else "real"

        return DetectionReport(
            verdict=verdict,
            fake_score=fake_score,
            confidence=confidence,
            classifier=cls_result,
            ela=ela,
            frequency=freq,
            noise=noise,
            metadata=meta,
            signal_scores=signal_scores,
            source=source_str,
        )


def _weighted_score(scores: dict[str, float]) -> float:
    total_weight = 0.0
    total = 0.0
    for name, value in scores.items():
        w = _WEIGHTS.get(name, 0.0)
        total_weight += w
        total += w * value
    return total / total_weight if total_weight > 0 else 0.0


def _load(source: str | Path | Image.Image) -> tuple[Image.Image, str | None]:
    if isinstance(source, Image.Image):
        return source, None
    path = Path(source)
    return Image.open(path), str(path)

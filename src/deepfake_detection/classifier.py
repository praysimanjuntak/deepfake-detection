"""Pretrained deepfake classifier.

Wraps a HuggingFace image-classification model. The default checkpoint is
``prithivMLmods/Deep-Fake-Detector-v2-Model``, an open SigLIP-based detector
trained for AI-generated vs. real images. It's swappable via constructor
argument or the ``DEEPFAKE_MODEL_ID`` env var.

The model is loaded lazily on first call so the rest of the package (CLI help,
forensics-only flows, tests) doesn't pay the import/download cost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_MODEL_ID = os.environ.get(
    "DEEPFAKE_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-v2-Model"
)

# Different checkpoints disagree on label spelling. Anything that looks like
# a synthetic / fake / AI-generated label maps to "fake".
_FAKE_LABEL_TOKENS = ("fake", "deepfake", "synthetic", "ai", "generated", "manipulated")


@dataclass(slots=True)
class ClassifierResult:
    label: str  # "real" or "fake"
    fake_probability: float
    real_probability: float
    raw_scores: dict[str, float]
    model_id: str


class DeepfakeClassifier:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @cached_property
    def _bundle(self) -> tuple[AutoImageProcessor, AutoModelForImageClassification]:
        processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageClassification.from_pretrained(self.model_id)
        model.to(self.device).eval()
        return processor, model

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> ClassifierResult:
        processor, model = self._bundle
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

        id2label = model.config.id2label
        raw = {id2label[i].lower(): float(p) for i, p in enumerate(probs)}

        fake_prob = sum(p for label, p in raw.items() if any(t in label for t in _FAKE_LABEL_TOKENS))
        real_prob = max(0.0, 1.0 - fake_prob)
        label = "fake" if fake_prob >= 0.5 else "real"

        return ClassifierResult(
            label=label,
            fake_probability=fake_prob,
            real_probability=real_prob,
            raw_scores=raw,
            model_id=self.model_id,
        )

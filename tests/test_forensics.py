"""Smoke tests — they only assert on shapes and bounded ranges, not on
specific suspicion values. The forensics signals are heuristics: the goal is
to catch regressions where they crash, return wrong shapes, or emit
out-of-range scores.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from deepfake_detection import (
    DetectionPipeline,
    error_level_analysis,
    frequency_analysis,
    inspect_metadata,
    noise_residual,
)


def _photo_like() -> Image.Image:
    rng = np.random.default_rng(0)
    base = rng.integers(80, 180, size=(128, 128, 3), dtype=np.uint8)
    # Add a smooth gradient so it isn't pure noise.
    yy, xx = np.mgrid[0:128, 0:128]
    base[..., 0] = np.clip(base[..., 0] + (xx // 2), 0, 255)
    return Image.fromarray(base)


def _composite() -> Image.Image:
    img = _photo_like()
    arr = np.asarray(img).copy()
    # Paste a flat-color patch — the kind of thing ELA + noise should react to.
    arr[40:80, 40:80] = (220, 30, 30)
    return Image.fromarray(arr)


def test_ela_shape_and_range() -> None:
    img = _photo_like()
    out = error_level_analysis(img)
    assert out.heatmap.shape == (128, 128)
    assert out.heatmap.dtype == np.uint8
    assert 0.0 <= out.suspicion <= 1.0


def test_frequency_shape_and_range() -> None:
    img = _photo_like()
    out = frequency_analysis(img)
    assert out.spectrum.shape == (128, 128)
    assert 0.0 <= out.suspicion <= 1.0


def test_noise_residual_shape_and_range() -> None:
    img = _photo_like()
    out = noise_residual(img)
    assert out.residual.shape == (128, 128)
    assert 0.0 <= out.suspicion <= 1.0


def test_metadata_no_exif_flags() -> None:
    img = _photo_like()
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    out = inspect_metadata(Image.open(buf))
    assert not out.has_exif
    assert "no EXIF metadata" in out.flags
    assert out.suspicion > 0


def test_pipeline_forensics_only_runs() -> None:
    pipeline = DetectionPipeline.forensics_only()
    report = pipeline.analyze(_composite())
    assert report.verdict in {"fake", "real", "uncertain"}
    assert 0.0 <= report.fake_score <= 1.0
    assert "classifier" not in report.signal_scores
    # Forensics-only report still serializes cleanly.
    payload = report.to_dict()
    assert "forensics" in payload
    assert "ela" in payload["forensics"]

"""Render the README demo image.

Builds a synthetic forged JPEG (a flat-color patch composited onto a
textured gradient — the kind of edit ELA was made for), runs the full
forensics pipeline, and assembles a 2x2 panel comparing the input to the
ELA / FFT / noise-residual heatmaps. Output: ``docs/demo.png``.

Run with:
    uv run python scripts/make_demo.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deepfake_detection.forensics import (
    error_level_analysis,
    frequency_analysis,
    noise_residual,
)
from deepfake_detection.pipeline import DetectionPipeline

OUT = Path(__file__).resolve().parents[1] / "docs" / "demo.png"
SIZE = 320  # per panel


def _natural_looking(seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.clip(50 + xx * 0.45 + rng.integers(0, 28, (SIZE, SIZE)), 0, 255)
    g = np.clip(80 + (yy + xx) * 0.28 + rng.integers(0, 22, (SIZE, SIZE)), 0, 255)
    b = np.clip(190 - xx * 0.4 + rng.integers(0, 22, (SIZE, SIZE)), 0, 255)
    base = np.stack([r, g, b], axis=-1).astype(np.uint8)

    # A few patches of texture so the image has some structure rather than
    # being pure gradient — gives ELA something realistic to chew on.
    for _ in range(7):
        h, w = rng.integers(20, 55, size=2)
        y, x = rng.integers(0, SIZE - h), rng.integers(0, SIZE - w)
        tint = rng.integers(40, 220, 3, dtype=np.int32)
        base[y : y + h, x : x + w] = (
            base[y : y + h, x : x + w].astype(np.int32) * 0.55 + tint * 0.45
        ).astype(np.uint8)
    return _jpeg_roundtrip(Image.fromarray(base))


def _composite_forgery(img: Image.Image) -> Image.Image:
    """Paste a flat-color patch — a textbook splice for ELA to detect."""
    arr = np.asarray(img).copy()
    arr[110:200, 130:230] = (220, 35, 40)  # bright red rectangle
    return _jpeg_roundtrip(Image.fromarray(arr))


def _jpeg_roundtrip(img: Image.Image, quality: int = 88) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _colorize(gray: np.ndarray) -> Image.Image:
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    forged = _composite_forgery(_natural_looking())
    ela = error_level_analysis(forged)
    freq = frequency_analysis(forged)
    noise = noise_residual(forged)

    report = DetectionPipeline.forensics_only().analyze(forged)

    pad = 24
    title_h = 80
    label_h = 28
    cell = SIZE
    canvas_w = pad * 3 + cell * 2
    canvas_h = title_h + pad * 3 + cell * 2 + label_h * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 23))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(22)
    sub_font = _font(14)
    label_font = _font(15)

    draw.text(
        (pad, 18),
        "Deepfake & Image Forgery Detection — sample output",
        fill=(230, 230, 240),
        font=title_font,
    )
    verdict_color = {"fake": (240, 90, 90), "real": (80, 200, 120), "uncertain": (240, 200, 80)}[
        report.verdict
    ]
    draw.text(
        (pad, 50),
        f"verdict: {report.verdict.upper()}    fake_score: {report.fake_score:.3f}    "
        f"ELA: {ela.suspicion:.2f}    Freq: {freq.suspicion:.2f}    Noise: {noise.suspicion:.2f}",
        fill=verdict_color,
        font=sub_font,
    )

    panels = [
        ("Forged input — red patch is the splice", forged, 0, 0),
        ("Error Level Analysis", _colorize(ela.heatmap), 1, 0),
        ("Frequency spectrum (log-FFT)", _colorize(freq.spectrum), 0, 1),
        ("Noise residual", _colorize(noise.residual), 1, 1),
    ]
    for label, img, col, row in panels:
        x = pad + col * (cell + pad)
        y = title_h + pad + row * (cell + label_h + pad)
        draw.text((x, y), label, fill=(220, 220, 230), font=label_font)
        canvas.paste(img.resize((cell, cell), Image.LANCZOS), (x, y + label_h))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, format="PNG", optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

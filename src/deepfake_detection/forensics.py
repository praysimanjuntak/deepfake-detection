"""Classical image-forgery forensics.

Three lightweight signals that don't need a trained model:

- Error Level Analysis (ELA): re-saves a JPEG at a known quality and looks at
  per-pixel difference. Edited regions tend to compress differently from the
  rest of the image, so they "light up" in the residual.
- Frequency-domain analysis: GAN/diffusion outputs leave distinctive spectral
  fingerprints (often a high-frequency cross or grid in the FFT magnitude).
- Noise residual: the noise pattern of a real sensor is locally consistent.
  Inserted regions usually break that consistency.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(slots=True)
class ELAResult:
    heatmap: np.ndarray  # uint8 grayscale, same HxW as input
    mean_error: float
    max_error: float
    suspicion: float  # 0..1


@dataclass(slots=True)
class FrequencyResult:
    spectrum: np.ndarray  # log-magnitude FFT, uint8 grayscale
    high_freq_energy: float
    spectral_flatness: float
    suspicion: float  # 0..1


@dataclass(slots=True)
class NoiseResult:
    residual: np.ndarray  # uint8 grayscale
    variance_dispersion: float
    suspicion: float  # 0..1


def error_level_analysis(image: Image.Image, quality: int = 90, scale: int = 15) -> ELAResult:
    """Compute ELA by re-encoding at a fixed JPEG quality and diffing.

    The returned heatmap is amplified by ``scale`` so subtle edits are visible.
    """
    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    original = np.asarray(rgb, dtype=np.int16)
    again = np.asarray(recompressed, dtype=np.int16)
    diff = np.abs(original - again).astype(np.uint8)

    amplified = np.clip(diff.astype(np.int32) * scale, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(amplified, cv2.COLOR_RGB2GRAY)

    mean_error = float(diff.mean())
    max_error = float(diff.max())
    # Suspicion grows with both mean error and how concentrated the bright
    # pixels are. Pure JPEG compression artifacts are diffuse; edits cluster.
    bright_ratio = float((gray > 40).mean())
    suspicion = float(np.clip(mean_error / 12.0 + bright_ratio * 1.5, 0.0, 1.0))

    return ELAResult(heatmap=gray, mean_error=mean_error, max_error=max_error, suspicion=suspicion)


def frequency_analysis(image: Image.Image) -> FrequencyResult:
    """Look at the log-magnitude FFT for GAN/diffusion-style fingerprints."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(f))

    # Normalize for display.
    spectrum = magnitude - magnitude.min()
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    spectrum_u8 = (spectrum * 255).astype(np.uint8)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    yy, xx = np.ogrid[:h, :w]
    high_mask = (yy - cy) ** 2 + (xx - cx) ** 2 > radius**2
    low_mask = ~high_mask

    high_energy = float(magnitude[high_mask].mean())
    low_energy = float(magnitude[low_mask].mean()) + 1e-6
    ratio = high_energy / low_energy

    # Spectral flatness: geometric mean / arithmetic mean of |F|^2.
    power = np.abs(f) ** 2 + 1e-12
    geo = np.exp(np.log(power).mean())
    arith = power.mean()
    flatness = float(geo / arith)

    # GAN/diffusion outputs tend to push *up* high-frequency energy.
    suspicion = float(np.clip((ratio - 0.55) / 0.4 + (0.5 - flatness) * 0.5, 0.0, 1.0))

    return FrequencyResult(
        spectrum=spectrum_u8,
        high_freq_energy=ratio,
        spectral_flatness=flatness,
        suspicion=suspicion,
    )


def noise_residual(image: Image.Image, ksize: int = 3) -> NoiseResult:
    """Estimate the high-frequency noise residual via median-filter subtraction.

    Real camera sensors imprint a roughly stationary noise field (PRNU-ish).
    Pasted regions break that stationarity, which shows up as a mismatch in the
    block-wise variance of the residual.
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.medianBlur(gray, ksize)
    residual = cv2.absdiff(gray, smooth)

    h, w = residual.shape
    bh, bw = max(h // 16, 8), max(w // 16, 8)
    blocks = []
    for y in range(0, h - bh, bh):
        for x in range(0, w - bw, bw):
            blocks.append(residual[y : y + bh, x : x + bw].var())
    if not blocks:
        return NoiseResult(residual=residual, variance_dispersion=0.0, suspicion=0.0)

    blocks_arr = np.asarray(blocks, dtype=np.float64)
    median = float(np.median(blocks_arr)) + 1e-6
    mad = float(np.median(np.abs(blocks_arr - median)))
    dispersion = mad / median  # robust coefficient of variation

    # Highly inhomogeneous noise → edited or composited.
    suspicion = float(np.clip((dispersion - 0.3) / 0.7, 0.0, 1.0))

    return NoiseResult(
        residual=cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        variance_dispersion=dispersion,
        suspicion=suspicion,
    )

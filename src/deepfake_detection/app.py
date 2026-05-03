"""Gradio demo app.

Run with:
    uv run python -m deepfake_detection.app
"""

from __future__ import annotations

from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from .pipeline import DetectionPipeline

_pipeline_singleton: DetectionPipeline | None = None
_forensics_only_singleton: DetectionPipeline | None = None


def _get_pipeline(use_model: bool) -> DetectionPipeline:
    global _pipeline_singleton, _forensics_only_singleton
    if use_model:
        if _pipeline_singleton is None:
            _pipeline_singleton = DetectionPipeline()
        return _pipeline_singleton
    if _forensics_only_singleton is None:
        _forensics_only_singleton = DetectionPipeline.forensics_only()
    return _forensics_only_singleton


def _colormap(gray: np.ndarray) -> np.ndarray:
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _format_summary(report: Any) -> str:
    color = {"fake": "🔴", "real": "🟢", "uncertain": "🟡"}[report.verdict]
    lines = [
        f"## {color} Verdict: **{report.verdict.upper()}**",
        f"- **Fake score:** `{report.fake_score:.3f}`  (0 = real, 1 = fake)",
        f"- **Confidence:** `{report.confidence:.3f}`",
        "",
        "### Per-signal suspicion",
    ]
    if report.classifier:
        lines.append(
            f"- **ML classifier** — fake prob `{report.classifier.fake_probability:.3f}` "
            f"(label: `{report.classifier.label}`)"
        )
    lines.extend(
        [
            f"- **ELA** — `{report.ela.suspicion:.3f}` "
            f"(mean error `{report.ela.mean_error:.2f}`)",
            f"- **Frequency** — `{report.frequency.suspicion:.3f}` "
            f"(high-freq ratio `{report.frequency.high_freq_energy:.2f}`, "
            f"flatness `{report.frequency.spectral_flatness:.3f}`)",
            f"- **Noise residual** — `{report.noise.suspicion:.3f}` "
            f"(variance dispersion `{report.noise.variance_dispersion:.3f}`)",
            f"- **Metadata** — `{report.metadata.suspicion:.3f}`",
        ]
    )
    if report.metadata.flags:
        lines.append("  - " + "; ".join(report.metadata.flags))
    if report.metadata.software:
        lines.append(f"  - software: `{report.metadata.software}`")
    if report.metadata.camera_model:
        lines.append(
            f"  - camera: `{report.metadata.camera_make or '?'} {report.metadata.camera_model}`"
        )
    return "\n".join(lines)


def analyze(image: Image.Image | None, use_model: bool) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if image is None:
        return "Upload an image to begin.", None, None, None
    pipeline = _get_pipeline(use_model)
    report = pipeline.analyze(image)
    return (
        _format_summary(report),
        _colormap(report.ela.heatmap),
        _colormap(report.frequency.spectrum),
        _colormap(report.noise.residual),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Deepfake & Forgery Detection") as demo:
        gr.Markdown(
            """
            # Deepfake & Image Forgery Detection
            Upload an image to get a verdict combining a pretrained classifier
            with classical forensics signals (ELA, FFT, noise residual) and
            EXIF metadata. Heatmaps below show *where* the signals fired.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Image", type="pil", sources=["upload", "clipboard"])
                use_model = gr.Checkbox(label="Use ML classifier", value=True)
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                summary = gr.Markdown()
        with gr.Row():
            ela = gr.Image(label="Error Level Analysis")
            freq = gr.Image(label="Frequency spectrum (log-FFT)")
            noise = gr.Image(label="Noise residual")
        btn.click(analyze, inputs=[inp, use_model], outputs=[summary, ela, freq, noise])
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()

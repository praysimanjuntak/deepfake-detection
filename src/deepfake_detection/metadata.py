"""EXIF / container metadata inspection.

Missing or stripped EXIF is not proof of forgery, but it's a meaningful prior:
real photos out of cameras and most phones carry rich EXIF; AI-generated and
heavily-edited images usually don't. Software tags ("Photoshop", "Stable
Diffusion", "Midjourney") are a louder signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import ExifTags, Image

_GENERATOR_HINTS = (
    "stable diffusion",
    "stable-diffusion",
    "midjourney",
    "dalle",
    "dall-e",
    "comfyui",
    "automatic1111",
    "flux",
    "imagen",
    "sora",
)
_EDIT_HINTS = ("photoshop", "gimp", "lightroom", "affinity", "pixelmator")


@dataclass(slots=True)
class MetadataResult:
    has_exif: bool
    camera_make: str | None
    camera_model: str | None
    software: str | None
    flags: list[str] = field(default_factory=list)
    suspicion: float = 0.0


def inspect_metadata(image: Image.Image) -> MetadataResult:
    raw = getattr(image, "_getexif", lambda: None)() or {}
    exif = {ExifTags.TAGS.get(tag, str(tag)): value for tag, value in raw.items()}

    make = _stringify(exif.get("Make"))
    model = _stringify(exif.get("Model"))
    software = _stringify(exif.get("Software"))

    flags: list[str] = []
    suspicion = 0.0

    if not exif:
        flags.append("no EXIF metadata")
        suspicion += 0.4
    else:
        if not make and not model:
            flags.append("no camera make/model")
            suspicion += 0.2

    if software:
        lowered = software.lower()
        if any(h in lowered for h in _GENERATOR_HINTS):
            flags.append(f"AI generator tag: {software}")
            suspicion += 0.5
        elif any(h in lowered for h in _EDIT_HINTS):
            flags.append(f"editor tag: {software}")
            suspicion += 0.15

    return MetadataResult(
        has_exif=bool(exif),
        camera_make=make,
        camera_model=model,
        software=software,
        flags=flags,
        suspicion=min(suspicion, 1.0),
    )


def _stringify(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip("\x00 ").strip() or None
        except Exception:
            return None
    text = str(value).strip()
    return text or None

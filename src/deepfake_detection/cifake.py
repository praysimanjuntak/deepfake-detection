"""CIFAKE dataset loader.

CIFAKE = 60k real (CIFAR-10) + 60k Stable-Diffusion generated images at 32x32.
We pull it from the HuggingFace mirror so users don't need a Kaggle account.

Original labels: 0 = FAKE, 1 = REAL.
We re-map to the convention used elsewhere in this project: target=1 means fake.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_HF_ID = "dragonintelligence/CIFAKE-image-dataset"

# CIFAR-10 stats — close enough since CIFAKE's real half *is* CIFAR-10 and the
# fake half is rendered to match.
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2470, 0.2435, 0.2616)


@dataclass(slots=True)
class Sample:
    image: Image.Image
    target: int  # 1 = fake, 0 = real


class CIFAKEDataset(Dataset):
    """Wraps a HuggingFace split as a torch Dataset returning normalized tensors."""

    def __init__(self, hf_split, augment: bool = False) -> None:
        self.hf = hf_split
        self.augment = augment
        self._mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
        self._std = torch.tensor(NORM_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.hf[idx]
        img = _to_pil(row["image"])
        target = _remap_label(int(row["label"]))
        if self.augment and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - self._mean) / self._std
        return tensor, target

    def get_pil(self, idx: int) -> Sample:
        """Return the raw PIL image — used by the forensics pipeline."""
        row = self.hf[idx]
        return Sample(image=_to_pil(row["image"]).convert("RGB"), target=_remap_label(int(row["label"])))


def load_cifake(hf_id: str = DEFAULT_HF_ID):
    """Download (or load from cache) and return the HF DatasetDict."""
    return load_dataset(hf_id)


def make_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    hf_id: str = DEFAULT_HF_ID,
    subset: int | None = None,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, CIFAKEDataset, CIFAKEDataset]:
    """Build train/test DataLoaders. ``subset`` truncates each split for quick smoke tests.

    The CIFAKE splits are sorted by class on disk, so a naive head-N slice is
    single-class. We always shuffle before slicing.
    """
    ds = load_cifake(hf_id)
    train_hf = ds["train"].shuffle(seed=seed)
    test_hf = ds["test"].shuffle(seed=seed)
    if subset is not None:
        train_hf = train_hf.select(range(min(subset, len(train_hf))))
        test_hf = test_hf.select(range(min(subset, len(test_hf))))

    train_ds = CIFAKEDataset(train_hf, augment=augment)
    test_ds = CIFAKEDataset(test_hf, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, train_ds, test_ds


def _remap_label(raw: int) -> int:
    # Source: 0 = FAKE, 1 = REAL → invert so target=1 means fake.
    return 1 - raw


def _to_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and "bytes" in value:
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    raise TypeError(f"Unexpected image payload type: {type(value)!r}")

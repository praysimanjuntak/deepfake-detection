"""Small CIFAR-style CNN for binary real/fake classification.

A vanilla torchvision ResNet-18 is overkill for 32x32 input — its 7x7 stride-2
stem destroys most of the signal before the first residual block. This module
defines a small ResNet variant adapted to 32x32, similar to the reference
ResNet-20 used in the original CIFAR papers. Roughly ~270k params, trains in
minutes on a laptop, and is plenty for CIFAKE.

Normalization: we use GroupNorm rather than BatchNorm. PyTorch's MPS backend
has historically corrupted BN's running mean/var into NaN after a handful of
training steps; the model trains fine but eval-mode predictions then collapse
to a single class. GroupNorm has no running stats and sidesteps that whole
class of bug while training just as well at this scale.
"""

from __future__ import annotations

import torch
from torch import nn

# Channels are kept divisible by 4 throughout, so group_size=4 always works.
_GROUP_SIZE = 4


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=max(1, channels // _GROUP_SIZE), num_channels=channels)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = _norm(out_ch)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _norm(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class SmallResNet(nn.Module):
    """ResNet-style net for 32x32 input. blocks_per_stage=3 → ~ResNet-20."""

    def __init__(self, num_classes: int = 2, blocks_per_stage: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            _norm(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(16, 16, blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(16, 32, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(32, 64, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def best_device() -> str:
    """Default device. Avoids MPS by default — see note below.

    Apple's MPS backend has had recurring bugs that corrupt small networks'
    normalization parameters into NaN after a handful of training steps. The
    model is small enough (~270k params, 32x32 input) that CPU training is
    perfectly fast (a couple of minutes for full CIFAKE), so we default to
    CPU for correctness. Pass ``--device mps`` explicitly to opt back in.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

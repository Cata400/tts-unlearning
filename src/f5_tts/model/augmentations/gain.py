from __future__ import annotations

import math
from typing import Any

import torch

from f5_tts.model.augmentations.base import LOG_MEL_FLOOR, Augmentation, uniform


class MelGain(Augmentation):
    """Loudness jitter applied as an offset in the log-mel domain.

    `log(g * |S|) == log|S| + log(g)`, so this is exactly equivalent to scaling the waveform.
    """

    name = "gain"
    stage = "mel"

    def __init__(self, p: float, gain_db: tuple[float, float]) -> None:
        super().__init__(p)
        low, high = float(gain_db[0]), float(gain_db[1])
        if low > high:
            raise ValueError(f"gain: gain_db must be [min, max], got {gain_db}")
        self.gain_db = (low, high)

    def sample_params(self) -> dict[str, Any]:
        return {"gain_db": uniform(*self.gain_db)}

    def apply_with(
        self,
        x: torch.Tensor,
        params: dict[str, Any],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        offset = params["gain_db"] / 20.0 * math.log(10.0)
        out = (x + offset).clamp(min=LOG_MEL_FLOOR)
        return out if mask is None else torch.where(mask, out, x)

    def describe(self) -> str:
        return f"gain(p={self.p}, gain_db={list(self.gain_db)})"

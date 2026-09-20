from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

# MelSpec clamps magnitudes at 1e-5 before taking the log, so this is the value of silence.
# Padded frames hold 0.0, which is magnitude 1.0 - loud, not silent.
LOG_MEL_FLOOR = math.log(1e-5)


def uniform(low: float, high: float) -> float:
    """One sample from U(low, high), drawn from the global torch RNG."""
    return float(torch.empty(()).uniform_(float(low), float(high)))


def draws(p: float) -> bool:
    """True with probability `p`, drawn from the global torch RNG."""
    return p > 0.0 and float(torch.rand(())) < p


class Augmentation(ABC):
    """A single augmentation operator.

    Drawing is deliberately split from applying so one implementation serves both granularities:
    the per-sample path calls `__call__`, while `BatchAugmentor` calls `sample_params` once and
    reuses the result across every sample in a batch.
    """

    name: str = "base"
    stage: str = "mel"  # waveform | mel

    def __init__(self, p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"augmentation '{self.name}': p must be in [0, 1], got {p}")
        self.p = float(p)

    @abstractmethod
    def sample_params(self) -> dict[str, Any]:
        """Randomly drawn parameters for one application of this operator."""

    @abstractmethod
    def apply_with(
        self,
        x: torch.Tensor,
        params: dict[str, Any],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`x` transformed by `params`, with the length of the last dimension preserved.

        `x` is `[..., T]` so an optional leading batch dimension broadcasts. `mask` marks valid
        frames and must be broadcastable to `x`; padded positions may neither influence the result
        nor be modified by it.
        """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not draws(self.p):
            return x
        return self.apply_with(x, self.sample_params())

    def describe(self) -> str:
        return f"{self.name}(p={self.p})"

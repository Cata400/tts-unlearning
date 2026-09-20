from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from f5_tts.model.augmentations.base import Augmentation, uniform


class MelReverb(Augmentation):
    """Approximate reverberation: the per-bin energy envelope convolved with an exponential decay.

    A real impulse response cannot be applied in the mel domain - phase is gone and the mel filters
    have already smeared frequency. This models only the energy decay, which is the part the mel
    actually carries. Use the waveform-stage `rir` operator when physical accuracy matters.
    """

    name = "reverb"
    stage = "mel"

    def __init__(
        self,
        p: float,
        rt60_s: tuple[float, float],
        wet_level: tuple[float, float],
        hop_length: int,
        target_sample_rate: int,
        max_kernel_frames: int = 200,
    ) -> None:
        super().__init__(p)
        if float(rt60_s[0]) <= 0.0:
            raise ValueError(f"reverb: rt60_s must be positive, got {rt60_s}")
        if float(rt60_s[0]) > float(rt60_s[1]):
            raise ValueError(f"reverb: rt60_s must be [min, max], got {rt60_s}")
        if not 0.0 <= float(wet_level[0]) <= float(wet_level[1]) <= 1.0:
            raise ValueError(f"reverb: wet_level must be [min, max] within [0, 1], got {wet_level}")

        self.rt60_s = (float(rt60_s[0]), float(rt60_s[1]))
        self.wet_level = (float(wet_level[0]), float(wet_level[1]))
        self.max_kernel_frames = int(max_kernel_frames)
        self.frame_rate = target_sample_rate / hop_length

    def sample_params(self) -> dict[str, Any]:
        return {"rt60_s": uniform(*self.rt60_s), "wet_level": uniform(*self.wet_level)}

    def _kernel(self, rt60: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        decay_frames = rt60 * self.frame_rate
        length = max(2, min(int(math.ceil(decay_frames)), self.max_kernel_frames))
        taps = torch.arange(length, device=device, dtype=dtype)
        kernel = torch.exp(-3.0 * math.log(10.0) * taps / decay_frames)
        return kernel / kernel.sum()

    def apply_with(
        self,
        x: torch.Tensor,
        params: dict[str, Any],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = x.dim() == 2
        mel = x.unsqueeze(0) if unbatched else x

        energy = mel.exp().pow(2)
        if mask is not None:
            # Padding is 0.0 in the log domain, i.e. magnitude 1.0; left in, it would smear
            # backwards into real frames through the convolution.
            energy = energy * mask.to(energy.dtype)

        kernel = self._kernel(params["rt60_s"], mel.device, mel.dtype)
        n_mel_channels = energy.size(-2)
        taps = kernel.numel()
        # conv1d cross-correlates, so flip for a causal convolution; left padding keeps T fixed.
        weight = kernel.flip(0).view(1, 1, taps).expand(n_mel_channels, 1, taps)
        wet = F.conv1d(F.pad(energy, (taps - 1, 0)), weight, groups=n_mel_channels)

        level = params["wet_level"]
        out = 0.5 * ((1.0 - level) * energy + level * wet).clamp(min=1e-10).log()
        if unbatched:
            out = out.squeeze(0)
        return out if mask is None else torch.where(mask, out, x)

    def describe(self) -> str:
        return f"reverb(p={self.p}, rt60_s={list(self.rt60_s)}, wet_level={list(self.wet_level)})"

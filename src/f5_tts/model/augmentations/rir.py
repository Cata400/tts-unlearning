from __future__ import annotations

from typing import Any

import torch
import torchaudio

from f5_tts.model.augmentations.base import Augmentation


class WaveformRIR(Augmentation):
    """Convolution with a synthesised room impulse response, applied to the raw waveform.

    Always per sample: by the time a batch exists the audio is gone, so this operator keeps drawing
    independently even under `apply_to: batch`.
    """

    name = "rir"
    stage = "waveform"

    def __init__(self, p: float, bank: list[torch.Tensor], match_input_rms: bool = True) -> None:
        super().__init__(p)
        if not bank:
            raise ValueError("rir: the impulse response bank is empty")
        self.bank = bank
        # The direct path is the peak; slicing from it drops both the ISM fractional-delay offset
        # and the propagation delay, so the output stays aligned with the transcript.
        self.peaks = [int(rir.abs().argmax()) for rir in bank]
        self.match_input_rms = bool(match_input_rms)

    def sample_params(self) -> dict[str, Any]:
        return {"index": int(torch.randint(len(self.bank), ()))}

    def apply_with(
        self,
        x: torch.Tensor,
        params: dict[str, Any],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        index = params["index"]
        rir = self.bank[index].to(device=x.device, dtype=x.dtype)
        samples = x.shape[-1]
        peak = self.peaks[index]

        # `peak + samples` never exceeds the full convolution length `samples + rir.numel() - 1`.
        wet = torchaudio.functional.fftconvolve(x, rir.expand(*x.shape[:-1], rir.numel()))
        out = wet[..., peak : peak + samples]

        if self.match_input_rms:
            scale = x.pow(2).mean().sqrt() / out.pow(2).mean().sqrt().clamp(min=1e-8)
            out = out * scale
        return out

    def describe(self) -> str:
        return f"rir(p={self.p}, bank_size={len(self.bank)})"

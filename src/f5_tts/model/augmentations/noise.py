from __future__ import annotations

import math
from typing import Any

import torch

from f5_tts.model.augmentations.base import Augmentation, uniform

COLOURS = ("white", "pink", "brown")


def mel_centre_frequencies(n_mel_channels: int, sample_rate: int, f_min: float = 0.0) -> torch.Tensor:
    """Centre frequency in Hz of each mel bin, on the HTK mel scale used by the mel front-end."""
    f_max = sample_rate / 2.0

    def to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    points = torch.linspace(to_mel(f_min), to_mel(f_max), n_mel_channels + 2)
    frequencies = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    return frequencies[1:-1]


def colour_profile(colour: str, n_mel_channels: int, sample_rate: int) -> torch.Tensor:
    """Per-bin noise amplitude weighting, normalised to unit mean power. Shape `[n_mel_channels, 1]`."""
    if colour not in COLOURS:
        raise ValueError(f"noise: colour must be one of {COLOURS}, got '{colour}'")

    centres = mel_centre_frequencies(n_mel_channels, sample_rate).clamp(min=1.0)
    if colour == "white":
        profile = torch.ones_like(centres)
    elif colour == "pink":
        profile = centres.rsqrt()
    else:
        profile = 1.0 / centres

    profile = profile / profile.pow(2).mean().sqrt()
    return profile.unsqueeze(-1)


class MelNoise(Augmentation):
    """A synthetic noise floor mixed in at a target SNR.

    The mel holds log-magnitudes, so the noise is added in the linear domain and re-logged; adding
    it directly to the log values would be a multiplicative, signal-dependent distortion instead.
    """

    name = "noise"
    stage = "mel"

    def __init__(
        self,
        p: float,
        snr_db: tuple[float, float],
        colour: str,
        n_mel_channels: int,
        target_sample_rate: int,
    ) -> None:
        super().__init__(p)
        low, high = float(snr_db[0]), float(snr_db[1])
        if low > high:
            raise ValueError(f"noise: snr_db must be [min, max], got {snr_db}")
        self.snr_db = (low, high)
        self.colour = colour
        self.profile = colour_profile(colour, n_mel_channels, target_sample_rate)

    def sample_params(self) -> dict[str, Any]:
        return {"snr_db": uniform(*self.snr_db)}

    def apply_with(
        self,
        x: torch.Tensor,
        params: dict[str, Any],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        linear = x.exp()
        if mask is None:
            power = linear.pow(2).mean()
        else:
            # Per-sample power over valid frames only, so one short utterance in a batch of long
            # ones still lands on the requested SNR.
            valid = mask.expand_as(linear).to(linear.dtype)
            dims = (-2, -1)
            power = (linear.pow(2) * valid).sum(dim=dims, keepdim=True) / valid.sum(dim=dims, keepdim=True).clamp(min=1)

        sigma = (power / 10.0 ** (params["snr_db"] / 10.0)).sqrt()
        noise = sigma * self.profile.to(device=x.device, dtype=x.dtype) * torch.randn_like(linear).abs()
        out = (linear + noise).clamp(min=1e-5).log()
        return out if mask is None else torch.where(mask, out, x)

    def describe(self) -> str:
        return f"noise(p={self.p}, snr_db={list(self.snr_db)}, colour={self.colour})"

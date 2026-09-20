from __future__ import annotations

import torch

from f5_tts.model.augmentations.base import Augmentation, draws


def _valid_frame_mask(mel: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """`[b, 1, T]` boolean mask of real frames, for a `[b, d, T]` padded batch."""
    frames = torch.arange(mel.shape[-1], device=mel.device)
    return (frames.unsqueeze(0) < lengths.to(mel.device).unsqueeze(-1)).unsqueeze(1)


class BatchAugmentor:
    """Applies the mel operators once per batch, with a single parameter draw shared by every sample.

    Holds only the mel operators - the RIR bank stays with the dataset so it is not pickled into
    every worker a second time. Sharing one draw across both halves is what makes `apply_to: batch`
    free of the retain/forget seam that independent per-sample draws introduce.
    """

    def __init__(self, mel_augs: list[Augmentation]) -> None:
        self.mel_augs = list(mel_augs)

    def __call__(
        self,
        mel_retain: torch.Tensor,
        lengths_retain: torch.Tensor,
        mel_forget: torch.Tensor,
        lengths_forget: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for augmentation in self.mel_augs:
            if not draws(augmentation.p):
                continue
            params = augmentation.sample_params()
            mel_retain = self._apply(augmentation, mel_retain, lengths_retain, params)
            mel_forget = self._apply(augmentation, mel_forget, lengths_forget, params)
        return mel_retain, mel_forget

    @staticmethod
    def _apply(
        augmentation: Augmentation,
        mel: torch.Tensor,
        lengths: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        if mel.numel() == 0:  # an empty forget half is normal without oversampling
            return mel
        return augmentation.apply_with(mel, params, mask=_valid_frame_mask(mel, lengths))

    def describe(self) -> str:
        return ", ".join(augmentation.describe() for augmentation in self.mel_augs)

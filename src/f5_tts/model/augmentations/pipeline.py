from __future__ import annotations

import copy

import torch

from f5_tts.model.augmentations.base import Augmentation
from f5_tts.model.augmentations.batch import BatchAugmentor

APPLY_TO_VALUES = ("forget", "retain", "both", "batch")

FORGET_LABEL = -1
RETAIN_LABEL = 1


class AugmentationPipeline:
    """The augmentation operators for one run, plus the rule for which samples they reach.

    `apply_to` selects eligibility *and* granularity: `forget` / `retain` / `both` draw per sample in
    the dataset's `__getitem__`, while `batch` draws once per batch in the collate function and
    applies the identical transform to every sample. Waveform operators always run per sample -
    there is no audio left by collate time.
    """

    def __init__(
        self,
        waveform_augs: list[Augmentation],
        mel_augs: list[Augmentation],
        apply_to: str,
    ) -> None:
        if apply_to not in APPLY_TO_VALUES:
            raise ValueError(f"augment.apply_to must be one of {APPLY_TO_VALUES}, got '{apply_to}'")
        self.waveform_augs = list(waveform_augs)
        self.mel_augs = list(mel_augs)
        self.apply_to = apply_to

    @property
    def is_empty(self) -> bool:
        return not self.waveform_augs and not self.mel_augs

    @property
    def batch_level(self) -> bool:
        return self.apply_to == "batch"

    def _eligible(self, unlearn_label: int | None) -> bool:
        if self.apply_to in ("both", "batch"):
            return True
        if self.apply_to == "forget":
            return unlearn_label == FORGET_LABEL
        return unlearn_label == RETAIN_LABEL

    def apply_waveform(self, audio: torch.Tensor, unlearn_label: int | None) -> torch.Tensor:
        if not self.waveform_augs or not self._eligible(unlearn_label):
            return audio
        for augmentation in self.waveform_augs:
            audio = augmentation(audio)
        return audio

    def apply_mel(self, mel: torch.Tensor, unlearn_label: int | None) -> torch.Tensor:
        if self.batch_level or not self.mel_augs or not self._eligible(unlearn_label):
            return mel
        for augmentation in self.mel_augs:
            mel = augmentation(mel)
        return mel

    def batch_augmentor(self) -> BatchAugmentor | None:
        """The collate-time augmentor, or None when the mel operators run per sample."""
        if not self.batch_level or not self.mel_augs:
            return None
        return BatchAugmentor(self.mel_augs)

    def describe(self) -> str:
        if self.is_empty:
            return "augmentation: none"
        parts = []
        if self.waveform_augs:
            stage = "per sample" if not self.batch_level else "per sample, forced"
            parts.append(f"waveform [{stage}]: " + ", ".join(a.describe() for a in self.waveform_augs))
        if self.mel_augs:
            stage = "per batch" if self.batch_level else "per sample"
            parts.append(f"mel [{stage}]: " + ", ".join(a.describe() for a in self.mel_augs))
        return f"augmentation (apply_to={self.apply_to}) " + " | ".join(parts)


def dataset_without_augmentation(dataset):
    """A shallow copy of `dataset` with augmentation switched off, or `dataset` if it has none.

    `persistent_workers=True` means workers hold forked copies, so flipping an attribute on the live
    dataset would never reach them; profiling needs a distinct object instead.
    """
    pipeline = getattr(dataset, "augment_pipeline", None)
    if pipeline is None or pipeline.is_empty:
        return dataset
    clean = copy.copy(dataset)
    clean.augment_pipeline = None
    return clean

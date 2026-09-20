from __future__ import annotations

from typing import Any, Mapping

from f5_tts.model.augmentations.gain import MelGain
from f5_tts.model.augmentations.noise import MelNoise
from f5_tts.model.augmentations.pipeline import APPLY_TO_VALUES, AugmentationPipeline
from f5_tts.model.augmentations.reverb import MelReverb
from f5_tts.model.augmentations.rir import WaveformRIR
from f5_tts.model.augmentations.rir_bank import build_rir_bank

MEL_AUGMENTATIONS = ("gain", "noise", "reverb")
WAVEFORM_AUGMENTATIONS = ("rir",)


def _plain(config: Any) -> Any:
    """`config` as plain Python containers, accepting either a dict or an OmegaConf node."""
    try:
        from omegaconf import OmegaConf
    except ImportError:  # pragma: no cover - omegaconf ships with hydra
        return config
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _section(config: Mapping, key: str) -> dict:
    section = config.get(key) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"augment.{key} must be a mapping, got {type(section).__name__}")
    return dict(section)


def _check_known(section: Mapping, allowed: tuple[str, ...], label: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown augmentation(s) under augment.{label}: {unknown}. Allowed: {list(allowed)}")


def _enabled(section: Mapping, name: str) -> dict | None:
    entry = section.get(name) or {}
    if not entry.get("use", False):
        return None
    entry = dict(entry)
    if float(entry.get("p", 0.0)) <= 0.0:
        print(f"F5-TTS WARNING: augmentation '{name}' has use=True but p=0; it will never fire.")
    return entry


def build_augmentation_pipeline(
    augment_cfg: Mapping | None,
    mel_spec_cfg: Mapping,
) -> AugmentationPipeline | None:
    """The augmentation pipeline described by `datasets.augment`, or None when it is off.

    Builds the RIR bank eagerly so pyroomacoustics runs in the parent process, never in a worker.
    """
    augment_cfg = _plain(augment_cfg)
    if not augment_cfg or not augment_cfg.get("use", False):
        return None

    mel_spec_cfg = _plain(mel_spec_cfg)
    sample_rate = int(mel_spec_cfg["target_sample_rate"])
    n_mel_channels = int(mel_spec_cfg["n_mel_channels"])
    hop_length = int(mel_spec_cfg["hop_length"])

    apply_to = augment_cfg.get("apply_to", "both")
    if apply_to not in APPLY_TO_VALUES:
        raise ValueError(f"augment.apply_to must be one of {list(APPLY_TO_VALUES)}, got '{apply_to}'")

    mel_cfg = _section(augment_cfg, "mel")
    waveform_cfg = _section(augment_cfg, "waveform")
    _check_known(mel_cfg, MEL_AUGMENTATIONS, "mel")
    _check_known(waveform_cfg, WAVEFORM_AUGMENTATIONS, "waveform")

    mel_augs = []
    if (entry := _enabled(mel_cfg, "gain")) is not None:
        mel_augs.append(MelGain(p=entry["p"], gain_db=entry["gain_db"]))
    if (entry := _enabled(mel_cfg, "noise")) is not None:
        mel_augs.append(
            MelNoise(
                p=entry["p"],
                snr_db=entry["snr_db"],
                colour=entry.get("colour", "white"),
                n_mel_channels=n_mel_channels,
                target_sample_rate=sample_rate,
            )
        )
    if (entry := _enabled(mel_cfg, "reverb")) is not None:
        mel_augs.append(
            MelReverb(
                p=entry["p"],
                rt60_s=entry["rt60_s"],
                wet_level=entry["wet_level"],
                hop_length=hop_length,
                target_sample_rate=sample_rate,
            )
        )

    waveform_augs = []
    if (entry := _enabled(waveform_cfg, "rir")) is not None:
        bank = build_rir_bank(
            bank_size=int(entry["bank_size"]),
            bank_path=entry.get("bank_path"),
            sample_rate=sample_rate,
            rt60_s=entry["rt60_s"],
            room_dim_m=entry["room_dim_m"],
            max_order_cap=int(entry["max_order_cap"]),
            seed=int(entry.get("bank_seed", 0)),
        )
        waveform_augs.append(WaveformRIR(p=entry["p"], bank=bank))

    pipeline = AugmentationPipeline(waveform_augs, mel_augs, apply_to)
    if pipeline.is_empty:
        print("F5-TTS WARNING: datasets.augment.use=True but no augmentation is enabled; disabling augmentation.")
        return None

    if pipeline.batch_level and pipeline.waveform_augs:
        print(
            "F5-TTS WARNING: apply_to='batch' shares one draw across the batch for the mel operators, but "
            f"{[a.name for a in pipeline.waveform_augs]} can only run per sample, so batches stay partially "
            "inconsistent."
        )
    return pipeline

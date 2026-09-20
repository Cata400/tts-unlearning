from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np
import torch

CACHE_FORMAT_VERSION = 1


def _import_pyroomacoustics():
    try:
        import pyroomacoustics as pra
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "augment.waveform.rir requires pyroomacoustics. Install it with `pip install pyroomacoustics`, "
            "or set datasets.augment.waveform.rir.use=False."
        ) from error
    return pra


def _cache_metadata(
    *,
    bank_size: int,
    sample_rate: int,
    rt60_s: Sequence[float],
    room_dim_m: Sequence[Sequence[float]],
    max_order_cap: int,
    seed: int,
) -> str:
    return json.dumps(
        {
            "version": CACHE_FORMAT_VERSION,
            "bank_size": int(bank_size),
            "sample_rate": int(sample_rate),
            "rt60_s": [float(v) for v in rt60_s],
            "room_dim_m": [[float(v) for v in corner] for corner in room_dim_m],
            "max_order_cap": int(max_order_cap),
            "seed": int(seed),
        },
        sort_keys=True,
    )


def _load_cached_bank(path: str, metadata: str) -> list[torch.Tensor] | None:
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as payload:
            if str(payload["metadata"]) != metadata:
                print(f"[augment:rir] cached bank at {path} was built with different settings; regenerating.")
                return None
            count = int(payload["count"])
            return [torch.from_numpy(payload[f"rir_{index}"].astype(np.float32)) for index in range(count)]
    except Exception as error:  # a truncated or half-written cache must not abort training
        print(f"[augment:rir] could not read cached bank at {path} ({error}); regenerating.")
        return None


def _write_cached_bank(path: str, metadata: str, bank: list[torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {f"rir_{index}": rir.numpy() for index, rir in enumerate(bank)}
    payload["metadata"] = np.array(metadata)
    payload["count"] = np.array(len(bank))
    # Another rank may be writing the same bank; both produce identical content, so let rename race.
    tmp_path = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp_path, "wb") as handle:
            np.savez(handle, **payload)
        os.replace(tmp_path, path)
    except Exception as error:
        print(f"[augment:rir] could not cache the bank to {path} ({error}); continuing in memory.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _synthesise_bank(
    *,
    bank_size: int,
    sample_rate: int,
    rt60_s: Sequence[float],
    room_dim_m: Sequence[Sequence[float]],
    max_order_cap: int,
    seed: int,
    wall_margin_m: float,
    min_source_mic_distance_m: float,
    max_attempts: int,
) -> list[torch.Tensor]:
    pra = _import_pyroomacoustics()
    if hasattr(pra, "random") and hasattr(pra.random, "seed"):
        pra.random.seed(seed)  # the randomised image method draws from pyroomacoustics' own RNG

    rng = np.random.default_rng(seed)
    low_dim = np.asarray(room_dim_m[0], dtype=np.float64)
    high_dim = np.asarray(room_dim_m[1], dtype=np.float64)

    print(f"[augment:rir] synthesising {bank_size} shoebox RIRs at {sample_rate} Hz (one-off, then cached)")
    bank: list[torch.Tensor] = []
    for index in range(bank_size):
        for _ in range(max_attempts):
            dims = rng.uniform(low_dim, high_dim)
            rt60 = float(rng.uniform(float(rt60_s[0]), float(rt60_s[1])))
            try:
                absorption, max_order = pra.inverse_sabine(rt60, dims)
            except ValueError:
                continue  # this RT60 is unreachable in this room; draw another room

            room = pra.ShoeBox(
                dims,
                fs=sample_rate,
                materials=pra.Material(absorption),
                max_order=int(min(max_order, max_order_cap)),
                use_rand_ism=True,
                max_rand_disp=0.05,
            )
            source = rng.uniform(wall_margin_m, dims - wall_margin_m)
            mic = rng.uniform(wall_margin_m, dims - wall_margin_m)
            if np.linalg.norm(source - mic) < min_source_mic_distance_m:
                continue

            room.add_source(source)
            room.add_microphone(mic)
            room.compute_rir()
            rir = np.asarray(room.rir[0][0], dtype=np.float32)
            peak = float(np.abs(rir).max()) if rir.size else 0.0
            if rir.size < 2 or not np.isfinite(rir).all() or peak <= 0.0:
                continue

            bank.append(torch.from_numpy(rir / peak))
            break
        else:
            raise RuntimeError(
                f"[augment:rir] could not synthesise RIR {index} in {max_attempts} attempts. "
                "Widen augment.waveform.rir.room_dim_m or rt60_s."
            )

    return bank


def build_rir_bank(
    *,
    bank_size: int,
    bank_path: str | None,
    sample_rate: int,
    rt60_s: Sequence[float],
    room_dim_m: Sequence[Sequence[float]],
    max_order_cap: int,
    seed: int,
    wall_margin_m: float = 0.5,
    min_source_mic_distance_m: float = 0.5,
    max_attempts: int = 50,
) -> list[torch.Tensor]:
    """A bank of peak-normalised shoebox RIRs, loaded from `bank_path` or synthesised and cached there.

    The image-source method costs far too much to run per sample, so the bank is built once and
    sampled from at training time. Call this from the parent process: pyroomacoustics must never run
    inside a dataloader worker.
    """
    low_dim = np.asarray(room_dim_m[0], dtype=np.float64)
    high_dim = np.asarray(room_dim_m[1], dtype=np.float64)
    if low_dim.shape != (3,) or high_dim.shape != (3,):
        raise ValueError(f"rir: room_dim_m must be two 3-element corners, got {room_dim_m}")
    if np.any(low_dim > high_dim):
        raise ValueError(f"rir: room_dim_m must be [min_corner, max_corner], got {room_dim_m}")
    if np.any(low_dim <= 2.0 * wall_margin_m):
        raise ValueError(
            f"rir: every room dimension must exceed 2 * wall_margin_m ({2.0 * wall_margin_m} m), got {room_dim_m}"
        )

    metadata = _cache_metadata(
        bank_size=bank_size,
        sample_rate=sample_rate,
        rt60_s=rt60_s,
        room_dim_m=room_dim_m,
        max_order_cap=max_order_cap,
        seed=seed,
    )

    if bank_path:
        cached = _load_cached_bank(bank_path, metadata)
        if cached is not None:
            print(f"[augment:rir] loaded {len(cached)} cached RIRs from {bank_path}")
            return cached

    bank = _synthesise_bank(
        bank_size=bank_size,
        sample_rate=sample_rate,
        rt60_s=rt60_s,
        room_dim_m=room_dim_m,
        max_order_cap=max_order_cap,
        seed=seed,
        wall_margin_m=wall_margin_m,
        min_source_mic_distance_m=min_source_mic_distance_m,
        max_attempts=max_attempts,
    )

    if bank_path:
        _write_cached_bank(bank_path, metadata, bank)
        print(f"[augment:rir] cached {len(bank)} RIRs to {bank_path}")

    return bank

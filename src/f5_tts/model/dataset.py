import json
from importlib.resources import files
from typing import List

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }


class CustomUnlearningDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        forget_speakers: List[int],
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.forget_speakers = forget_speakers
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        print(f"Speakers to forget: {self.forget_speakers}")

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]
            speaker_id = row.get("speaker_id", None)

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            if speaker_id is None:
                raise ValueError("Speaker ID not found in the dataset for unlearning.")

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
            "unlearn_label": -1 if speaker_id in self.forget_speakers else 1,
        }


def build_unlearning_sample_weights(dataset, class_weights):
    weights = []

    for i in range(len(dataset.data)):
        row = dataset.data[i]
        speaker_id = row.get("speaker_id", None)

        if speaker_id is not None and speaker_id in dataset.forget_speakers:
            weights.append(class_weights[-1])
        else:
            weights.append(class_weights[1])

    return torch.DoubleTensor(weights)


def get_dataset_num_speakers(dataset):
    speaker_ids = set()

    for i in range(len(dataset.data)):
        row = dataset.data[i]
        speaker_id = row.get("speaker_id", None)

        if speaker_id is not None:
            speaker_ids.add(speaker_id)

    return len(speaker_ids)


def limit_indices_by_total_frames(
    dataset: Dataset,
    indices: list[int],
    max_total_frames: float,
    random_seed: int | None = None,
) -> list[int]:
    if max_total_frames <= 0:
        raise ValueError(f"max_total_frames must be positive, but received {max_total_frames}")

    if not hasattr(dataset, "get_frame_len"):
        raise ValueError("Duration-based retain limiting requires the dataset to implement `get_frame_len`.")

    if len(indices) == 0:
        return []

    generator = torch.Generator()
    if random_seed is not None:
        generator.manual_seed(random_seed)
        shuffled_positions = torch.randperm(len(indices), generator=generator).tolist()
    else:
        shuffled_positions = torch.randperm(len(indices)).tolist()

    limited_indices = []
    total_frames = 0.0
    for pos in shuffled_positions:
        idx = indices[pos]
        frame_len = dataset.get_frame_len(idx)

        if total_frames + frame_len <= max_total_frames or len(limited_indices) == 0:
            limited_indices.append(idx)
            total_frames += frame_len

    if len(limited_indices) == 0:
        raise ValueError("Duration-based retain limiting removed all retain samples.")

    return limited_indices


class BalancedUnlearningSampleBatchSampler(Sampler[list[int]]):
    """Batch sampler with a fixed 50/50 retain/forget composition per batch.

    Retain samples are iterated once per epoch (shuffled).
    If oversample_forget is True, forget samples are sampled with replacement, so they can appear multiple times per epoch.
    If oversample_forget is False, when forget samples are exhausted after one pass through, remaining batches will contain only retain samples.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        random_seed: int | None = None,
        oversample_forget: bool = False,
        retain_duration_budget_ratio: float | None = None,
    ):
        if batch_size % 2 != 0:
            raise ValueError(f"balanced_unlearn_sample requires an even batch_size_per_gpu, but received {batch_size}")
        if not hasattr(dataset, "data") or not hasattr(dataset, "forget_speakers"):
            raise ValueError("balanced_unlearn_sample requires a dataset with `data` and `forget_speakers` attributes.")
        if retain_duration_budget_ratio is not None and retain_duration_budget_ratio <= 0:
            raise ValueError(
                "retain_duration_budget_ratio must be positive when provided, "
                f"but received {retain_duration_budget_ratio}"
            )
        if retain_duration_budget_ratio is not None and oversample_forget:
            raise ValueError("retain_duration_budget_ratio is not compatible with oversample_forget=True")

        forget_speakers = set(dataset.forget_speakers)
        self.oversample_forget = oversample_forget
        self.forget_indices = []
        self.retain_indices = []
        for idx in range(len(dataset.data)):
            speaker_id = dataset.data[idx].get("speaker_id", None)
            if speaker_id in forget_speakers:
                self.forget_indices.append(idx)
            else:
                self.retain_indices.append(idx)

        if len(self.forget_indices) == 0:
            raise ValueError("balanced_unlearn_sample requires at least one forget sample.")
        if len(self.retain_indices) == 0:
            raise ValueError("balanced_unlearn_sample requires at least one retain sample.")

        if retain_duration_budget_ratio is not None:
            forget_total_frames = sum(dataset.get_frame_len(idx) for idx in self.forget_indices)
            retain_total_frame_budget = forget_total_frames * retain_duration_budget_ratio
            self.retain_indices = limit_indices_by_total_frames(
                dataset,
                self.retain_indices,
                max_total_frames=retain_total_frame_budget,
                random_seed=random_seed,
            )

        self.half_batch_size = batch_size // 2
        self.random_seed = random_seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.oversample_forget:
            generator = torch.Generator()
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed + self.epoch)
                retain_perm = torch.randperm(len(self.retain_indices), generator=generator).tolist()
            else:
                retain_perm = torch.randperm(len(self.retain_indices)).tolist()
            shuffled_retain = [self.retain_indices[i] for i in retain_perm]

            num_batches = len(self)
            for batch_idx in range(num_batches):
                start = batch_idx * self.half_batch_size
                retain_batch = shuffled_retain[start : start + self.half_batch_size]

                if self.random_seed is not None:
                    forget_positions = torch.randint(
                        low=0,
                        high=len(self.forget_indices),
                        size=(self.half_batch_size,),
                        generator=generator,
                    ).tolist()
                else:
                    forget_positions = torch.randint(
                        low=0,
                        high=len(self.forget_indices),
                        size=(self.half_batch_size,),
                    ).tolist()
                forget_batch = [self.forget_indices[i] for i in forget_positions]

                batch = retain_batch + forget_batch
                if self.random_seed is not None:
                    batch_perm = torch.randperm(len(batch), generator=generator).tolist()
                else:
                    batch_perm = torch.randperm(len(batch)).tolist()
                yield [batch[i] for i in batch_perm]
        else:
            generator = torch.Generator()
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed + self.epoch)
                retain_perm = torch.randperm(len(self.retain_indices), generator=generator).tolist()
                forget_perm = torch.randperm(len(self.forget_indices), generator=generator).tolist()
            else:
                retain_perm = torch.randperm(len(self.retain_indices)).tolist()
                forget_perm = torch.randperm(len(self.forget_indices)).tolist()

            shuffled_retain = [self.retain_indices[i] for i in retain_perm]
            shuffled_forget = [self.forget_indices[i] for i in forget_perm]
            # shuffled_retain = shuffled_retain[:len(shuffled_forget)]

            num_forget_batches = (len(self.forget_indices) + self.half_batch_size - 1) // self.half_batch_size
            num_retain_batches = (len(self.retain_indices) + self.half_batch_size - 1) // self.half_batch_size
            for batch_idx in range(num_forget_batches):
                start = batch_idx * self.half_batch_size
                forget_batch = shuffled_forget[start : start + self.half_batch_size]
                retain_batch = shuffled_retain[start : start + self.half_batch_size]

                batch = retain_batch + forget_batch
                if self.random_seed is not None:
                    batch_perm = torch.randperm(len(batch), generator=generator).tolist()
                else:
                    batch_perm = torch.randperm(len(batch)).tolist()
                yield [batch[i] for i in batch_perm]

            # After all forget samples are used, yield remaining retain samples in batches
            for batch_idx in range(num_forget_batches, num_retain_batches):
                start = batch_idx * self.half_batch_size
                retain_batch = shuffled_retain[start : start + self.half_batch_size]

                if len(retain_batch) == 0:
                    continue

                if self.random_seed is not None:
                    batch_perm = torch.randperm(len(retain_batch), generator=generator).tolist()
                else:
                    batch_perm = torch.randperm(len(retain_batch)).tolist()
                yield [retain_batch[i] for i in batch_perm]

    def __len__(self):
        # Keep strict 50/50 composition by only emitting full half-batches.
        if self.oversample_forget:
            return len(self.retain_indices) // self.half_batch_size
        else:
            return (len(self.retain_indices) + self.half_batch_size - 1) // self.half_batch_size


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Dynamic Unlearning Batch Sampler
class DynamicUnlearningBatchSampler(Sampler[list[int]]):
    """Dynamic batch sampler that prioritizes forget samples first.

    The first batches are guaranteed to contain at least one forget sample
    (unlearn_label == -1) until all forget samples are used. Remaining batches
    contain only retain samples (unlearn_label == 1).
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        data_source = self.sampler.data_source
        if not hasattr(data_source, "data") or not hasattr(data_source, "forget_speakers"):
            raise ValueError("DynamicUnlearningB1600atchSampler requires a dataset with data and forget_speakers.")

        forget_speakers = set(data_source.forget_speakers)
        forget_indices = []
        retain_indices = []

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            frame_len = data_source.get_frame_len(idx)
            row = data_source.data[idx]
            speaker_id = row.get("speaker_id", None)
            if speaker_id in forget_speakers:
                forget_indices.append((idx, frame_len))
            else:
                retain_indices.append((idx, frame_len))

        forget_indices.sort(key=lambda elem: elem[1])
        retain_indices.sort(key=lambda elem: elem[1])

        forget_batches = []
        retain_batches = []

        retain_pos = 0
        for idx, frame_len in tqdm(
            forget_indices, desc=f"Creating unlearning batches with {frames_threshold} audio frames per gpu"
        ):
            if frame_len > self.frames_threshold:
                continue

            batch = [idx]
            batch_frames = frame_len

            while retain_pos < len(retain_indices):
                retain_idx, retain_frame_len = retain_indices[retain_pos]
                if batch_frames + retain_frame_len <= self.frames_threshold and (
                    max_samples == 0 or len(batch) < max_samples
                ):
                    batch.append(retain_idx)
                    batch_frames += retain_frame_len
                    retain_pos += 1
                else:
                    break

            if len(batch) > 0:
                forget_batches.append(batch)

        # # Build remaining retain-only batches
        # batch = []
        # batch_frames = 0
        # for idx, frame_len in retain_indices[retain_pos:]:
        #     if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
        #         batch.append(idx)
        #         batch_frames += frame_len
        #     else:
        #         if len(batch) > 0:
        #             retain_batches.append(batch)
        #         if frame_len <= self.frames_threshold:
        #             batch = [idx]
        #             batch_frames = frame_len
        #         else:
        #             batch = []
        #             batch_frames = 0

        if not drop_residual and len(batch) > 0:
            retain_batches.append(batch)

        self.forget_batches = forget_batches
        self.retain_batches = retain_batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            forget_order = torch.randperm(len(self.forget_batches), generator=g).tolist()
            retain_order = torch.randperm(len(self.retain_batches), generator=g).tolist()
            batches = [self.forget_batches[i] for i in forget_order] + [self.retain_batches[i] for i in retain_order]
        else:
            batches = self.forget_batches + self.retain_batches
        return iter(batches)

    def __len__(self):
        return len(self.forget_batches) + len(self.retain_batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    forget_speakers: List[int] | None = None,
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset" or dataset_type == "CustomUnlearningDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]

        if dataset_type == "CustomDataset":
            train_dataset = CustomDataset(
                train_dataset,
                durations=durations,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )
        elif dataset_type == "CustomUnlearningDataset":
            train_dataset = CustomUnlearningDataset(
                train_dataset,
                forget_speakers,
                durations=durations,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,  # records for padding mask
        text=text,
        text_lengths=text_lengths,
    )


def collate_fn_unlearning(batch):
    mel_specs_retain = [item["mel_spec"].squeeze(0) for item in batch if item["unlearn_label"] == 1]
    mel_lengths_retain = torch.LongTensor([spec.shape[-1] for spec in mel_specs_retain])
    max_mel_length_retain = mel_lengths_retain.amax() if mel_lengths_retain.numel() > 0 else 0

    padded_mel_specs_retain = []
    for spec in mel_specs_retain:
        padding = (0, max_mel_length_retain - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs_retain.append(padded_spec)

    mel_specs_retain = (
        torch.stack(padded_mel_specs_retain) if len(padded_mel_specs_retain) > 0 else torch.empty((0, 0, 0))
    )

    text_retain = [item["text"] for item in batch if item["unlearn_label"] == 1]
    text_lengths_retain = torch.LongTensor([len(item) for item in text_retain])

    mel_specs_forget = [item["mel_spec"].squeeze(0) for item in batch if item["unlearn_label"] == -1]
    mel_lengths_forget = torch.LongTensor([spec.shape[-1] for spec in mel_specs_forget])
    max_mel_length_forget = mel_lengths_forget.amax() if mel_lengths_forget.numel() > 0 else 0

    padded_mel_specs_forget = []
    for spec in mel_specs_forget:
        padding = (0, max_mel_length_forget - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs_forget.append(padded_spec)

    mel_specs_forget = (
        torch.stack(padded_mel_specs_forget) if len(padded_mel_specs_forget) > 0 else torch.empty((0, 0, 0))
    )

    text_forget = [item["text"] for item in batch if item["unlearn_label"] == -1]
    text_lengths_forget = torch.LongTensor([len(item) for item in text_forget])

    unlearn_labels = torch.LongTensor([item["unlearn_label"] for item in batch])

    return dict(
        mel_retain=mel_specs_retain,
        mel_lengths_retain=mel_lengths_retain,
        text_retain=text_retain,
        text_lengths_retain=text_lengths_retain,
        mel_forget=mel_specs_forget,
        mel_lengths_forget=mel_lengths_forget,
        text_forget=text_forget,
        text_lengths_forget=text_lengths_forget,
        unlearn_labels=unlearn_labels,
    )

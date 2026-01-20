import os
import shutil
import sys

sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    vocab_set = set()
    speaker_ids_set = set()
    audio_lists = list(audio_dir.rglob("*.wav"))

    for line in audio_lists:
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        speaker_id = int(audio_dir.parts[-1])  # This assumes the structure .../root/<speaker_id>/...
        if duration < 0.4 or duration > 30:
            continue
        sub_result.append({"audio_path": str(line), "text": text, "duration": duration, "speaker_id": speaker_id})
        durations.append(duration)
        vocab_set.update(list(text))
        speaker_ids_set.add(speaker_id)

    return sub_result, durations, vocab_set, speaker_ids_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    speaker_id_set = set()

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set, sub_speaker_ids_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
        speaker_id_set.update(sub_speaker_ids_set)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # save speaker_id list
    with open(f"{save_dir}/speaker_ids.txt", "w") as f:
        for speaker_id in sorted(speaker_id_set):
            f.write(str(speaker_id) + "\n")

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab_orig.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    print(f"For {dataset_name}, speaker count: {len(speaker_id_set)}")

    pretrained_model_vocab = str(files("f5_tts").joinpath("../../")) + "/data/Emilia_ZH_EN_pinyin/vocab.txt"
    shutil.copy2(pretrained_model_vocab, f"{save_dir}/vocab.txt")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "pinyin"  # "pinyin" | "char"

    # SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    SUB_SET = ["train-clean-100_train_intra_speaker_split_0.2"]
    dataset_dir = "/home/catalin/Desktop/Datasets/LibriTTS"
    dataset_name = f"LibriTTS_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")

    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours

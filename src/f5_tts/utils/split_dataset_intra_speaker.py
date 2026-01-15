import os
from pathlib import Path
import shutil
import argparse
import random
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into intra-speaker subsets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to split")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input dataset directory")
    parser.add_argument("--split", type=float, default=0.2, help="Fraction of data to use for validation set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling data")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    input_dir = args.input_dir
    split = args.split
    seed = args.seed
    random.seed(seed)

    if dataset_name.lower() == 'libritts':
        if input_dir.endswith(os.path.sep):
            input_dir = input_dir[:-1]
        
        input_dir_prefix = os.path.sep.join(input_dir.split(os.path.sep)[:-1])
        input_dir_basename = input_dir.split(os.path.sep)[-1]
        output_train_dir = os.path.join(input_dir_prefix, f"{input_dir_basename}_train_intra_speaker_split_{split}")
        output_val_dir = os.path.join(input_dir_prefix, f"{input_dir_basename}_val_intra_speaker_split_{split}")

        os.makedirs(output_train_dir, exist_ok=True)
        os.makedirs(output_val_dir, exist_ok=True)

        for speaker in tqdm(sorted(os.listdir(input_dir))):
            os.makedirs(os.path.join(output_train_dir, speaker), exist_ok=True)
            os.makedirs(os.path.join(output_val_dir, speaker), exist_ok=True)
            
            for recording in sorted(os.listdir(os.path.join(input_dir, speaker))):
                audio_files = glob(os.path.join(input_dir, speaker, recording, "*.wav"))
                audio_files_names = [Path(f).stem for f in audio_files]
                random.shuffle(audio_files)
                
                # Split and copy audio and text files
                for i, audio_file in enumerate(audio_files_names):
                    split_set = "train" if i < int(len(audio_files) * (1 - split)) else "val"
                    
                    shutil.copy2(
                        os.path.join(input_dir, speaker, recording, f"{audio_file}.wav"),
                        os.path.join(output_train_dir if split_set == "train" else output_val_dir, speaker, f"{audio_file}.wav"),
                    )
                    shutil.copy2(
                        os.path.join(input_dir, speaker, recording, f"{audio_file}.normalized.txt"),
                        os.path.join(output_train_dir if split_set == "train" else output_val_dir, speaker, f"{audio_file}.normalized.txt"),
                    )
                    shutil.copy2(
                        os.path.join(input_dir, speaker, recording, f"{audio_file}.normalized.txt"),
                        os.path.join(output_train_dir if split_set == "train" else output_val_dir, speaker, f"{audio_file}.original.txt"),
                    )
                    
                # Copy for both splits all other file types besides .wav and .txt
                other_files = [f for f in os.listdir(os.path.join(input_dir, speaker, recording)) if not f.endswith(('.wav', '.txt'))]
                for other_file in other_files:
                    shutil.copy2(
                        os.path.join(input_dir, speaker, recording, other_file),
                        os.path.join(output_train_dir, speaker, other_file),
                    )
                    shutil.copy2(
                        os.path.join(input_dir, speaker, recording, other_file),
                        os.path.join(output_val_dir, speaker, other_file),
                    )

    else:
        raise ValueError(f"Dataset {dataset_name} not supported for intra-speaker splitting.")

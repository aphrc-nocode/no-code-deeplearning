"""
Utility functions for speech recognition pipeline.
Contains functions for dataset splitting, duration calculation, text cleaning, and CSV-based dataset loading.
"""
import math
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import os

import datasets
import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, DatasetDict, Audio


def format_duration(total_seconds: Optional[float]) -> str:
    """Formats a duration in seconds into a human-readable string (Hh Mm Ss.s)."""
    if total_seconds is None or not isinstance(total_seconds, (int, float)) or total_seconds < 0:
        return "N/A"
    total_seconds = float(total_seconds)
    if math.isclose(total_seconds, 0):
        return "0.0s"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 1e-6 or not parts:
        parts.append(f"{seconds:.1f}s")
    return " ".join(parts) if parts else "0.0s"


def random_split(dataset, train_ratio, dev_ratio, random_seed=None):
    """Randomly splits a dataset into train, dev, and test sets."""
    dataset = dataset["train"]
    indices = list(range(len(dataset)))

    if random_seed is not None:
        random.seed(random_seed)

    random.shuffle(indices)

    num_items = len(indices)
    num_train = math.floor(num_items * train_ratio)
    num_dev = math.floor(num_items * dev_ratio)

    # Divide the shuffled items into sets
    train = dataset.select(indices[:num_train])
    dev = dataset.select(indices[num_train : num_train + num_dev])
    test = dataset.select(indices[num_train + num_dev :])

    return train, dev, test


def add_duration_column(
    dataset: Union[Dataset, DatasetDict],
    audio_column: str = "audio",
    duration_column: str = "duration",
    num_proc: Optional[int] = None,
) -> Union[Dataset, DatasetDict]:
    """Adds a duration column (in seconds) to a Hugging Face speech dataset."""

    def _calculate_duration_batch(batch):
        audio_data_list = batch[audio_column]
        durations = []
        for audio_data in audio_data_list:
            if (
                audio_data is not None
                and isinstance(audio_data, dict)
                and "array" in audio_data
                and "sampling_rate" in audio_data
                and isinstance(audio_data["array"], (np.ndarray, list))
                and isinstance(audio_data["sampling_rate"], int)
                and audio_data["sampling_rate"] > 0
            ):
                duration = len(audio_data["array"]) / audio_data["sampling_rate"]
                durations.append(duration)
            else:
                durations.append(None)
        return {duration_column: durations}

    if isinstance(dataset, DatasetDict):
        processed_splits = {}
        for split_name, ds_split in dataset.items():
            print(f"Adding duration for split: {split_name}")
            if duration_column in ds_split.column_names:
                print(f"Duration column '{duration_column}' already exists in split '{split_name}'. Skipping.")
                processed_splits[split_name] = ds_split
                continue
            if len(ds_split) == 0:
                print(f"Split '{split_name}' is empty. Skipping duration calculation.")
                processed_splits[split_name] = ds_split
                continue
            processed_splits[split_name] = add_duration_column(ds_split, audio_column, duration_column, num_proc)
        return DatasetDict(processed_splits)

    elif isinstance(dataset, Dataset):
        if len(dataset) == 0:
            print("Dataset is empty. Skipping duration calculation.")
            return dataset
        if audio_column not in dataset.column_names:
            raise ValueError(f"Audio column '{audio_column}' not found in dataset columns: {dataset.column_names}")

        print(f"Adding '{duration_column}' column...")
        dataset_with_duration = dataset.map(
            _calculate_duration_batch, batched=True, num_proc=num_proc, desc=f"Calculating {duration_column}"
        )
        new_features = dataset_with_duration.features.copy()
        new_features[duration_column] = datasets.Value("float32")
        dataset_with_duration = dataset_with_duration.cast(new_features)
        print(f"Finished adding '{duration_column}' column.")
        return dataset_with_duration
    else:
        raise TypeError("Input must be a Hugging Face Dataset or DatasetDict.")


def find_problematic_audio_files(
    dataset: Dataset,
    audio_column: str = "audio",
    identifier_column: Optional[str] = "path",
    stop_on_first_error: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Identifies items causing errors during data loading/decoding."""
    problematic_items: List[Dict[str, Any]] = []
    if len(dataset) == 0:
        if verbose:
            print("Dataset is empty. Skipping audio file check.")
        return problematic_items

    num_examples = len(dataset)

    if verbose:
        print(f"Starting audio file check on {num_examples} examples...")
        iterator = tqdm.tqdm(range(num_examples), desc="Checking dataset items")
    else:
        iterator = range(num_examples)

    has_identifier_col = identifier_column and identifier_column in dataset.column_names

    for i in iterator:
        error_occurred = False
        error_info = {"index": i, "identifier": None, "error": None}

        try:
            _ = dataset[i]  # Accessing the item triggers decoding
        except (ValueError, TypeError, OSError, Exception) as e:
            error_occurred = True
            error_info["error"] = str(e)
            if verbose:
                print(f"\nCaught error ({type(e).__name__}) at index {i}: {e}")
        finally:
            if error_occurred:
                identifier = None
                if has_identifier_col:
                    try:
                        item_metadata = dataset.select([i], keep_in_memory=True).select_columns([identifier_column])[0]
                        identifier = item_metadata[identifier_column]
                        error_info["identifier"] = identifier
                        if verbose:
                            print(f"  Identifier ('{identifier_column}'): {identifier}")
                    except Exception as meta_e:
                        if verbose:
                            print(f"  Could not retrieve identifier for index {i}: {meta_e}")

                problematic_items.append(error_info)
                if stop_on_first_error:
                    if verbose:
                        print("Stopping check after first error.")
                    break
    if verbose:
        print("\n--- Audio File Check Complete ---")
        if problematic_items:
            print(f"Found {len(problematic_items)} items causing loading errors.")
        else:
            print("No data loading errors detected during check.")
    return problematic_items


def clean_transcript(text: str, punctuation_to_remove: str = r"[^\w\s']", lowercase: bool = True) -> str:
    """Basic transcript cleaning: NFKC normalization, lowercase, punctuation removal."""
    if not isinstance(text, str):
        text = ""
    text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f]", "'", text)
    text = re.sub(punctuation_to_remove, "", text)
    text = " ".join(text.split())
    return text


def split_hf_dataset(
    dataset: Dataset,
    speaker_id_col: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = None,
    duration_col: Optional[str] = None,
    random_seed: int = None,
    verbose: bool = True,
    num_proc: int = 1,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits a Hugging Face Dataset into train, development (dev), and test sets
    ensuring speaker disjointness.
    """
    if not isinstance(dataset, Dataset):
        raise TypeError("Input data must be a Hugging Face datasets.Dataset object.")
    if random_seed is not None:
        random.seed(random_seed)
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - dev_ratio
    if not math.isclose(train_ratio + dev_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to ~1.0 (sum: {train_ratio + dev_ratio + test_ratio})")
    if train_ratio < 0 or dev_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios cannot be negative.")
    if speaker_id_col not in dataset.column_names:
        raise ValueError(f"Speaker ID column '{speaker_id_col}' not found.")

    unique_speakers = dataset.unique(speaker_id_col)
    if not unique_speakers:
        raise ValueError(f"No unique speakers found in column '{speaker_id_col}'.")

    num_speakers = len(unique_speakers)
    if verbose:
        print(f"Found {num_speakers} unique speakers in column '{speaker_id_col}'.")
    if num_speakers < 3 and verbose:
        print(f"Warning: Very few speakers ({num_speakers}). Split ratios may be skewed.")

    random.shuffle(unique_speakers)

    n_train = math.floor(num_speakers * train_ratio)
    n_dev = math.floor(num_speakers * dev_ratio)
    n_test = num_speakers - n_train - n_dev

    # Adjust splits if needed
    if dev_ratio > 0 and n_dev == 0 and num_speakers > n_train:
        n_dev = 1
    if test_ratio > 0 and n_test == 0 and num_speakers > (n_train + n_dev):
        n_test = 1
        if n_train + n_dev + n_test > num_speakers:
            n_train = num_speakers - n_dev - n_test

    train_speaker_set = set(unique_speakers[:n_train])
    dev_speaker_set = set(unique_speakers[n_train : n_train + n_dev])
    test_speaker_set = set(unique_speakers[n_train + n_dev :])

    # Filter datasets
    train_data = dataset.filter(lambda ex: ex[speaker_id_col] in train_speaker_set, num_proc=num_proc)
    dev_data = dataset.filter(lambda ex: ex[speaker_id_col] in dev_speaker_set, num_proc=num_proc)
    test_data = dataset.filter(lambda ex: ex[speaker_id_col] in test_speaker_set, num_proc=num_proc)

    if verbose:
        print(f"Split results: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")

    return train_data, dev_data, test_data


def load_csv_speech_dataset(
    dataset_path: str,
    audio_column: str = "audio",
    text_column: str = "sentence",
    file_name_column: str = "file_name",
    csv_filename: str = "metadata.csv",
    target_sampling_rate: Optional[int] = 16000,
    split_names: Optional[List[str]] = None
) -> DatasetDict:
    """
    Load a speech dataset from CSV metadata files.
    
    Expects a structure like:
    dataset_path/
        train/
            metadata.csv  (columns: file_name, sentence, [other columns])
            audio1.wav
            audio2.wav
            ...
        validation/
            metadata.csv
            audio1.wav
            ...
        test/
            metadata.csv
            audio1.wav
            ...
    
    Args:
        dataset_path: Path to the root dataset directory
        audio_column: Name for the audio column in the resulting dataset
        text_column: Name for the text column in the resulting dataset  
        file_name_column: Name of the column in CSV containing audio filenames
        csv_filename: Name of the CSV metadata file (default: "metadata.csv")
        target_sampling_rate: Target sampling rate for audio (None to keep original)
        split_names: List of split names to look for (auto-detected if None)
    
    Returns:
        DatasetDict with loaded audio and text data
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Auto-detect splits if not provided
    if split_names is None:
        split_names = []
        for item in dataset_path.iterdir():
            if item.is_dir() and (item / csv_filename).exists():
                split_names.append(item.name)
        
        if not split_names:
            raise ValueError(f"No splits found with {csv_filename} files in {dataset_path}")
    
    datasets_dict = {}
    
    for split_name in split_names:
        split_path = dataset_path / split_name
        csv_path = split_path / csv_filename
        
        if not split_path.exists() or not csv_path.exists():
            print(f"Warning: Skipping split '{split_name}' - missing directory or {csv_filename}")
            continue
            
        print(f"Loading split: {split_name}")
        
        # Load CSV metadata
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
            continue
            
        if file_name_column not in df.columns:
            print(f"Warning: Column '{file_name_column}' not found in {csv_path}. Columns: {list(df.columns)}")
            continue
            
        # Convert to required column names and add audio paths
        dataset_data = {}
        
        # Add audio file paths
        audio_paths = []
        valid_rows = []
        
        for idx, row in df.iterrows():
            audio_file = split_path / row[file_name_column]
            if audio_file.exists():
                audio_paths.append(str(audio_file))
                valid_rows.append(idx)
            else:
                print(f"Warning: Audio file not found: {audio_file}")
        
        if not audio_paths:
            print(f"Warning: No valid audio files found for split '{split_name}'")
            continue
            
        # Filter to valid rows only
        df_valid = df.iloc[valid_rows].reset_index(drop=True)
        
        # Build dataset dictionary
        dataset_data["path"] = audio_paths
        
        # Add text column (try different common column names)
        text_found = False
        for potential_text_col in [text_column, "sentence", "text", "transcript", "transcription"]:
            if potential_text_col in df_valid.columns:
                dataset_data[text_column] = df_valid[potential_text_col].astype(str).tolist()
                text_found = True
                break
                
        if not text_found:
            print(f"Warning: No text column found in {csv_path}. Tried: {[text_column, 'sentence', 'text', 'transcript', 'transcription']}")
            continue
            
        # Add any other columns from CSV (excluding file_name which we already processed)
        for col in df_valid.columns:
            if col != file_name_column and col not in dataset_data:
                dataset_data[col] = df_valid[col].tolist()
        
        # Create Hugging Face dataset
        try:
            split_dataset = Dataset.from_dict(dataset_data)
            
            # Cast audio column to Audio type
            if target_sampling_rate:
                split_dataset = split_dataset.cast_column("path", Audio(sampling_rate=target_sampling_rate))
            else:
                split_dataset = split_dataset.cast_column("path", Audio())
                
            # Rename path column to audio_column
            if "path" != audio_column:
                split_dataset = split_dataset.rename_column("path", audio_column)
                
            datasets_dict[split_name] = split_dataset
            print(f"Loaded {len(split_dataset)} examples for split '{split_name}'")
            
        except Exception as e:
            print(f"Error creating dataset for split '{split_name}': {e}")
            continue
    
    if not datasets_dict:
        raise ValueError("No valid dataset splits were loaded")
        
    return DatasetDict(datasets_dict)


def detect_dataset_format(dataset_path: str) -> str:
    """
    Detect whether a dataset uses CSV metadata format or standard audiofolder format.
    
    Returns:
        "csv" if CSV metadata files are found
        "audiofolder" if standard Hugging Face audiofolder structure is detected
        "unknown" if format cannot be determined
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        return "unknown"
    
    # Check for CSV metadata files
    csv_files_found = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            csv_path = item / "metadata.csv"
            if csv_path.exists():
                csv_files_found.append(item.name)
    
    if csv_files_found:
        return "csv"
    
    # Check for standard audiofolder structure (audio files directly in split directories)
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    for item in dataset_path.iterdir():
        if item.is_dir():
            # Check if directory contains audio files
            for audio_file in item.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    return "audiofolder"
    
    return "unknown"

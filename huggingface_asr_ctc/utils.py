import math
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, DatasetDict


# --- Helper Function for Formatting Duration ---
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


# --- Main Splitting Function (Modified) ---
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

    # Adjust n_dev to be at least 1 if dev_ratio > 0 and there are enough speakers
    if dev_ratio > 0 and n_dev == 0 and num_speakers > n_train:
        n_dev = 1

    n_test = num_speakers - n_train - n_dev

    # Adjust n_test to be at least 1 if test_ratio > 0 and there are enough speakers
    if test_ratio > 0 and n_test == 0 and num_speakers > (n_train + n_dev):
        n_test = 1
        # Recalculate n_train if n_test was forced to 1 and n_dev was also calculated
        if n_train + n_dev + n_test > num_speakers:
            n_train = num_speakers - n_dev - n_test

    if verbose:
        p_train = n_train / num_speakers if num_speakers > 0 else 0.0
        p_dev = n_dev / num_speakers if num_speakers > 0 else 0.0
        p_test = n_test / num_speakers if num_speakers > 0 else 0.0
        print(f"Target speaker split: Train={n_train} ({p_train:.1%}), Dev={n_dev} ({p_dev:.1%}), Test={n_test} ({p_test:.1%})")
        if n_train <= 0 and train_ratio > 0:
            print(f"Warning: 0 speakers allocated for train set despite train_ratio={train_ratio}.")
        if n_dev <= 0 and dev_ratio > 0:
            print(f"Warning: 0 speakers allocated for dev set despite dev_ratio={dev_ratio}.")
        if n_test <= 0 and test_ratio > 0:
            print(f"Warning: 0 speakers allocated for test set despite test_ratio={test_ratio}.")

    train_speaker_set = set(unique_speakers[:n_train])
    dev_speaker_set = set(unique_speakers[n_train : n_train + n_dev])
    test_speaker_set = set(unique_speakers[n_train + n_dev :])

    assert train_speaker_set.isdisjoint(dev_speaker_set), "Train/Dev speaker overlap!"
    assert train_speaker_set.isdisjoint(test_speaker_set), "Train/Test speaker overlap!"
    assert dev_speaker_set.isdisjoint(test_speaker_set), "Dev/Test speaker overlap!"
    assert (
        len(train_speaker_set) + len(dev_speaker_set) + len(test_speaker_set)
    ) == num_speakers, "Speaker set sum mismatch!"

    if verbose:
        print(f"\nFiltering dataset for train split ({len(train_speaker_set)} speakers)...")
    train_data = dataset.filter(lambda ex: ex[speaker_id_col] in train_speaker_set, num_proc=num_proc)
    if verbose:
        print(f"Filtering dataset for dev split ({len(dev_speaker_set)} speakers)...")
    dev_data = dataset.filter(lambda ex: ex[speaker_id_col] in dev_speaker_set, num_proc=num_proc)
    if verbose:
        print(f"Filtering dataset for test split ({len(test_speaker_set)} speakers)...")
    test_data = dataset.filter(lambda ex: ex[speaker_id_col] in test_speaker_set, num_proc=num_proc)

    if verbose:
        total_items = len(dataset)
        print("\nActual split (data items):")
        train_item_perc = len(train_data) / total_items if total_items > 0 else 0.0
        dev_item_perc = len(dev_data) / total_items if total_items > 0 else 0.0
        test_item_perc = len(test_data) / total_items if total_items > 0 else 0.0
        print(f"  Train: {len(train_data):,} items ({train_item_perc:.1%})")
        print(f"  Dev:   {len(dev_data):,} items ({dev_item_perc:.1%})")
        print(f"  Test:  {len(test_data):,} items ({test_item_perc:.1%})")
        print("-" * 30)

        if duration_col and duration_col in dataset.column_names:
            print(f"Actual split (duration - based on '{duration_col}' column, assumes seconds):")
            try:
                total_duration = sum(d for d in dataset[duration_col] if d is not None)
                train_duration = sum(d for d in train_data[duration_col] if d is not None)
                dev_duration = sum(d for d in dev_data[duration_col] if d is not None)
                test_duration = sum(d for d in test_data[duration_col] if d is not None)

                train_dur_perc = train_duration / total_duration if total_duration > 1e-6 else 0.0
                dev_dur_perc = dev_duration / total_duration if total_duration > 1e-6 else 0.0
                test_dur_perc = test_duration / total_duration if total_duration > 1e-6 else 0.0

                print(f"  Total: {format_duration(total_duration)}")
                print(f"  Train: {format_duration(train_duration)} ({train_dur_perc:.1%})")
                print(f"  Dev:   {format_duration(dev_duration)} ({dev_dur_perc:.1%})")
                print(f"  Test:  {format_duration(test_duration)} ({test_dur_perc:.1%})")
            except (TypeError, KeyError) as e:
                print(
                    f"  Error: Could not calculate duration split. Column '{duration_col}' may contain non-numeric or missing data: {e}"
                )
            except Exception as e:
                print(f"  Error: Could not calculate duration split due to an unexpected error: {e}")
            print("-" * 30)
        elif duration_col:
            print(f"Duration column '{duration_col}' not found in dataset. Skipping duration split report.")

        try:
            train_spk_actual = set(train_data.unique(speaker_id_col)) if len(train_data) > 0 else set()
            dev_spk_actual = set(dev_data.unique(speaker_id_col)) if len(dev_data) > 0 else set()
            test_spk_actual = set(test_data.unique(speaker_id_col)) if len(test_data) > 0 else set()
            print("Verifying speaker disjointness in resulting datasets:")
            overlap_tr_dv = train_spk_actual.intersection(dev_spk_actual)
            overlap_dv_te = dev_spk_actual.intersection(test_spk_actual)
            overlap_tr_te = train_spk_actual.intersection(test_spk_actual)
            print(f"  Overlap train/dev: {len(overlap_tr_dv)} speakers")
            print(f"  Overlap dev/test:  {len(overlap_dv_te)} speakers")
            print(f"  Overlap train/test: {len(overlap_tr_te)} speakers")
            if not overlap_tr_dv and not overlap_dv_te and not overlap_tr_te:
                print("  Disjointness check passed.")
            else:
                print("  Warning: Speaker overlap detected between splits!")
            print("-" * 30)
        except Exception as e:
            print(f"Could not perform verification check due to error: {e}")
    return train_data, dev_data, test_data


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
            if len(ds_split) == 0:  # Skip empty datasets
                print(f"Split '{split_name}' is empty. Skipping duration calculation.")
                processed_splits[split_name] = ds_split
                continue
            processed_splits[split_name] = add_duration_column(ds_split, audio_column, duration_column, num_proc)
        return DatasetDict(processed_splits)

    elif isinstance(dataset, Dataset):
        if len(dataset) == 0:  # Skip empty datasets
            print("Dataset is empty. Skipping duration calculation.")
            return dataset
        if audio_column not in dataset.column_names:
            raise ValueError(f"Audio column '{audio_column}' not found in dataset columns: {dataset.column_names}")

        first_valid_item_audio = None
        for i in range(len(dataset)):
            try:
                item_audio = dataset[i][audio_column]
                if item_audio and "array" in item_audio and "sampling_rate" in item_audio:
                    first_valid_item_audio = item_audio
                    break
            except IndexError:
                break

        if first_valid_item_audio is None:
            print(
                f"Warning: No valid audio items found to check structure in column '{audio_column}'. Proceeding with caution."
            )
        elif not (
            isinstance(first_valid_item_audio, dict)
            and "array" in first_valid_item_audio
            and "sampling_rate" in first_valid_item_audio
        ):
            print(
                f"Warning: First valid item in column '{audio_column}' does not strictly match {{'array': ..., 'sampling_rate': ...}} structure: {first_valid_item_audio}. Proceeding, but errors may occur if structure varies."
            )

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

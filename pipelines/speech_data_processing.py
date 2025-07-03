"""
Data processing utilities for speech recognition pipeline.
Contains functions for dataset filtering, normalization, and CTC preparation.
"""
import math
import warnings
from typing import Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from pipelines.speech_utils import clean_transcript


def process_prepared_speech_dataset(
    input_dataset: Union[Dataset, DatasetDict],
    text_column: str,
    normalized_text_column: str,
    duration_column: str,
    min_duration_s: float,
    max_duration_s: float,
    min_transcript_len: int,
    max_transcript_len: int,
    outlier_std_devs: float,
    apply_outlier_filtering: bool,
    num_proc: int,
) -> Union[Dataset, DatasetDict]:
    """Performs text normalization and filtering on a speech dataset."""

    if isinstance(input_dataset, Dataset):
        raw_datasets = DatasetDict({"train": input_dataset})
        was_single_dataset = True
    elif isinstance(input_dataset, DatasetDict):
        raw_datasets = input_dataset
        was_single_dataset = False
    else:
        raise TypeError("input_dataset must be a datasets.Dataset or datasets.DatasetDict")

    original_counts = {split: len(ds) for split, ds in raw_datasets.items()}
    print(f"Input counts for processing: {original_counts}")

    def normalize_and_get_length(example):
        normalized = clean_transcript(example[text_column])
        example[normalized_text_column] = normalized
        example["transcript_len"] = len(normalized)
        return example

    def calculate_duration_length_ratio(example):
        if example["transcript_len"] > 0 and duration_column in example and example[duration_column] is not None:
            example["duration_len_ratio"] = example[duration_column] / example["transcript_len"]
        else:
            example["duration_len_ratio"] = float("nan")
        return example

    processed_datasets = {}
    for split, ds in raw_datasets.items():
        print(f"\n--- Processing split: {split} ---")
        if not ds or len(ds) == 0:
            print(f"Warning: Split '{split}' is empty or None. Skipping processing for this split.")
            processed_datasets[split] = Dataset.from_dict({}) if ds is None else ds
            continue

        if duration_column not in ds.column_names:
            raise ValueError(f"Duration column '{duration_column}' not found in dataset split '{split}'.")
        if text_column not in ds.column_names:
            raise ValueError(f"Text column '{text_column}' not found in dataset split '{split}'.")

        print(f"Filtering by duration ({min_duration_s}s - {max_duration_s}s) using '{duration_column}'...")
        ds = ds.filter(
            lambda x: x[duration_column] is not None and min_duration_s <= x[duration_column] <= max_duration_s,
            num_proc=num_proc,
        )
        print(f"Count after duration filtering: {len(ds)}")
        if len(ds) == 0:
            processed_datasets[split] = ds
            continue

        print("Normalizing transcripts and getting length...")
        ds = ds.map(normalize_and_get_length, num_proc=num_proc)

        print(f"Filtering by transcript length ({min_transcript_len} - {max_transcript_len} chars)...")
        ds = ds.filter(
            lambda x: min_transcript_len <= x["transcript_len"] <= max_transcript_len,
            num_proc=num_proc,
        )
        print(f"Count after transcript length filtering: {len(ds)}")
        if len(ds) == 0:
            processed_datasets[split] = ds
            continue

        if apply_outlier_filtering and len(ds) > 10:
            print(f"Calculating duration/length ratio using '{duration_column}' for outlier detection...")
            ds = ds.map(calculate_duration_length_ratio, num_proc=num_proc)

            ratios = [r for r in ds["duration_len_ratio"] if pd.notna(r) and not math.isinf(r)]
            if not ratios:
                print("Warning: No valid ratios found for outlier calculation. Skipping outlier filtering.")
            else:
                mean_ratio = np.mean(ratios)
                std_ratio = np.std(ratios)

                if std_ratio <= 1e-6:
                    print("Warning: Standard deviation of ratio is zero or very small. Skipping outlier filtering.")
                else:
                    min_r_thresh = mean_ratio - outlier_std_devs * std_ratio
                    max_r_thresh = mean_ratio + outlier_std_devs * std_ratio
                    print(
                        f"Filtering outliers (ratio mean={mean_ratio:.2f}, std={std_ratio:.2f}). Keeping: {min_r_thresh:.2f} - {max_r_thresh:.2f}"
                    )
                    ds = ds.filter(
                        lambda x: (
                            pd.notna(x["duration_len_ratio"])
                            and min_r_thresh <= x["duration_len_ratio"] <= max_r_thresh
                        ),
                        num_proc=num_proc,
                    )
                    print(f"Count after outlier filtering: {len(ds)}")
            if "duration_len_ratio" in ds.column_names:
                ds = ds.remove_columns(["duration_len_ratio"])
        elif apply_outlier_filtering:
            print("Skipping outlier filtering: Not enough data points (<10) or apply_outlier_filtering is False.")

        processed_datasets[split] = ds
        final_count = len(ds)
        original_count = original_counts[split]
        removed_count = original_count - final_count
        percent_removed = (removed_count / original_count * 100) if original_count > 0 else 0
        print(
            f"--- Finished split: {split} --- Final count: {final_count} (Removed {removed_count}, {percent_removed:.2f}%)"
        )

    final_datasets = DatasetDict(processed_datasets)
    print("\nDataset processing complete.")
    return final_datasets["train"] if was_single_dataset else final_datasets


def prepare_dataset_for_ctc(batch, processor, audio_col, normalized_text_col):
    """Prepares a single batch for CTC model training/evaluation."""
    audio = batch[audio_col]
    if audio is None or audio.get("array") is None or audio.get("sampling_rate") is None:
        batch["input_features"] = None
        batch["input_length"] = 0
        batch["labels"] = []
        return batch

    try:
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])
        text_to_process = batch[normalized_text_col] if isinstance(batch[normalized_text_col], str) else ""
        batch["labels"] = processor(text=text_to_process).input_ids
    except Exception as e:
        warnings.warn(
            f"Error processing batch for CTC: {e}. Audio path: {audio.get('path', 'N/A')}, Text: {batch.get(normalized_text_col, 'N/A')}"
        )
        batch["input_features"] = np.array([])
        batch["input_length"] = 0
        batch["labels"] = []
    return batch


def check_ctc_compatibility(batch):
    """Checks if processed audio length is sufficient for transcript length."""
    try:
        if batch.get("input_features") is None:
            return False
        processed_audio_length = batch["input_length"]
        labels = batch["labels"]
        if not hasattr(labels, "__len__"):
            raise TypeError(f"'labels' field must be a sequence, got {type(labels)}")
        target_length = len(labels)
        return processed_audio_length >= target_length
    except (KeyError, TypeError) as e:
        warnings.warn(f"CTC compatibility check failed for a batch: {e}. Marking as incompatible.")
        return False
    except Exception as e:
        warnings.warn(f"Unexpected error in CTC compatibility check: {e}. Marking as incompatible.")
        return False

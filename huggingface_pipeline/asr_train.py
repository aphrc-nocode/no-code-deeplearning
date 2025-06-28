import argparse
import json
import os
import random
import traceback
from functools import partial

import datasets
import evaluate
import numpy as np
import torch
import wandb
from datasets import Audio, Dataset, DatasetDict, load_dataset
from transformers import (
    EarlyStoppingCallback,
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
)

from asr_utils.data_processing import (
    check_ctc_compatibility,
    prepare_dataset_for_ctc,
    process_prepared_speech_dataset,
)
from asr_utils.model import DataCollatorCTCWithPadding, compute_metrics_fn
from asr_utils.utils import (
    add_duration_column,
    find_problematic_audio_files,
    random_split,
    split_hf_dataset,
)


def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Wav2Vec2-BERT model for ASR.")

    # --- General & Path Arguments ---
    parser.add_argument("--language_code", type=str, default="en", help="Language code for the dataset.")
    parser.add_argument("--language", type=str, default="english", help="Language of the dataset.")
    parser.add_argument("--num_proc", type=int, default=int(os.cpu_count() / 2), help="Number of processes for data processing.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the audio data.")
    parser.add_argument("--output_dir", type=str, default="w2v-bert-2.0-en-finetuned", help="The output directory to save the model.")
    parser.add_argument("--model_version", type=str, default="v0.0", help="Version for the trained model.")
    parser.add_argument("--vocab_file_path", type=str, default="./vocab.json", help="Path to save the vocabulary file.")

    # --- Dataset Column Names ---
    parser.add_argument("--audio_column", type=str, default="audio", help="Column name for audio data.")
    parser.add_argument("--duration_column", type=str, default="duration", help="Column name for audio duration.")
    parser.add_argument("--speaker_id_column", type=str, default=None, help="Column name for speaker ID.")
    parser.add_argument("--text_column", type=str, default="sentence", help="Column name for original transcript.")
    parser.add_argument("--normalized_text_column", type=str, default="normalized_text", help="Column name for cleaned transcript.")

    # --- Preprocessing Parameters ---
    parser.add_argument("--target_sampling_rate", type=int, default=16_000, help="Target sampling rate for audio.")
    parser.add_argument("--min_duration_s", type=float, default=1.0, help="Minimum audio duration in seconds.")
    parser.add_argument("--max_duration_s", type=float, default=30.0, help="Maximum audio duration in seconds.")
    parser.add_argument("--min_transcript_len", type=int, default=10, help="Minimum transcript length.")
    parser.add_argument("--max_transcript_len", type=int, default=300, help="Maximum transcript length.")
    parser.add_argument("--outlier_std_devs", type=float, default=2.0, help="Standard deviations for outlier filtering.")
    parser.add_argument("--apply_outlier_filtering", action="store_true", help="Apply outlier filtering.")
    parser.add_argument("--speaker_disjointness", action="store_true", help="Ensure speaker disjointness in splits.")
    parser.add_argument("--is_presplit", action="store_true", help="Whether the dataset is already split into train/val/test.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio.")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Development set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio.")

    # --- Vocabulary & Tokenizer ---
    parser.add_argument("--alphabet", type=str, default=" 'abcdefghijklmnopqrstuvwxyz", help="Alphabet for the tokenizer.")

    # --- Model & Processor ---
    parser.add_argument("--model_checkpoint", type=str, default="facebook/w2v-bert-2.0", help="Model checkpoint to fine-tune.")

    # --- Training Parameters ---
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", type=float, default=5.0, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Total number of checkpoints to save.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--early_stopping_threshold", type=float, default=1e-3, help="Threshold for early stopping.")
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer to use.")

    # --- Hugging Face Hub ---
    parser.add_argument("--hub_model_id", type=str, default="", help="The Hub model repository to push the model to.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to the Hugging Face Hub.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Push to a private repository on the Hub.")

    # --- W&B Logging (Optional) ---
    parser.add_argument("--wandb_project", type=str, default="", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity name.")
    parser.add_argument("--log_to_wandb", action="store_true", help="Log metrics to W&B.")

    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    args = get_args()

    # --- Constants ---
    UNK_TOKEN = "[UNK]"
    PAD_TOKEN = "[PAD]"
    WORD_DELIMITER_TOKEN = "|"

    # Set seed for reproducibility
    datasets.utils.logging.set_verbosity_error()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # --- 1. Load and Preprocess Datasets ---
    print(f"\n{'='*10} Loading and Processing: Dataset {'='*10}")

    try:
        dataset = load_dataset("audiofolder", data_dir=args.data_dir)
        dataset = dataset.cast_column(args.audio_column, Audio(sampling_rate=args.target_sampling_rate))

        for split in dataset.keys():
            problematic_files = find_problematic_audio_files(
                dataset[split], audio_column=args.audio_column, verbose=True
            )

            if problematic_files:
                indices_to_remove = {item["index"] for item in problematic_files}
                indices_to_keep = sorted(list(set(range(len(dataset[split]))) - indices_to_remove))
                dataset[split] = dataset[split].select(indices_to_keep)

        if len(dataset) == 0:
            raise ValueError("Dataset is empty after removing problematic files.")

        dataset = add_duration_column(
            dataset,
            num_proc=args.num_proc,
            audio_column=args.audio_column,
            duration_column=args.duration_column,
        )

        if not args.is_presplit:
            if args.speaker_id_column and args.speaker_disjointness:
                train, dev, test = split_hf_dataset(
                    dataset=dataset,
                    speaker_id_col=args.speaker_id_column,
                    duration_col=args.duration_column,
                    train_ratio=args.train_ratio,
                    dev_ratio=args.dev_ratio,
                    test_ratio=args.test_ratio,
                    random_seed=args.random_seed,
                    num_proc=args.num_proc,
                )
            else:
                train, dev, test = random_split(
                    dataset,
                    train_ratio=args.train_ratio,
                    dev_ratio=args.dev_ratio,
                    random_seed=args.random_seed,
                )

            dataset = DatasetDict({"train": train, "validation": dev, "test": test})

        dataset = process_prepared_speech_dataset(
            input_dataset=dataset,
            text_column=args.text_column,
            normalized_text_column=args.normalized_text_column,
            duration_column=args.duration_column,
            min_duration_s=args.min_duration_s,
            max_duration_s=args.max_duration_s,
            min_transcript_len=args.min_transcript_len,
            max_transcript_len=args.max_transcript_len,
            outlier_std_devs=args.outlier_std_devs,
            apply_outlier_filtering=args.apply_outlier_filtering,
            num_proc=args.num_proc,
        )

    except Exception as e:
        print(f"Critical error processing dataset: {e}")
        traceback.print_exc()
        dataset = DatasetDict(
            {"train": Dataset.from_dict({}), "validation": Dataset.from_dict({}), "test": Dataset.from_dict({})}
        )

    # --- 2. Create Vocabulary, Tokenizer, Feature Extractor, Processor ---
    print("\n--- Creating Tokenizer, Feature Extractor, and Processor ---")
    alphabet_list = sorted(list(set(args.alphabet)))
    vocab_dict = {v: k for k, v in enumerate(alphabet_list)}
    vocab_dict[WORD_DELIMITER_TOKEN] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict[UNK_TOKEN] = len(vocab_dict)
    vocab_dict[PAD_TOKEN] = len(vocab_dict)

    vocab_dir = os.path.dirname(args.vocab_file_path)
    if vocab_dir and not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir, exist_ok=True)

    with open(args.vocab_file_path, "w", encoding="utf-8") as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        os.path.dirname(args.vocab_file_path) if os.path.dirname(args.vocab_file_path) else "./",
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        word_delimiter_token=WORD_DELIMITER_TOKEN,
    )

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.model_checkpoint)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # --- 3. Prepare Datasets for CTC Model ---
    print("\n--- Preparing datasets for CTC model ---")

    prepare_ctc_partial = partial(
        prepare_dataset_for_ctc,
        processor=processor,
        audio_col=args.audio_column,
        normalized_text_col=args.normalized_text_column,
    )

    dataset["train"] = dataset["train"].map(
        prepare_ctc_partial,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_proc,
        desc="Preparing train data for CTC",
    )

    dataset["validation"] = dataset["validation"].map(
        prepare_ctc_partial,
        remove_columns=dataset["validation"].column_names,
        num_proc=args.num_proc,
        desc="Preparing dev data for CTC",
    )

    dataset["test"] = dataset["test"].map(
        prepare_ctc_partial,
        remove_columns=dataset["test"].column_names,
        num_proc=args.num_proc,
        desc="Preparing test data for CTC",
    )

    # Filter incompatible samples
    print("Checking prepared datasets for CTC compatibility (input_length >= label_length)...")
    train_dataset = dataset["train"].filter(check_ctc_compatibility, num_proc=args.num_proc)
    removed_train_samples = len(dataset["train"]) - len(train_dataset)
    print(f"Train dataset: {len(train_dataset)} samples (Removed {removed_train_samples} incompatible samples).")

    dev_dataset = dataset["validation"].filter(check_ctc_compatibility, num_proc=args.num_proc)
    removed_dev_samples = len(dataset["validation"]) - len(dev_dataset)
    print(f"Dev dataset: {len(dev_dataset)} samples (Removed {removed_dev_samples} incompatible samples).")

    test_dataset = dataset["test"].filter(check_ctc_compatibility, num_proc=args.num_proc)
    removed_test_samples = len(dataset["test"]) - len(test_dataset)
    print(f"Test dataset: {len(test_dataset)} samples (Removed {removed_test_samples} incompatible samples).")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after CTC compatibility filtering. Stopping.")

    # --- 4. Setup Model Training ---
    print("\n--- Setting up Model Training ---")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    compute_metrics_callback = compute_metrics_fn(processor, wer_metric, cer_metric)

    model = Wav2Vec2BertForCTC.from_pretrained(
        args.model_checkpoint,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    
    if args.log_to_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.output_dir,
                config=vars(args),
            )
        except ImportError:
            print("WandB not installed. Skipping W&B logging.")
            args.log_to_wandb = False
        except Exception as e:
            print(f"Error initializing WandB: {e}. Skipping W&B logging.")
            args.log_to_wandb = False
    else:
        wandb.init(mode="disabled")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim=args.optimizer,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        hub_private_repo=args.hub_private_repo,
        dataloader_num_workers=args.num_proc,
        report_to="wandb" if args.log_to_wandb and wandb.run else None,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics_callback,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )

    # --- 5. Train Model ---
    print("\n--- Starting Model Training ---")
    trainer.train()

    # --- 6. Model Testing ---
    print("\n--- Model Testing ---")
    results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    print(f"Results for test dataset: {results}")

    print("\n--- End of Script ---")


if __name__ == "__main__":
    main()

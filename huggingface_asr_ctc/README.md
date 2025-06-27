# Fine-Tuning for Connectionist Temporal Classification (CTC) Automatic Speech Recognition (ASR)

This project provides a comprehensive pipeline for fine-tuning any pretrained Connectionist Temporal Classification (CTC) model for Automatic Speech Recognition (ASR) tasks. The scripts are designed to be modular, configurable, and easy to use, allowing for robust experimentation with different datasets and training parameters.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and training purposes.

### Prerequisites

You will need Python 3.8+ and a package manager like `pip`. It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv asr_env
source asr_env/bin/activate  # On Windows, use `asr_env\Scripts\activate`
```

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: For GPU support, ensure you have the correct version of PyTorch installed for your CUDA version. See the [official PyTorch website](https://pytorch.org/get-started/locally/) for installation instructions.*

## Data Directory Structure

Store your dataset in a directory structure like:

```bash
data_dir/train/metadata.csv
data_dir/train/0001.mp3
data_dir/train/0002.mp3
data_dir/train/0003.mp3
```

Your metadata.csv file must have a file_name column which links audio files with their transcript e.g.:

```bash
file_name,sentence
0001.mp3,This is a transcript for 0001.mp3
0002.mp3,This is a transcript for 0002.mp3
```

## Usage

The main training process is handled by the `train.py` script, which accepts various command-line arguments to customize the training run.

### Running the Training Script

To start training, run the `train.py` script with the required `data_dir` argument and any other custom parameters.

**Example:**

```bash
python train.py \
    --data_dir /path/to/your/audio/dataset \
    --output_dir_prefix my-asr-model \
    --model_version v1.0 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 10 \
    --push_to_hub \
    --hub_model_id_prefix "your-hf-username" \
    --log_to_wandb \
    --wandb_project "my-asr-project"
```

### Command-Line Arguments

Here is a list of all available arguments to configure the training pipeline:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data_dir` | `str` | **Required** | Directory containing the audio data. |
| `--language_code` | `str` | `en` | Language code for the dataset. |
| `--language` | `str` | `english` | Language of the dataset. |
| `--num_proc` | `int` | `cpu_count / 2` | Number of processes for data processing. |
| `--random_seed` | `int` | `42` | Random seed for reproducibility. |
| `--output_dir_prefix` | `str` | `w2v-bert-2.0-en-finetuned` | Prefix for the output directory. |
| `--model_version` | `str` | `v0.0` | Version for the trained model. |
| `--vocab_file_path` | `str` | `./vocab.json` | Path to save the vocabulary file. |
| `--speaker_id_column` | `str` | `None` | Column name for speaker ID. |
| `--text_column` | `str` | `sentence` | Column name for transcript. |
| `--target_sampling_rate` | `int` | `16000` | Target sampling rate for audio. |
| `--min_duration_s` | `float` | `1.0` | Minimum audio duration in seconds. |
| `--max_duration_s` | `float` | `30.0` | Maximum audio duration in seconds. |
| `--min_transcript_len` | `int` | `10` | Minimum transcript length. |
| `--max_transcript_len` | `int` | `300` | Maximum transcript length. |
| `--outlier_std_devs` | `float` | `2.0` | Standard deviations for outlier filtering. |
| `--apply_outlier_filtering` | `bool` | `False` | Apply outlier filtering. |
| `--speaker_disjointness` | `bool` | `False` | Ensure speaker disjointness in splits. |
| `--is_presplit` | `bool` | `False` | Whether the dataset is already split. |
| `--train_ratio` | `float` | `0.8` | Training set ratio. |
| `--dev_ratio` | `float` | `0.1` | Development set ratio. |
| `--test_ratio` | `float` | `0.1` | Test set ratio. |
| `--alphabet` | `str` | `" 'abcdefghijklmnopqrstuvwxyz"` | Alphabet for the tokenizer. |
| `--model_checkpoint` | `str` | `facebook/w2v-bert-2.0` | Model checkpoint to fine-tune. |
| `--train_batch_size` | `int` | `16` | Training batch size per device. |
| `--eval_batch_size` | `int` | `16` | Evaluation batch size per device. |
| `--gradient_accumulation_steps` | `int` | `1` | Gradient accumulation steps. |
| `--num_train_epochs` | `float` | `5.0` | Number of training epochs. |
| `--learning_rate` | `float` | `3e-4` | Learning rate. |
| `--lr_scheduler_type` | `str` | `linear` | Learning rate scheduler type. |
| `--warmup_ratio` | `float` | `0.1` | Warmup ratio for the LR scheduler. |
| `--save_total_limit` | `int` | `2` | Total number of checkpoints to save. |
| `--early_stopping_patience` | `int` | `5` | Patience for early stopping. |
| `--early_stopping_threshold` | `float` | `1e-3` | Threshold for early stopping. |
| `--optimizer` | `str` | `adamw_torch` | Optimizer to use. |
| `--hub_model_id_prefix` | `str` | `""` | Prefix for the Hub model ID. |
| `--push_to_hub` | `bool` | `False` | Push model to the Hugging Face Hub. |
| `--hub_private_repo` | `bool` | `False` | Push to a private repository on the Hub. |
| `--wandb_project` | `str` | `""` | W\&B project name. |
| `--wandb_entity` | `str` | `""` | W\&B entity name. |
| `--log_to_wandb` | `bool` | `False` | Log metrics to W\&B. |

## Project Structure

The project is organized into several modules to separate concerns and improve maintainability:

  - **`train.py`**: The main script that orchestrates the entire training pipeline, from data loading to model evaluation. It handles command-line argument parsing and coordinates the other modules.
  - **`model.py`**: Defines the model-related components, including the `DataCollatorCTCWithPadding` for batching and the `compute_metrics_fn` for calculating WER and CER during evaluation.
  - **`data_processing.py`**: Contains all functions related to data cleaning, filtering, and preparation. This includes normalizing transcripts, filtering by duration/length, and preparing the dataset for the CTC model.
  - **`utils.py`**: A collection of helper functions used across the project, such as dataset splitting, duration calculation, and audio file validation.

## Training Process

The training pipeline consists of the following steps, executed sequentially in `train.py`:

1.  **Argument Parsing**: All command-line arguments are parsed to configure the run.
2.  **Dataset Loading**: The audio dataset is loaded from the specified `data_dir`.
3.  **Data Validation & Preprocessing**:
      - Problematic audio files are identified and removed.
      - Audio durations are calculated and added as a new column.
      - If not pre-split, the dataset is split into training, validation, and test sets.
      - Transcripts are cleaned and normalized.
      - Samples are filtered based on audio duration, transcript length, and (optionally) outlier detection.
4.  **Processor Setup**: A vocabulary is created from the specified alphabet, and the `Wav2Vec2BertProcessor` (combining the feature extractor and tokenizer) is initialized.
5.  **CTC Data Preparation**: The processed datasets are mapped to the format required by the CTC model, including creating `input_features` and `labels`.
6.  **Model & Trainer Setup**:
      - The `Wav2Vec2BertForCTC` model is loaded from the specified checkpoint.
      - The `DataCollator`, evaluation metrics, and `TrainingArguments` are configured.
      - The `Trainer` is initialized with all the necessary components.
7.  **Training**: The `trainer.train()` method is called to start the fine-tuning process.
8.  **Evaluation**: After training, the model is evaluated on the test set, and the final WER/CER metrics are printed.

## Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

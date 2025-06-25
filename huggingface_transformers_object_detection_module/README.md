# Modular Object Detection Training Pipeline
This project provides a configurable and modular pipeline for fine-tuning DETR-based object detection models using the Hugging Face transformers library. The code is structured to be easily extensible and reusable for different datasets and models.

# File Structure
The project is organized into several Python modules, each with a specific responsibility:
- `train.py`: The main script for executing the training and evaluation pipeline. It handles command-line argument parsing, orchestrates the overall workflow, and ties all the other modules together.
- `data_utils.py`: Contains functions for loading the dataset and processing annotations.
- `augmentations.py`: Defines the data augmentation pipelines for training and validation using albumentations.
- `model_utils.py`: Includes helper functions for loading the object detection model and the custom data collator.
- `metrics.py`: Handles the computation of evaluation metrics, such as mean Average Precision (mAP).
- `requirements.txt`: A list of all the Python packages required to run this pipeline.
- `detection_dataset_builder.py`: A script that defines how to load the custom dataset.
- `app.py`: A web-based UI built with Gradio to easily configure and run the training pipeline.

# Setup and Installation
Follow these steps to set up your local environment to run the pipeline.

## 1. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

### Create a new virtual environment

    python -m venv venv
    
### Activate the virtual environment
#### On Windows:

    venv\Scripts\activate
    
#### On macOS and Linux:

    source venv/bin/activate

## 2. Install Dependencies
Install all the required packages using the `requirements.txt` file.

    pip install -r requirements.txt

## 3. Dataset Preparation
This pipeline is designed to work with a local dataset loaded by the `detection_dataset_builder.py` script. You must specify the path to your dataset's base directory using the `--base_dir` argument.\
The dataset builder expects the following directory structure within the `--base_dir`:

    <base_dir>/
        └── train/
            ├── _annotations.coco.json
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
        └── validation/
            ├── _annotations.coco.json
            ├── image3.jpg
            ├── image4.jpg
            ├── ...
        └── test/
            ├── _annotations.coco.json
            ├── image5.jpg
            ├── image6.jpg
            ├── ...

Each subdirectory (`train`, `validation`, `test`) represents a data split.\
Each subdirectory must contain its corresponding image files (.jpg, .png, etc.) and a single COCO-formatted _annotations.coco.json file that holds the annotations for all images in that split.

# How to Run the Pipeline
You have two ways to run the training pipeline:
- Using the interactive web UI (Recommended for ease of use).
- Directly from the command line.

## Method 1: Running the Interactive UI
This project includes a simple web-based UI built with Gradio to make it easier to configure and run the training pipeline.

### Launch the UI
To start the UI, run the `app.py` script:

    python app.py

This will start a local web server and provide you with a URL (usually `http://127.0.0.1:7860`). Open this URL in your web browser to access the application.

### Using the UI
- **Fill in the parameters**: Adjust the hyperparameters and settings using the interactive form fields on the right.
- **Start Training**: Click the **Start Training** button.
- **Monitor Progress**: The training logs will be streamed in real-time to the **Training Log** textbox on the left.

## Method 2: Running from the Command Line
The `train.py` script is the entry point for the entire process. You can configure the training run using command-line arguments.

### Basic Training Command
To start training with the default hyperparameters, run the following command. Make sure to replace `/path/to/your/dataset` with the actual path to your data.

    python train.py --base_dir /path/to/your/dataset

### Advanced Training with Custom Hyperparameters
You can customize various aspects of the training run, such as the learning rate, batch size, number of epochs, and more.\
Here is an example of a more complex command:

    python train.py \
        --base_dir /path/to/your/dataset \
        --model_checkpoint "facebook/detr-resnet-101" \
        --run_name "detr-resnet101-finetune" \
        --version "1.0" \
        --epochs 20 \
        --lr 5e-5 \
        --batch_size 8 \
        --eval_batch_size 8 \
        --fp16 \
        --use_wandb \
        --push_to_hub

### Command-Line Arguments
- `--base_dir`: (Required) Path to the base directory of your dataset.
- `--model_checkpoint`: The Hugging Face model to use for fine-tuning. (Default: `facebook/detr-resnet-50`)
- `run_name`: A custom name for your training run. (Default: `dswb_test`)
- `--version`: The version of your training run. (Default: `0.0`)
- `--epochs`: The number of training epochs. (Default: `10`)
- `--lr`: The learning rate for the optimizer. (Default: `1e-4`)
- `--batch_size`: The batch size for training. (Default: `4`)
- `--eval_batch_size`: The batch size for evaluation. (Default: `4`)
- `--fp16`: If specified, enables mixed-precision training.
- `--use_wandb`: If specified, enables logging with Weights & Biases.
- `--push_to_hub`: If specified, pushes the final trained model to the Hugging Face Hub.

# Output
After the training is complete, the script will:
- Save the best model checkpoint, tokenizer, and training configuration in the specified `output_dir`.
- Print the evaluation metrics for the test set to the console.
- If `--push_to_hub` is enabled, upload the model to your Hugging Face account.

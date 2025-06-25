# Modular Object Detection Training Pipeline
This project provides a configurable and modular pipeline for fine-tuning DETR-based object detection models using the Hugging Face transformers library. The code is structured to be easily extensible and reusable for different datasets and models.

# File Structure
The project is organized into several Python modules, each with a specific responsibility:
- **train.py**: The main script for executing the training and evaluation pipeline. It handles command-line argument parsing, orchestrates the overall workflow, and ties all the other modules together.
- **data_utils.py**: Contains functions for loading the dataset and processing annotations.
- **augmentations.py**: Defines the data augmentation pipelines for training and validation using albumentations.
- **model_utils.py**: Includes helper functions for loading the object detection model and the custom data collator.
- **metrics.py**: Handles the computation of evaluation metrics, such as mean Average Precision (mAP).
- **requirements.txt**: A list of all the Python packages required to run this pipeline.
- **detection_dataset_builder.py**: A script that defines how to load the custom dataset.

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
Install all the required packages using the **requirements.txt** file.

    pip install -r requirements.txt


## 3. Dataset Preparation
This pipeline is designed to work with a local dataset loaded by the **detection_dataset_builder.py** script. You must specify the path to your dataset's base directory using the _--base_dir_ argument.\
The dataset builder expects the following directory structure within the _--base_dir_:

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

Each subdirectory (train, validation, test) represents a data split.\
Each subdirectory must contain its corresponding image files (.jpg, .png, etc.) and a single COCO-formatted _annotations.coco.json file that holds the annotations for all images in that split.

# How to Run the Pipeline
The **train.py** script is the entry point for the entire process. You can configure the training run using command-line arguments.

# Basic Training Command
To start training with the default hyperparameters, run the following command. Make sure to replace _/path/to/your/dataset_ with the actual path to your data.

    python train.py --base_dir /path/to/your/dataset

# Advanced Training with Custom Hyperparameters
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

# Command-Line Arguments
- **--base_dir**: (Required) Path to the base directory of your dataset.
- **--model_checkpoint**: The Hugging Face model to use for fine-tuning. _(Default: facebook/detr-resnet-50)_
- **--run_name**: A custom name for your training run. _(Default: dswb_test)_
- **--version**: The version of your training run. _(Default: 0.0)_
- **--epochs**: The number of training epochs. _(Default: 10)_
- **--lr**: The learning rate for the optimizer. _(Default: 1e-4)_
- **--batch_size**: The batch size for training. _(Default: 4)_
- **--eval_batch_size**: The batch size for evaluation. _(Default: 4)_
- **--fp16**: If specified, enables mixed-precision training.
- **--use_wandb**: If specified, enables logging with Weights & Biases.
- **--push_to_hub**: If specified, pushes the final trained model to the Hugging Face Hub.

# Output
After the training is complete, the script will:
- Save the best model checkpoint, tokenizer, and training configuration in the specified output_dir.
- Print the evaluation metrics for the test set to the console.
- If --push_to_hub is enabled, upload the model to your Hugging Face account.

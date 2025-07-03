#!/usr/bin/env python
"""
Test script for object detection pipeline with Hugging Face transformer models.
Example usage:
    python test_huggingface_detection.py --dataset_path ./test_data --model detr_resnet50
"""

import argparse
import asyncio
from pathlib import Path
import sys

from main import PipelineConfig, TaskType, ModelArchitecture, PipelineFactory

async def main():
    parser = argparse.ArgumentParser(description="Test Hugging Face object detection pipeline.")
    parser.add_argument("--dataset_path", type=str, default="./test_data", help="Path to test dataset")
    parser.add_argument("--model", type=str, default="detr_resnet50", 
                        choices=["detr_resnet50", "detr_resnet101", "yolos_small", "yolos_base", "owlv2_base"],
                        help="Model architecture to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create architecture enum from string
    architecture_map = {
        "detr_resnet50": ModelArchitecture.DETR_RESNET50,
        "detr_resnet101": ModelArchitecture.DETR_RESNET101,
        "yolos_small": ModelArchitecture.YOLOS_SMALL,
        "yolos_base": ModelArchitecture.YOLOS_BASE,
        "owlv2_base": ModelArchitecture.OWLV2_BASE,
    }
    architecture = architecture_map.get(args.model)
    
    if not architecture:
        print(f"Invalid model architecture: {args.model}")
        return
    
    # Determine number of classes from dataset
    # This is a placeholder; in a real scenario, we'd analyze the dataset
    num_classes = 1  # Default to 2 classes for testing
    
    # Create pipeline config
    config = PipelineConfig(
        name=f"HF-{args.model} Test",
        task_type=TaskType.OBJECT_DETECTION,
        architecture=architecture,
        num_classes=num_classes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        early_stopping=False  # Disable early stopping for testing
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Create a unique job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    print(f"Starting training job {job_id} with {args.model} on {args.dataset_path}")
    print(f"Config: {config}")
    
    # Run training
    result = await pipeline.train(args.dataset_path, job_id)
    
    # Print results
    print("\n===== TRAINING RESULTS =====")
    for k, v in result.items():
        if k != "metrics":
            print(f"{k}: {v}")
    
    if "metrics" in result:
        print("\n===== METRICS =====")
        for k, v in result["metrics"].items():
            print(f"{k}: {v}")
    
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())

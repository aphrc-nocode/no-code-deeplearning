"""
Examples for using the different computer vision pipelines.
This script demonstrates how to use the various pipelines for image classification, object detection, and image segmentation.
"""
import asyncio
import os
from pathlib import Path
from main import (
    TaskType, ModelArchitecture, PipelineConfig, SegmentationType, 
    PipelineFactory, 
    # setup_directories
)

async def run_image_classification_example(dataset_path: str):
    """Run image classification pipeline example"""
    print("\n===== Image Classification Example =====")
    
    # Create config for image classification
    config = PipelineConfig(
        name="Image Classification Example",
        task_type=TaskType.IMAGE_CLASSIFICATION,
        architecture=ModelArchitecture.RESNET18,
        num_classes=2,
        batch_size=8,
        epochs=10,  # Use small number of epochs for example
        learning_rate=0.001,
        image_size=(224, 224),
        augmentation_enabled=True,
        early_stopping=True,
        patience=2
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Train model
    print(f"Training image classification model on {dataset_path}")
    result = await pipeline.train(dataset_path, "example_classification")
    
    print(f"Training result: {result['status']}")
    if result['status'] == 'completed':
        print(f"Model saved to: {result['model_path']}")
        print("Metrics:")
        for name, value in result['metrics'].items():
            if value is not None:
                print(f"  {name}: {value:.4f}")

async def run_object_detection_example(dataset_path: str):
    """Run object detection pipeline example"""
    print("\n===== Object Detection Example =====")
    
    # Create config for object detection
    config = PipelineConfig(
        name="Object Detection Example",
        task_type=TaskType.OBJECT_DETECTION,
        architecture=ModelArchitecture.FASTER_RCNN,
        num_classes=2,  # Background + 1 object class
        batch_size=2,  # Smaller batch size for detection
        epochs=2,  # Use small number of epochs for example
        learning_rate=0.005,
        image_size=(800, 800),  # Larger image size for detection
        augmentation_enabled=True,
        early_stopping=True,
        patience=2
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Train model
    print(f"Training object detection model on {dataset_path}")
    result = await pipeline.train(dataset_path, "example_detection")
    
    print(f"Training result: {result['status']}")
    if result['status'] == 'completed':
        print(f"Model saved to: {result['model_path']}")
        print("Metrics:")
        for name, value in result['metrics'].items():
            if value is not None:
                print(f"  {name}: {value:.4f}")

async def run_semantic_segmentation_example(dataset_path: str):
    """Run semantic segmentation pipeline example"""
    print("\n===== Semantic Segmentation Example =====")
    
    # Create config for semantic segmentation
    config = PipelineConfig(
        name="Semantic Segmentation Example",
        task_type=TaskType.IMAGE_SEGMENTATION,
        architecture=ModelArchitecture.FCN,  # Using FCN instead of DEEPLABV3
        num_classes=2,  # Background + 1 segmentation class
        batch_size=2,  # Smaller batch size for segmentation
        epochs=2,  # Use small number of epochs for example
        learning_rate=0.001,
        image_size=(512, 512),
        augmentation_enabled=True,
        early_stopping=True,
        patience=2,
        segmentation_type=SegmentationType.SEMANTIC
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Train model
    print(f"Training semantic segmentation model on {dataset_path}")
    result = await pipeline.train(dataset_path, "example_semantic_seg")
    
    print(f"Training result: {result['status']}")
    if result['status'] == 'completed':
        print(f"Model saved to: {result['model_path']}")
        print("Metrics:")
        for name, value in result['metrics'].items():
            if value is not None:
                print(f"  {name}: {value:.4f}")

async def run_instance_segmentation_example(dataset_path: str):
    """Run instance segmentation pipeline example"""
    print("\n===== Instance Segmentation Example =====")
    
    # Create config for instance segmentation
    config = PipelineConfig(
        name="Instance Segmentation Example",
        task_type=TaskType.IMAGE_SEGMENTATION,
        architecture=ModelArchitecture.MASK_RCNN,
        num_classes=3,  # Background + 2 instance classes (circle, rectangle)
        batch_size=2,  # Smaller batch size for instance segmentation
        epochs=2,  # Use small number of epochs for example
        learning_rate=0.005,
        image_size=(800, 800),
        augmentation_enabled=True,
        early_stopping=True,
        patience=2,
        segmentation_type=SegmentationType.INSTANCE
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Train model
    print(f"Training instance segmentation model on {dataset_path}")
    result = await pipeline.train(dataset_path, "example_instance_seg")
    
    print(f"Training result: {result['status']}")
    if result['status'] == 'completed':
        print(f"Model saved to: {result['model_path']}")
        print("Metrics:")
        for name, value in result['metrics'].items():
            if value is not None:
                print(f"  {name}: {value:.4f}")

async def main():
    """Run examples for all pipelines"""
    # Setup required directories
    # setup_directories()
    
    # Use the same test dataset for all examples
    # In a real scenario, you would use appropriate datasets for each task
    test_dataset_path = "/home/alvin/demo"
    
    # Ensure test dataset exists
    if not os.path.exists(test_dataset_path):
        print(f"Test dataset not found at {test_dataset_path}")
        print("Please create a test dataset with class subdirectories")
        return
    
    # Run examples
    await run_image_classification_example(test_dataset_path)
    # Uncomment to run other examples once you have appropriate datasets
    # await run_object_detection_example("datasets/detection_data")
    # await run_semantic_segmentation_example("datasets/segmentation_data")
    # await run_instance_segmentation_example("datasets/instance_seg_data")
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    asyncio.run(main())

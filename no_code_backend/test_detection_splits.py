"""
Test script to verify detection dataset split handling.
This script checks if the object detection pipeline properly handles pre-split datasets
and avoids creating unnecessary splits.
"""
import asyncio
import os
from pathlib import Path
from main import TaskType, ModelArchitecture, PipelineConfig
from pipelines.object_detection_pipeline import ObjectDetectionPipeline
from datasets_module.detection.dataloaders import create_dataloaders, ObjectDetectionDataset

async def test_detection_splits():
    """Test detection dataset split handling"""
    print("\n===== Testing Detection Dataset Split Handling =====")
    
    # Define dataset paths to test
    detection_data_path = Path("datasets/detection_data")
    
    # Create config
    config = PipelineConfig(
        name="Detection Split Test",
        task_type=TaskType.OBJECT_DETECTION,
        architecture=ModelArchitecture.FASTER_RCNN,
        num_classes=2,
        batch_size=2,
        epochs=1,  # Just 1 epoch for testing
        learning_rate=0.005,
        image_size=(640, 640)
    )
    
    # Create a temporary job ID that we can use to track splits
    job_id = "test_detection_splits"
    
    # Clean up any existing splits for this job ID
    try:
        ObjectDetectionDataset.cleanup_splits(job_id)
        print(f"Cleaned up any existing splits for job ID {job_id}")
    except:
        pass
    
    # Test if split is needed
    print("\nTesting dataset structure detection:")
    dataset_path = detection_data_path
    
    print(f"\nAnalyzing dataset at {dataset_path}...")
    
    # Create dataloaders to see if splits are created
    from datasets_module.detection.transforms import get_detection_transforms
    transform = get_detection_transforms(train=True)
    
    splits_dir = Path("dataset_splits") / job_id
    if splits_dir.exists():
        print(f"WARNING: Splits directory {splits_dir} already exists before test")
    else:
        print(f"Splits directory {splits_dir} does not exist before test (expected)")
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        dataset_path,
        transform=transform,
        batch_size=2,
        val_split=0.2,
        test_split=0.1,
        job_id=job_id,
        use_saved_splits=False
    )
    
    # Check if splits were created
    if splits_dir.exists():
        print(f"RESULT: Splits were created at {splits_dir}")
        split_files = list(splits_dir.glob("*"))
        print(f"Found {len(split_files)} files in splits directory: {[f.name for f in split_files]}")
        
        # Load splits to see what they contain
        try:
            splits_data = ObjectDetectionDataset.load_splits(job_id)
            print(f"Loaded splits contain: {len(splits_data['train'])} train, {len(splits_data['val'])} val, {len(splits_data['test'])} test samples")
        except Exception as e:
            print(f"Failed to load splits: {e}")
    else:
        print(f"RESULT: No splits were created (splits directory {splits_dir} doesn't exist)")
    
    # Create a pipeline to test with verbose logging
    print("\nTesting with pipeline...")
    pipeline = ObjectDetectionPipeline(config)
    try:
        # Clean up before running
        ObjectDetectionDataset.cleanup_splits(job_id)
        
        # Run the pipeline training
        result = await pipeline.train(dataset_path, job_id)
        print(f"Pipeline result: {result['status']}")
        
        # Check if splits were created during pipeline execution
        if splits_dir.exists():
            print(f"RESULT: Pipeline created splits at {splits_dir}")
            # Clean up afterwards
            ObjectDetectionDataset.cleanup_splits(job_id)
        else:
            print(f"RESULT: Pipeline did not create splits")
    except Exception as e:
        print(f"Pipeline error: {e}")
    
    # Test evaluation function
    print("\nTesting evaluation function...")
    try:
        # Clean up before running
        if splits_dir.exists():
            ObjectDetectionDataset.cleanup_splits(job_id)
        
        # Create a test model for evaluation
        from models.detection import model_factory
        model = model_factory.create_model(
            ModelArchitecture.FASTER_RCNN,
            num_classes=2,
            pretrained=True
        )
        
        # Run evaluation
        from evaluate_model import evaluate_detection_model
        print("\nRunning evaluation...")
        metrics = evaluate_detection_model(
            model,
            dataset_path,
            batch_size=2,
            job_id=job_id
        )
        
        # Check if splits were created during evaluation
        if splits_dir.exists():
            print(f"RESULT: Evaluation created splits at {splits_dir}")
            # Clean up afterwards
            ObjectDetectionDataset.cleanup_splits(job_id)
        else:
            print(f"RESULT: Evaluation did not create splits")
    except Exception as e:
        print(f"Evaluation error: {e}")

if __name__ == "__main__":
    asyncio.run(test_detection_splits())

"""
Examples for using the different computer vision pipelines.
This script demonstrates how to use the various pipelines for image classification, object detection, and image segmentation.
"""
import asyncio
import os
import torch
import mlflow
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
        batch_size=2,
        epochs=10,  # Use small number of epochs for example
        learning_rate=0.001,
        image_size=(224, 224),
        augmentation_enabled=True,
        early_stopping=True,
        patience=2,
        feature_extraction_only=False  # Use fine-tuning approach
    )
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config)
    
    # Train model
    print(f"Training image classification model on {dataset_path}")
    result = await pipeline.train(dataset_path, "example_classification")
    
    print(f"Training result: {result['status']}")
    if result['status'] == 'completed':
        if result.get('mlflow_run_id'):
            print(f"Model saved to MLflow - Run ID: {result['mlflow_run_id']}")
            print(f"MLflow Model URI: {result['mlflow_model_uri']}")
        else:
            print(f"Model saved to: {result['model_path']}")
            
        print("Training Metrics:")
        # Print the metrics that are available in the result
        metrics = [
            ('Training Loss', result.get('final_train_loss')),
            ('Training Accuracy', result.get('final_train_accuracy')),
            ('Validation Loss', result.get('final_val_loss')),
            ('Validation Accuracy', result.get('final_val_accuracy')),
            ('Epochs Completed', result.get('epochs_completed')),
            ('Training Time (s)', result.get('training_time'))
        ]
        for name, value in metrics:
            if value is not None:
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
        
        # Evaluate model on test data
        print("\nEvaluating classification model...")
        print("Looking for test directory, then validation directory, or creating a validation split.")
        if result.get('mlflow_model_uri'):
            print(f"Loading model from MLflow URI: {result['mlflow_model_uri']}")
            import mlflow
            model = mlflow.pytorch.load_model(result['mlflow_model_uri'])
        else:
            print(f"Loading model from local path: {result['model_path']}")
            import torch
            from pathlib import Path
            
            model_path = Path(result['model_path'])
            # Check if path is a directory (MLflow format) or a file
            if model_path.is_dir():
                # Look for model.pth inside the directory
                model_file = model_path / "model.pth"
                if not model_file.exists():
                    # Try to find any .pth file
                    pth_files = list(model_path.glob("*.pth"))
                    if pth_files:
                        model_file = pth_files[0]
                    else:
                        # Try to use MLflow to load the model
                        print(f"No model file found in {model_path}, trying MLflow")
                        import mlflow
                        run_id = model_path.name  # Use directory name as run ID
                        try:
                            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
                            # If we successfully load the model, skip the rest of this block
                            print(f"Successfully loaded model from MLflow run ID: {run_id}")
                            checkpoint = None
                        except Exception as e:
                            raise FileNotFoundError(f"No model file found in {model_path} and MLflow loading failed: {e}")
                if 'checkpoint' not in locals() or checkpoint is not None:
                    print(f"Found model file: {model_file}")
                    checkpoint = torch.load(str(model_file), weights_only=False)
            else:
                # Direct file path
                checkpoint = torch.load(str(model_path), weights_only=False)
            
            # Recreate model and load weights if we have a checkpoint
            if 'checkpoint' in locals() and checkpoint is not None:
                from models.classification import model_factory
                model = model_factory.create_model(
                    config.architecture,
                    num_classes=config.num_classes
                )
                model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate the model
        from evaluate_model import evaluate_classification_model
        evaluation_result = evaluate_classification_model(
            model, 
            dataset_path, 
            batch_size=config.batch_size, 
            job_id="example_classification"  # Pass the same job_id to use saved splits
        )
        
        print("Evaluation Metrics:")
        for name, value in evaluation_result.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        
        # Clean up dataset split files
        from datasets_module.classification.dataloaders import ImageClassificationDataset
        print("Cleaning up dataset split files...")
        try:
            ImageClassificationDataset.cleanup_splits("example_classification")
            print("Dataset split files cleaned up successfully")
        except Exception as e:
            print(f"Error cleaning up dataset splits: {e}")

async def run_object_detection_example(dataset_path: str):
    """Run object detection pipeline example"""
    print("\n===== Object Detection Example =====")
    
    # Create config for object detection
    config = PipelineConfig(
        name="Object Detection Example",
        task_type=TaskType.OBJECT_DETECTION,
        architecture=ModelArchitecture.FASTER_RCNN,
        num_classes=2,  #
        batch_size=2,  # Smaller batch size for detection
        epochs=2,  # Use small number of epochs for example
        learning_rate=0.005,
        image_size=(640, 640),  # Larger image size for detection
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
        if result.get('mlflow_run_id'):
            print(f"Model saved to MLflow - Run ID: {result['mlflow_run_id']}")
            print(f"MLflow Model URI: {result['mlflow_model_uri']}")
        else:
            print(f"Model saved to: {result['model_path']}")
            
        print("Training Metrics:")
        # Print the metrics that are available in the result
        metrics = result.get('metrics', {})
        metrics_to_display = [
            ('Training Loss', metrics.get('final_train_loss')),
            ('Training mAP', metrics.get('final_train_mAP')),
            ('Validation Loss', metrics.get('final_val_loss')),
            ('Validation mAP', metrics.get('final_val_mAP')),
            ('Epochs Completed', result.get('epochs_completed')),
            ('Training Time (s)', result.get('training_time'))
        ]
        for name, value in metrics_to_display:
            if value is not None:
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")
        
        # Evaluate model on test data
        print("\nEvaluating detection model on the same dataset...")
        if result.get('mlflow_model_uri'):
            print(f"Loading model from MLflow URI: {result['mlflow_model_uri']}")
            import mlflow
            model = mlflow.pytorch.load_model(result['mlflow_model_uri'])
        else:
            print(f"Loading model from local path: {result['model_path']}")
            import torch
            from pathlib import Path
            
            model_path = Path(result['model_path'])
            # Check if path is a directory (MLflow format) or a file
            if model_path.is_dir():
                # Look for model.pth inside the directory
                model_file = model_path / "model.pth"
                if not model_file.exists():
                    # Try to find any .pth file
                    pth_files = list(model_path.glob("*.pth"))
                    if pth_files:
                        model_file = pth_files[0]
                    else:
                        # Try to use MLflow to load the model
                        print(f"No model file found in {model_path}, trying MLflow")
                        import mlflow
                        run_id = model_path.name  # Use directory name as run ID
                        try:
                            model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
                            # If we successfully load the model, skip the rest of this block
                            print(f"Successfully loaded model from MLflow run ID: {run_id}")
                            checkpoint = None
                        except Exception as e:
                            raise FileNotFoundError(f"No model file found in {model_path} and MLflow loading failed: {e}")
                if 'checkpoint' not in locals() or checkpoint is not None:
                    print(f"Found model file: {model_file}")
                    checkpoint = torch.load(str(model_file), weights_only=False)
            else:
                # Direct file path
                checkpoint = torch.load(str(model_path), weights_only=False)
            
            # Recreate model and load weights if we have a checkpoint
            if 'checkpoint' in locals() and checkpoint is not None:
                from models.detection import model_factory
                model = model_factory.create_model(
                    config.architecture,
                    num_classes=config.num_classes,
                    pretrained=False
                )
                model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate the model
        from evaluate_model import evaluate_detection_model
        evaluation_result = evaluate_detection_model(model, dataset_path, batch_size=config.batch_size, job_id="example_detection")
        
        print("Evaluation Metrics:")
        for name, value in evaluation_result.items():
            if isinstance(value, float):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")
        
        # Clean up any dataset split files
        try:
            from datasets_module.detection.dataloaders import ObjectDetectionDataset
            print("Cleaning up object detection dataset split files...")
            ObjectDetectionDataset.cleanup_splits("example_detection")
            print("Object detection dataset split files cleaned up successfully")
            
            # Also clean up any model directories created in models/
            from pathlib import Path
            model_dir = Path("models") / "example_detection"
            if model_dir.exists():
                import shutil
                print(f"Cleaning up model directory: {model_dir}")
                shutil.rmtree(model_dir)
                print(f"Removed directory: {model_dir}")
        except Exception as e:
            print(f"Error cleaning up files: {e}")

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
        print("Training Metrics:")
        # Print the metrics that are available in the result
        metrics = [
            ('Training Loss', result.get('final_train_loss')),
            ('Training Accuracy', result.get('final_train_accuracy')),
            ('Validation Loss', result.get('final_val_loss')),
            ('Validation Accuracy', result.get('final_val_accuracy')),
            ('Epochs Completed', result.get('epochs_completed')),
            ('Training Time (s)', result.get('training_time'))
        ]
        for name, value in metrics:
            if value is not None:
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

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
        print("Training Metrics:")
        # Print the metrics that are available in the result
        metrics = [
            ('Training Loss', result.get('final_train_loss')),
            ('Training Accuracy', result.get('final_train_accuracy')),
            ('Validation Loss', result.get('final_val_loss')),
            ('Validation Accuracy', result.get('final_val_accuracy')),
            ('Epochs Completed', result.get('epochs_completed')),
            ('Training Time (s)', result.get('training_time'))
        ]
        for name, value in metrics:
            if value is not None:
                if isinstance(value, float):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

async def main():
    """Run examples for all pipelines"""
    # Setup required directories
    # setup_directories()
    
    # Use the same test dataset for all examples
    # In a real scenario, you would use appropriate datasets for each task
    test_dataset_path = "test_data"
    
    # Ensure test dataset exists
    if not os.path.exists(test_dataset_path):
        print(f"Test dataset not found at {test_dataset_path}")
        print("Please create a test dataset with class subdirectories")
        return
    
    # Run examples
    # await run_image_classification_example(test_dataset_path)
    # Uncomment to run other examples once you have appropriate datasets
    await run_object_detection_example("/home/achuka/object-detect")
    # await run_semantic_segmentation_example("datasets/segmentation_data")
    # await run_instance_segmentation_example("datasets/instance_seg_data")
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    asyncio.run(main())

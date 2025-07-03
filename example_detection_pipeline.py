"""
Example script for using the Object Detection Pipeline following the article's approach.
This script demonstrates how to set up and run the pipeline for training custom Faster R-CNN models.
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipelines.object_detection_pipeline import create_detection_pipeline


def main():
    """
    Main function demonstrating the object detection pipeline workflow
    """
    print("=== Object Detection Pipeline Example ===")
    print("Following the 'Pipeline for Training Custom Faster-RCNN Object Detection models with Pytorch' article")
    
    # Example configuration for a custom dataset
    # Replace these paths with your actual dataset paths
    config = {
        # Dataset configuration
        'num_classes': 3,  # Including background (e.g., background, cat, dog)
        'class_names': ['background', 'cat', 'dog'],  # Replace with your actual classes
        
        # Dataset paths (COCO format)
        'train_image_dir': './datasets/custom/train/images',
        'train_annotation_path': './datasets/custom/train/annotations.json',
        'val_image_dir': './datasets/custom/val/images', 
        'val_annotation_path': './datasets/custom/val/annotations.json',
        
        # Model configuration
        'architecture': 'faster_rcnn',  # Following the article's approach
        
        # Training hyperparameters (following the article)
        'learning_rate': 0.005,
        'batch_size': 2,  # Adjust based on GPU memory
        'num_epochs': 10,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'step_size': 3,
        'gamma': 0.1,
        
        # Output configuration
        'output_dir': './outputs/detection',
        'model_save_path': './outputs/detection/faster_rcnn_custom.pth'
    }
    
    # Step 1: Create the pipeline
    print("\n1. Creating Object Detection Pipeline...")
    pipeline = create_detection_pipeline(config)
    
    # Step 2: Load data
    print("\n2. Loading datasets...")
    print("Note: Make sure your dataset is in COCO format!")
    print("Required structure:")
    print("  - Images in train/val directories")
    print("  - COCO-format JSON annotations")
    
    try:
        pipeline.load_data()
        print("✓ Datasets loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading datasets: {e}")
        print("Please check your dataset paths and format.")
        return
    
    # Step 3: Build model
    print("\n3. Building Faster R-CNN model...")
    pipeline.build_model()
    print("✓ Model built successfully!")
    
    # Step 4: Show model summary
    print("\n4. Model Summary:")
    summary = pipeline.get_model_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Step 5: Visualize training data (optional)
    print("\n5. Visualizing training samples...")
    try:
        pipeline.visualize_batch(num_samples=2)
        print("✓ Training samples visualized!")
    except Exception as e:
        print(f"Note: Visualization skipped due to: {e}")
    
    # Step 6: Train the model
    print("\n6. Starting training...")
    print("This will:")
    print("  - Use SGD optimizer with the article's hyperparameters")
    print("  - Apply learning rate scheduling")
    print("  - Log metrics to MLflow")
    print("  - Save the best model based on validation loss")
    
    try:
        results = pipeline.train()
        print("✓ Training completed successfully!")
        print(f"Final model saved to: {results['model_path']}")
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Step 7: Evaluate the model
    print("\n7. Evaluating trained model...")
    try:
        eval_results = pipeline.evaluate()
        print("✓ Evaluation completed!")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Note: Evaluation skipped due to: {e}")
    
    # Step 8: Make predictions (example)
    print("\n8. Example prediction workflow:")
    print("To make predictions on new images:")
    print("  predictions = pipeline.predict('path/to/image.jpg', threshold=0.5)")
    print("  pipeline.visualize_predictions('path/to/image.jpg', threshold=0.8)")
    
    print("\n=== Pipeline Workflow Complete ===")
    print("Your custom Faster R-CNN model is ready for use!")


def create_sample_config():
    """
    Create a sample configuration file for reference
    """
    sample_config = {
        "dataset_info": {
            "description": "Sample configuration for object detection pipeline",
            "format": "COCO format required",
            "structure": {
                "train_images": "./datasets/custom/train/images/",
                "train_annotations": "./datasets/custom/train/annotations.json",
                "val_images": "./datasets/custom/val/images/",
                "val_annotations": "./datasets/custom/val/annotations.json"
            }
        },
        "model_config": {
            "num_classes": "Total number of classes INCLUDING background",
            "class_names": ["background", "class1", "class2", "..."],
            "architecture": "faster_rcnn",
            "pretrained": True
        },
        "training_config": {
            "learning_rate": 0.005,
            "batch_size": 2,
            "num_epochs": 10,
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "step_size": 3,
            "gamma": 0.1
        },
        "output_config": {
            "output_dir": "./outputs/detection",
            "model_save_path": "./outputs/detection/model.pth"
        }
    }
    
    import json
    with open('sample_detection_config.json', 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print("Sample configuration saved to 'sample_detection_config.json'")


if __name__ == "__main__":
    # Uncomment to create a sample configuration file
    # create_sample_config()
    
    # Run the main example
    main()

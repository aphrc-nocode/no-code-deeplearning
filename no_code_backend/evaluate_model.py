"""
Model evaluation utilities for the no-code backend.
This module provides functions to evaluate trained models on test datasets.
"""

import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Union

def evaluate_classification_model(model, dataset_path: Union[str, Path], batch_size: int = 8, job_id: str = None) -> Dict[str, Any]:
    """
    Evaluate a classification model on a test dataset
    
    Args:
        model: The model to evaluate
        dataset_path: Path to the test dataset
        batch_size: Batch size for evaluation
        job_id: Optional job ID for loading saved splits
    
    Returns:
        Dict with evaluation metrics
    """
    from datasets_module.classification.dataloaders import create_dataloaders, ImageClassificationDataset
    from metrics.classification.metrics import calculate_classification_metrics
    from torchvision import transforms
    import torch
    
    # Set model to evaluation mode
    model.eval()
    
    # Get preprocessing transforms for evaluation
    # Use standard normalization for pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Evaluation transforms - center crop without augmentation
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Create dataloader - first check if there's a dedicated test directory
    dataset_path = Path(dataset_path)
    test_dir = dataset_path / "test"
    
    if test_dir.exists():
        print(f"Found test directory: {test_dir}")
        # Use the test directory for evaluation
        _, _, test_loader, classes = create_dataloaders(
            test_dir,
            transform=eval_transform,
            batch_size=batch_size,
            val_split=0.0,  # No need to split test data further
            test_split=0.0   # Use all data for testing
        )
    elif job_id:
        # Try to load saved splits if job_id is provided
        try:
            print(f"Looking for saved splits from training job: {job_id}")
            # Create dataloaders using the saved test split
            _, _, test_loader, classes = create_dataloaders(
                dataset_path, 
                transform=eval_transform,
                batch_size=batch_size,
                job_id=job_id,         # Use job_id to find splits
                use_saved_splits=True,  # Use the persistent splits
                shuffle=False
            )
            print(f"Using saved test split with {len(test_loader.dataset)} samples")
        except FileNotFoundError:
            # Fall back to standard approach
            print(f"No saved splits found. Falling back to standard approach.")
            test_loader = None  # Will be set in the next block
    else:
        # No test directory or job_id provided
        test_loader = None
        
    # If we still don't have a test loader, follow the previous approach
    if test_loader is None:
        # Look for val directory as second option
        val_dir = dataset_path / "val"
        if val_dir.exists():
            print(f"Found validation directory: {val_dir}")
            # Use the validation directory for evaluation
            _, _, test_loader, classes = create_dataloaders(
                val_dir,
                transform=eval_transform,
                batch_size=batch_size,
                val_split=0.0,  # No need to split val data further
                test_split=0.0   # Use all data for testing
            )
        else:
            # If no test or val directory, create a 80/10/10 split from the main dataset
            # and use the test split for evaluation
            print("No dedicated test or validation directory found. Creating a temporary test split.")
            train_loader, val_loader, test_loader, classes = create_dataloaders(
                dataset_path, 
                transform=eval_transform,
                batch_size=batch_size,
                val_split=0.1,  # Use 10% of data for validation
                test_split=0.1   # Use 10% of data for testing
            )
            
            if test_loader is None:
                # In case the dataset is too small to create a test split,
                # use the validation data or training data as fallback
                if val_loader is not None:
                    print("WARNING: Dataset too small to create a test split. Using validation data for evaluation.")
                    test_loader = val_loader
                else:
                    print("WARNING: Dataset too small to create splits. Using training data for evaluation.")
                    test_loader = train_loader
    
    # Print info about the test loader
    num_samples = len(test_loader.dataset) if hasattr(test_loader, "dataset") else "unknown"
    num_batches = len(test_loader) if hasattr(test_loader, "__len__") else "unknown"
    print(f"Evaluating on dataset with {num_samples} samples, {num_batches} batches")
    
    # Ensure model is on the correct device
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Lists to collect predictions and targets
    all_preds = []
    all_targets = []
    all_scores = []
    
    # Counter for tracking progress
    batch_count = 0
    
    with torch.no_grad():
        try:
            for inputs, targets in test_loader:
                batch_count += 1
                print(f"Processing batch {batch_count}/{num_batches}")
                
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Run forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                scores = torch.softmax(outputs, dim=1)
                
                # Add predictions and targets for metric calculation
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
                
                print(f"Batch {batch_count}: Processed {len(inputs)} samples")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Calculating metrics on {len(all_preds)} predictions")
    
    # Calculate metrics
    if len(all_preds) > 0:
        try:
            metrics = calculate_classification_metrics(
                y_true=np.array(all_targets),
                y_pred=np.array(all_preds),
                y_scores=np.array(all_scores)
            )
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            metrics = {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "error": f"Error calculating metrics: {str(e)}"
            }
    else:
        print("WARNING: No predictions to evaluate. Using default metrics.")
        metrics = {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "error": "No predictions available for evaluation"
        }
    
    # Add number of test samples to the metrics
    metrics['num_test_samples'] = len(all_preds)
    metrics['num_test_batches'] = batch_count
    
    # Print class distribution of predictions
    if len(all_preds) > 0:
        from collections import Counter
        pred_distribution = Counter(all_preds)
        target_distribution = Counter(all_targets)
        print(f"Prediction class distribution: {dict(pred_distribution)}")
        print(f"Target class distribution: {dict(target_distribution)}")
    
    return metrics

def evaluate_detection_model(model, dataset_path: Union[str, Path], batch_size: int = 4) -> Dict[str, Any]:
    """
    Evaluate an object detection model on a test dataset
    
    Args:
        model: The model to evaluate
        dataset_path: Path to the test dataset
        batch_size: Batch size for evaluation
        
    Returns:
        Dict with evaluation metrics
    """
    from datasets_module.detection.dataloaders import create_dataloaders
    from datasets_module.detection.transforms import get_detection_transforms
    from metrics.detection.metrics import calculate_detection_metrics
    import torch
    
    # Set model to evaluation mode
    model.eval()
    
    # Get detection transforms for test set (evaluation mode)
    try:
        transform = get_detection_transforms(train=False)
        print("Using evaluation transforms for detection model")
    except Exception as e:
        print(f"Error getting transforms: {e}. Using default transforms.")
        # If there's an issue with the transforms, try without them
        transform = None
    
    # Create dataloader - first check if there's a dedicated test directory
    dataset_path = Path(dataset_path)
    test_dir = dataset_path / "test"
    
    if test_dir.exists():
        print(f"Found test directory: {test_dir}")
        # Use the test directory for evaluation
        train_loader, test_loader, classes = create_dataloaders(
            test_dir,
            transform=transform,
            batch_size=batch_size,
            val_split=0.0,  # No need to split test data further
        )
        # Use the training loader for test since val_split=0.0
        test_loader = train_loader
    else:
        # Look for val directory instead as second option
        val_dir = dataset_path / "val"
        if val_dir.exists():
            print(f"Found validation directory: {val_dir}")
            # Use the validation directory for evaluation
            train_loader, test_loader, classes = create_dataloaders(
                val_dir,
                transform=transform,
                batch_size=batch_size,
                val_split=0.0,  # No need to split val data further
            )
            # Use the training loader for test since val_split=0.0
            test_loader = train_loader
        else:
            # If no test or val directory, create a validation split from the main dataset
            print("No dedicated test or validation directory found. Creating validation split from dataset.")
            train_loader, test_loader, classes = create_dataloaders(
                dataset_path,
                transform=transform,
                batch_size=batch_size,
                val_split=0.2,  # Create validation split for testing
            )
            
            # If even with splitting we couldn't get a validation set, use a small portion of training data
            if test_loader is None:
                print("WARNING: No validation set could be created. Using the training set for evaluation.")
                # Just use the training data for evaluation in this case
                test_loader = train_loader
                
                # We could alternatively create a manual split, but there's likely a reason the dataset
                # couldn't be split correctly (e.g., too few samples)
                '''
                from torch.utils.data import random_split
                
                # Get the dataset from the train_loader
                train_dataset = train_loader.dataset
                test_size = max(1, int(0.2 * len(train_dataset)))  # Ensure at least 1 sample
                train_size = len(train_dataset) - test_size
                
                print(f"Manually splitting train dataset: {len(train_dataset)} total samples -> {train_size} train, {test_size} test")
                
                # Create a random split
                train_subset, test_subset = random_split(
                    train_dataset, [train_size, test_size]
                )
                
                # Create a new dataloader for testing
                from torch.utils.data import DataLoader
                test_loader = DataLoader(
                    test_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=train_loader.collate_fn
                )
                '''
    
    try:
        # This is just placeholder to keep the try-except structure consistent
        pass
        
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise
    
    # Ensure model is on the correct device
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    
    # Lists to collect predictions and targets
    all_detections = []
    all_targets = []
    
    # Print info about the test loader
    num_batches = len(test_loader) if hasattr(test_loader, "__len__") else "unknown"
    print(f"Evaluating on test dataset with {num_batches} batches")
    
    # Counter for tracking progress
    batch_count = 0
    
    with torch.no_grad():
        try:
            for images, targets in test_loader:
                batch_count += 1
                print(f"Processing batch {batch_count}")
                
                # Move data to device
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Run forward pass
                outputs = model(images)
                
                # Check if we got valid outputs
                print(f"Batch {batch_count}: Found {len(outputs)} predictions")
                
                # Add detections and targets for metric calculation
                all_detections.extend(outputs)
                all_targets.extend(targets)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate metrics
    print(f"Calculating metrics on {len(all_detections)} predictions")
    
    if len(all_detections) > 0:
        metrics = calculate_detection_metrics(all_detections, all_targets)
    else:
        print("WARNING: No predictions to evaluate. Using default metrics.")
        metrics = {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "error": "No predictions available for evaluation"
        }
    
    # Add number of test samples to the metrics
    metrics['num_test_samples'] = len(all_detections)
    metrics['num_test_batches'] = batch_count
    
    return metrics

def evaluate_segmentation_model(model, dataset_path: Union[str, Path], batch_size: int = 4) -> Dict[str, Any]:
    """
    Evaluate an image segmentation model on a test dataset
    
    Args:
        model: The model to evaluate
        dataset_path: Path to the test dataset
        batch_size: Batch size for evaluation
    
    Returns:
        Dict with evaluation metrics
    """
    from datasets_module.segmentation.dataloaders import create_dataloaders
    from metrics.segmentation.metrics import calculate_segmentation_metrics
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader
    _, test_loader, classes = create_dataloaders(
        dataset_path, 
        transform=None,  # Default transform will be used
        batch_size=batch_size,
        val_split=0.0,  # Use all data for testing
    )
    
    device = next(model.parameters()).device
    
    # Lists to collect predictions and targets
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(masks.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_segmentation_metrics(
        np.array(all_targets),
        np.array(all_preds),
        len(classes)
    )
    
    return metrics

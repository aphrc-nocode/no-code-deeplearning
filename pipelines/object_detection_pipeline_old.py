"""
Pipeline for object detection tasks.
This module implements the BasePipeline interface for object detection.
Supports both PyTorch-based detection models and Hugging Face transformers.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import mlflow
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
from torchmetrics.detection import MeanAveragePrecision
from transformers.utils import ModelOutput

from pipelines.base_pipeline import BasePipeline
from models.detection import model_factory
from datasets_module.detection.dataloaders import create_dataloaders
from metrics.detection.metrics import calculate_detection_metrics
from mlflow_utils import log_model, start_run, log_metrics, log_batch_metrics, end_run

# Import Hugging Face transformers
try:
    import transformers
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    from detection_dataset_builder import build_hf_detection_dataset, split_dataset
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: Hugging Face transformers not available. Install with pip install transformers")
    
# Hugging Face model checkpoints for object detection
HF_MODEL_CHECKPOINTS = {
    "facebook/detr-resnet-50": "DETR with ResNet-50 backbone",
    "facebook/detr-resnet-101": "DETR with ResNet-101 backbone",
    "hustvl/yolos-small": "YOLOS small model",
    "hustvl/yolos-base": "YOLOS base model",
    "hustvl/yolos-tiny": "YOLOS tiny model",
    "owlv2-base-patch16-ensemble": "OWLv2 base model",
}

# Helper to convert YOLO to Pascal VOC format

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert YOLO format bounding boxes to Pascal VOC format.
    YOLO: [x_center, y_center, width, height] (normalized)
    Pascal: [x_min, y_min, x_max, y_max] (absolute coordinates)
    """
    height, width = image_size
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height
    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2
    x_max = x_center + box_width / 2
    y_max = y_center + box_height / 2
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

class ObjectDetectionPipeline(BasePipeline):
    """Pipeline for object detection tasks"""
    
    def __init__(self, config):
        """Initialize the pipeline with configuration
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__(config)
        self.is_hf_model = hasattr(self.config, 'use_hf_transformers') and self.config.use_hf_transformers
        self.hf_model_checkpoint = getattr(self.config, 'hf_model_checkpoint', 'facebook/detr-resnet-50')
    
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train the model with the given dataset"""
        try:
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.dict())
            
            # Use Hugging Face transformer models if specified
            if self.is_hf_model and HF_AVAILABLE:
                return await self._train_with_hf_transformers(dataset_path, job_id)
            # Otherwise use standard PyTorch models
            
            # Create model and setup training
            model = self.create_model()
            
            # Object detection uses different loss functions based on the architecture
            # They're typically computed within the model's forward pass
            
            # Get optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(params, lr=self.config.learning_rate, 
                                 momentum=0.9, weight_decay=0.0005)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=3, 
                                                gamma=0.1)
            
            # Log optimizer configuration
            log_metrics({
                "initial_learning_rate": self.config.learning_rate
            })
            
            # Create datasets and data loaders with persistent splits
            transform = self.get_transforms()
            
            # Analyze dataset structure before creating dataloaders
            dataset_path = Path(dataset_path)
            has_presplit_structure = False
            
            # Check for common pre-split dataset structures
            if (dataset_path / "train").exists() or \
               ((dataset_path / "images").exists() and (dataset_path / "annotations").exists() and 
                (dataset_path / "annotations" / "instances_train.json").exists()):
                await self._update_job_log(job_id, f"Detected pre-split dataset structure in {dataset_path}")
                has_presplit_structure = True
            
            # Use saved splits for validation/evaluation, but create new splits for training
            await self._update_job_log(job_id, f"Creating dataloaders for dataset: {dataset_path}")
            train_loader, val_loader, test_loader, classes = create_dataloaders(
                dataset_path, transform, 
                batch_size=self.config.batch_size,
                val_split=0.2 if self.config.early_stopping else 0.1,
                test_split=0.1,
                job_id=job_id,  # Pass job_id to ensure splits persistence
                use_saved_splits=False  # Create new splits for this training run
            )
            
            # Store class mapping
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            
            # Log dataset information
            await self._update_job_log(job_id, f"Dataset loaded with {len(classes)} classes: {classes}")
            await self._update_job_log(job_id, f"Training on {len(train_loader.dataset)} samples")
            if val_loader:
                await self._update_job_log(job_id, f"Validating on {len(val_loader.dataset)} samples")
            
            # Initialize tracking variables
            best_loss = float('inf')
            best_map = 0.0
            best_model_state = None
            patience_counter = 0
            
            # Track metrics for visualization
            train_losses = []
            train_maps = []
            val_losses = []
            val_maps = []
            
            # Training loop
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_detections = []
                train_targets = []
                
                for batch_idx, (images, targets) in enumerate(train_loader):
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += losses.item()
                    
                    # Collect detections and targets for MAP calculation
                    with torch.no_grad():
                        # Make sure the model is in eval mode for inference
                        model.eval()
                        detections = model(images)
                        train_detections.extend(detections)
                        train_targets.extend(targets)
                        # Set model back to training mode
                        model.train()
                    
                    # Log batch progress every 10 batches
                    if batch_idx % 10 == 0:
                        await self._update_job_log(
                            job_id, 
                            f"Epoch {epoch+1}/{self.config.epochs} - "
                            f"Batch {batch_idx+1}/{len(train_loader)} - "
                            f"Loss: {losses.item():.4f}"
                        )
                
                train_loss /= len(train_loader)
                
                # Calculate mAP for training set
                try:
                    metrics_result = calculate_detection_metrics(train_detections, train_targets)
                    train_map = metrics_result.get("mAP", 0.0)
                except Exception as e:
                    await self._update_job_log(job_id, f"Warning: Could not calculate training mAP: {str(e)}")
                    train_map = 0.0
                
                # Store metrics for visualization
                train_losses.append(train_loss)
                train_maps.append(train_map)
                
                # Update learning rate
                scheduler.step()
                
                # Validation phase
                val_loss = 0.0
                val_map = 0.0
                
                if val_loader:
                    model.eval()
                    val_detections = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for images, targets in val_loader:
                            images = list(image.to(self.device) for image in images)
                            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                            
                            # Compute loss
                            loss_dict = model(images, targets)
                            losses = sum(loss for loss in loss_dict.values())
                            
                            val_loss += losses.item()
                            
                            # Get detections (already in evaluation mode)
                            detections = model(images)
                            val_detections.extend(detections)
                            val_targets.extend(targets)
                    
                    val_loss /= len(val_loader)
                    
                    # Calculate mAP for validation set
                    try:
                        val_metrics = calculate_detection_metrics(val_detections, val_targets)
                        val_map = val_metrics.get("mAP", 0.0)
                    except Exception as e:
                        await self._update_job_log(job_id, f"Warning: Could not calculate validation mAP: {str(e)}")
                        val_metrics = {"mAP": 0.0}
                        val_map = 0.0
                    
                    # Store metrics for visualization
                    val_losses.append(val_loss)
                    val_maps.append(val_map)
                    
                    # Log metrics
                    log_metrics({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_mAP": train_map,
                        "val_loss": val_loss,
                        "val_mAP": val_map,
                        **val_metrics
                    })
                    
                    # Early stopping based on validation metrics
                    if self.config.early_stopping:
                        if val_map > best_map:
                            best_loss = val_loss
                            best_map = val_map
                            best_model_state = model.state_dict()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.config.patience:
                                await self._update_job_log(job_id, f"Early stopping at epoch {epoch+1}")
                                break
                else:
                    # If no validation set, use training metrics
                    if train_map > best_map:
                        best_loss = train_loss
                        best_map = train_map
                        best_model_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience and self.config.early_stopping:
                            await self._update_job_log(job_id, f"Early stopping at epoch {epoch+1}")
                            break
            
            # Create visualization plots
            from mlflow_utils import log_training_visualizations
            
            # Log training curves - customize for object detection metrics
            # TODO: Implement visualization for object detection metrics
            
            # Save best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Create the target directory for MLflow models
            mlflow_models_dir = Path("logs/mlflow/models")
            mlflow_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save class mapping separately for easier loading
            class_mapping_path = Path("class_mapping.json")  # Temporary file
            import json
            with open(class_mapping_path, 'w') as f:
                json.dump(class_to_idx, f)
            
            # Log class mapping as artifact
            mlflow.log_artifact(str(class_mapping_path), "metadata")
            
            # Clean up the temporary file
            if class_mapping_path.exists():
                class_mapping_path.unlink()                # Log model directly to MLflow
                try:
                    mlflow.pytorch.log_model(model, artifact_path="model")
                    await self._update_job_log(job_id, "Model logged to MLflow successfully")
                except Exception as e:
                    await self._update_job_log(job_id, f"Error logging model to MLflow: {str(e)}")
            
            # Get MLflow run info to use as model path
            run_info = mlflow.active_run()
            local_model_path = None
            
            if run_info:
                # Use our custom MLflow models directory
                run_id = run_info.info.run_id
                model_output_path = mlflow_models_dir / run_id
                model_output_path.mkdir(parents=True, exist_ok=True)
                
                # Save a copy of the model to our custom directory
                final_model_path = model_output_path / "model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'config': self.config.dict()
                }, final_model_path)
                
                # Use the actual model file path, not the directory
                local_model_path = str(final_model_path)
            else:
                # Fallback to local storage if MLflow run isn't active
                os.makedirs("models", exist_ok=True)
                local_model_path = f"models/{job_id}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'config': self.config.dict()
                }, local_model_path)
            
            # End MLflow run
            mlflow_run_id = self.run_id
            end_run()
            
            # Get MLflow model URI for the saved model
            mlflow_model_uri = f"runs:/{mlflow_run_id}/model" if mlflow_run_id else None
            
            await self._update_job_log(job_id, f"Training completed successfully. Model saved to {local_model_path}")
            # Return a dictionary with status and model path
            return {
                "status": "completed",
                "model_path": local_model_path,
                "mlflow_model_uri": mlflow_model_uri,
                "mlflow_run_id": mlflow_run_id,
                "metrics": {
                    "final_train_loss": train_loss,
                    "final_train_mAP": train_map,
                    "final_val_loss": val_loss if val_loader else None,
                    "final_val_mAP": val_map if val_loader else None,
                }
            }
            
        except Exception as e:
            # End MLflow run in case of error
            from mlflow_utils import end_run
            end_run()
            await self._update_job_log(job_id, f"Error during training: {str(e)}")
            # Return error status
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _train_with_hf_transformers(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """
        Train a Hugging Face transformers-based object detection model
        
        Args:
            dataset_path: Path to the dataset
            job_id: Job ID for tracking
            
        Returns:
            Dictionary with training results
        """
        try:
            await self._update_job_log(job_id, f"Starting training with Hugging Face model: {self.hf_model_checkpoint}")
            
            # Get the image processor for preprocessing
            image_processor = AutoImageProcessor.from_pretrained(self.hf_model_checkpoint)
            
            # Find annotations files - check for pre-split datasets
            dataset_path = Path(dataset_path)
            train_annotations_path = None
            val_annotations_path = None
            test_annotations_path = None
            
            # Check for pre-split dataset structure
            using_presplit_data = False
            
            # Check for train split
            train_dir = dataset_path / "train"
            if train_dir.exists() and train_dir.is_dir():
                # Look for annotations in train directory
                train_anno_files = list(train_dir.glob("*annotations*.json"))
                if train_anno_files:
                    train_annotations_path = train_anno_files[0]
                    await self._update_job_log(job_id, f"Found train annotations at: {train_annotations_path}")
                    using_presplit_data = True
            
            # Check for validation split
            val_dir = dataset_path / "validation"
            if not val_dir.exists():
                val_dir = dataset_path / "valid"  # Alternative name
            
            if val_dir.exists() and val_dir.is_dir():
                val_anno_files = list(val_dir.glob("*annotations*.json"))
                if val_anno_files:
                    val_annotations_path = val_anno_files[0]
                    await self._update_job_log(job_id, f"Found validation annotations at: {val_annotations_path}")
            
            # Check for test split
            test_dir = dataset_path / "test"
            if test_dir.exists() and test_dir.is_dir():
                test_anno_files = list(test_dir.glob("*annotations*.json"))
                if test_anno_files:
                    test_annotations_path = test_anno_files[0]
                    await self._update_job_log(job_id, f"Found test annotations at: {test_annotations_path}")
            
            # If we're not using pre-split data, fall back to single annotation file
            if not using_presplit_data:
                annotations_path = None
                
                # Look for annotations in standard locations
                potential_annotations = [
                    dataset_path / "annotations" / "instances_train.json",
                    dataset_path / "annotations.json",
                ]
                
                for path in potential_annotations:
                    if path.exists():
                        annotations_path = path
                        break
                
                # If still not found, try to find any JSON file with "annotations" in name
                if annotations_path is None:
                    json_files = list(dataset_path.glob("**/*.json"))
                    for path in json_files:
                        if "annotations" in path.name.lower() or "instances" in path.name.lower():
                            annotations_path = path
                            break
                
                if annotations_path is None:
                    raise ValueError(f"Could not find annotations file in {dataset_path}")
                
                await self._update_job_log(job_id, f"Found annotations at: {annotations_path}")
                train_annotations_path = annotations_path
            
            # Initialize datasets
            train_dataset = None
            val_dataset = None
            test_dataset = None
            class_names = []
            
            # Build datasets based on what we found
            if using_presplit_data:
                await self._update_job_log(job_id, "Using pre-split dataset structure")
                
                # Process train split
                train_images_dir = train_dir
                if (train_dir / "images").exists():
                    train_images_dir = train_dir / "images"
                
                await self._update_job_log(job_id, f"Processing train split from: {train_images_dir}")
                train_dataset, class_names = build_hf_detection_dataset(
                    images_dir=train_images_dir,
                    annotations_path=train_annotations_path,
                    image_processor=image_processor
                )
                
                # Process validation split if available
                if val_annotations_path:
                    val_images_dir = val_dir
                    if (val_dir / "images").exists():
                        val_images_dir = val_dir / "images"
                    
                    await self._update_job_log(job_id, f"Processing validation split from: {val_images_dir}")
                    val_dataset, _ = build_hf_detection_dataset(
                        images_dir=val_images_dir,
                        annotations_path=val_annotations_path,
                        image_processor=image_processor
                    )
                
                # Process test split if available
                if test_annotations_path:
                    test_images_dir = test_dir
                    if (test_dir / "images").exists():
                        test_images_dir = test_dir / "images"
                    
                    await self._update_job_log(job_id, f"Processing test split from: {test_images_dir}")
                    test_dataset, _ = build_hf_detection_dataset(
                        images_dir=test_images_dir,
                        annotations_path=test_annotations_path,
                        image_processor=image_processor
                    )
                
                # If val or test split not found, create it from train
                if val_dataset is None or test_dataset is None:
                    await self._update_job_log(job_id, "Some splits missing, creating from train dataset")
                    
                    # Set defaults for missing splits
                    if val_dataset is None and test_dataset is None:
                        # Both missing - create both from train
                        temp_train, val_dataset, test_dataset = split_dataset(train_dataset, val_split=0.1, test_split=0.1)
                        if len(temp_train) >= len(train_dataset) * 0.7:  # Only replace if we didn't lose too much data
                            train_dataset = temp_train
                    elif val_dataset is None:
                        # Only val missing - create from train
                        temp_train, val_dataset, _ = split_dataset(train_dataset, val_split=0.1, test_split=0)
                        if len(temp_train) >= len(train_dataset) * 0.85:  # Only replace if we didn't lose too much data
                            train_dataset = temp_train
                    elif test_dataset is None:
                        # Only test missing - use val for test too
                        test_dataset = val_dataset
            else:
                # Determine images directory for single dataset
                if (dataset_path / "images").exists():
                    images_dir = dataset_path / "images"
                else:
                    # Assume images are directly in the dataset directory
                    images_dir = dataset_path
                
                await self._update_job_log(job_id, f"Using images from: {images_dir}")
                
                # Build dataset
                await self._update_job_log(job_id, "Converting dataset to Hugging Face format...")
                dataset, class_names = build_hf_detection_dataset(
                    images_dir=images_dir,
                    annotations_path=train_annotations_path,
                    image_processor=image_processor
                )
                
                # Split dataset
                await self._update_job_log(job_id, f"Splitting single dataset into train/val/test")
                train_dataset, val_dataset, test_dataset = split_dataset(
                    dataset, 
                    val_split=0.2, 
                    test_split=0.1
                )
            
            # Log dataset split sizes
            train_size = len(train_dataset) if train_dataset else 0
            val_size = len(val_dataset) if val_dataset else 0
            test_size = len(test_dataset) if test_dataset else 0
            
            await self._update_job_log(job_id, 
                f"Dataset split: {train_size} train, {val_size} val, {test_size} test samples"
            )
            
            # Handle very small datasets - ensure we have enough data to train
            if train_size < 2:
                await self._update_job_log(job_id, "Dataset too small for training. Combining all available data.")
                # Combine all available data for training
                train_dataset = val_dataset if train_size == 0 and val_size > 0 else train_dataset
                if test_size > 0:
                    if train_size == 0:
                        train_dataset = test_dataset
                
                # Use the same data for validation if needed
                if val_size == 0:
                    val_dataset = train_dataset
                
                # Use the same data for testing if needed
                if test_dataset is None or test_size == 0:
                    test_dataset = val_dataset
            
            # Initialize model
            await self._update_job_log(job_id, f"Initializing model from {self.hf_model_checkpoint}")
            model = AutoModelForObjectDetection.from_pretrained(
                self.hf_model_checkpoint,
                num_labels=len(class_names),
                ignore_mismatched_sizes=True
            )
            
            # Define training arguments
            from transformers import TrainingArguments
            
            # Create output directory in logs
            output_dir = f"logs/mlflow/models/{job_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                num_train_epochs=self.config.epochs,
                learning_rate=self.config.learning_rate,
                weight_decay=0.0005,
                logging_dir=f"logs/mlflow/logs/{job_id}",
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                push_to_hub=False,
                report_to="mlflow",
            )
            
            # Define data collator for object detection
            def collate_fn(batch):
                """Custom collate function for object detection"""
                # Ensure each pixel_values is a tensor
                pixel_values = [torch.tensor(item["pixel_values"]) if not isinstance(item["pixel_values"], torch.Tensor) else item["pixel_values"] for item in batch]
                pixel_values = torch.stack(pixel_values)

                # Process labels - ensure all values are tensors
                labels = []
                for item in batch:
                    label_dict = item["labels"]
                    processed_label = {}
                    for key, value in label_dict.items():
                        if isinstance(value, list):
                            processed_label[key] = torch.tensor(value)
                        elif not isinstance(value, torch.Tensor):
                            processed_label[key] = torch.tensor(value)
                        else:
                            processed_label[key] = value
                    labels.append(processed_label)

                return {
                    "pixel_values": pixel_values,
                    "labels": labels
                }
            
            # Import the improved metrics calculation
            from metrics.detection.hf_metrics import create_compute_metrics_function
            
            # Create class index to name mapping
            id2label = {i: name for i, name in enumerate(class_names)} if class_names else None
            
            # Create the compute metrics function
            compute_metrics = create_compute_metrics_function(
                model=model,
                image_processor=image_processor,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                id2label=id2label,
                threshold=0.1  # Lower threshold for small datasets
            )
            
            # Initialize trainer
            # Always enable evaluation, even for very small datasets
            trainer = transformers.Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
            )
            
            # Train model
            await self._update_job_log(job_id, "Starting training...")
            trainer.train()
            
            # Evaluate model if applicable
            eval_results = {}
            if training_args.eval_strategy != "no" and trainer.eval_dataset is not None:
                await self._update_job_log(job_id, "Evaluating model...")
                try:
                    eval_results = trainer.evaluate()
                except Exception as e:
                    await self._update_job_log(job_id, f"Warning: Evaluation failed: {str(e)}")
                    eval_results = {"warning": "Evaluation skipped due to error"}
            else:
                await self._update_job_log(job_id, "Skipping evaluation (disabled or no validation data available)")
            
            # Save model and processor
            await self._update_job_log(job_id, f"Saving model to {output_dir}")
            trainer.save_model(output_dir)
            image_processor.save_pretrained(output_dir)
            
            # Save class names
            import json
            with open(f"{output_dir}/class_names.json", "w") as f:
                json.dump(class_names, f)
            
            # Log model to MLflow
            try:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": model, 
                        "image_processor": image_processor
                    },
                    artifact_path="model"
                )
                await self._update_job_log(job_id, "Model logged to MLflow successfully")
            except Exception as e:
                await self._update_job_log(job_id, f"Error logging model to MLflow: {str(e)}")
            
            # Get MLflow run info
            run_info = mlflow.active_run()
            mlflow_run_id = run_info.info.run_id if run_info else None
            
            # End MLflow run
            end_run()
            
            # Return results
            return {
                "status": "completed",
                "model_path": output_dir,
                "mlflow_run_id": mlflow_run_id,
                "mlflow_model_uri": f"runs:/{mlflow_run_id}/model" if mlflow_run_id else None,
                "metrics": eval_results,
                "class_names": class_names
            }
        
        except Exception as e:
            # Handle errors
            await self._update_job_log(job_id, f"Error in HF transformer training: {str(e)}")
            import traceback
            await self._update_job_log(job_id, traceback.format_exc())
            
            # End MLflow run in case of error
            end_run()
            
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def create_model(self) -> nn.Module:
        """Create a model based on the architecture specified in config"""
        # Handle Hugging Face models
        if self.is_hf_model and HF_AVAILABLE:
            # For HF models, we'll create them during training
            return None
        
        # Get model from factory
        model = model_factory.create_model(
            str(self.config.architecture.value),  # Convert enum to string
            num_classes=self.config.num_classes,
            pretrained=True
        )
        
        # Move to device if the model is not None
        if model is not None:
            model = model.to(self.device)
            
        return model
    
    def get_transforms(self):
        """Get data transforms for object detection"""
        from datasets_module.detection.transforms import get_detection_transforms
        return get_detection_transforms(train=True)
        
    async def predict(self, image, model=None) -> Dict[str, Any]:
        """Make a prediction using the trained model"""
        if model is None:
            raise ValueError("Model must be provided for prediction")
            
        # Check if this is a HF model
        is_hf_model = hasattr(model, 'config') and hasattr(model.config, 'model_type') 
        
        if is_hf_model and HF_AVAILABLE:
            return await self._predict_with_hf_model(image, model)
        
        # Standard PyTorch model prediction
        # Apply inference transforms to the image
        from datasets_module.detection.transforms import get_inference_transforms
        transform = get_inference_transforms()
        image_tensor = transform(image).to(self.device)  # Don't add batch dimension here
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            detections = model([image_tensor])  # Pass as list of 3D tensors
            
            # Format the results
            boxes = detections[0]['boxes'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            labels = detections[0]['labels'].cpu().numpy()
            
            return {
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "labels": labels.tolist()
            }
            
    async def _predict_with_hf_model(self, image, model) -> Dict[str, Any]:
        """Make a prediction using a Hugging Face transformer model"""
        # Load image processor
        model_path = getattr(model, '_model_path', None)
        if model_path and Path(model_path).exists():
            image_processor = AutoImageProcessor.from_pretrained(model_path)
        else:
            # Attempt to find image processor from model name
            model_checkpoint = self.hf_model_checkpoint
            image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        
        # Process image
        inputs = image_processor(images=image, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            model.eval()
            outputs = model(**inputs)
            
        # Process outputs based on model type
        if hasattr(outputs, "pred_boxes") and hasattr(outputs, "pred_logits"):
            # DETR-style output format
            pred_boxes = outputs.pred_boxes[0].cpu().numpy()
            pred_labels = outputs.pred_logits[0].argmax(dim=-1).cpu().numpy()
            pred_scores = outputs.pred_logits[0].softmax(dim=-1).max(dim=-1).values.cpu().numpy()
            
            # Filter out predictions with low confidence
            confidence_threshold = 0.5
            high_conf_mask = pred_scores > confidence_threshold
            pred_boxes = pred_boxes[high_conf_mask]
            pred_labels = pred_labels[high_conf_mask]
            pred_scores = pred_scores[high_conf_mask]
            
        elif hasattr(outputs, "prediction_logits") and hasattr(outputs, "pred_boxes"):
            # Alternative DETR format
            pred_boxes = outputs.pred_boxes[0].cpu().numpy()
            pred_labels = outputs.prediction_logits[0].argmax(dim=-1).cpu().numpy()
            pred_scores = outputs.prediction_logits[0].softmax(dim=-1).max(dim=-1).values.cpu().numpy()
            
            # Filter out predictions with low confidence
            confidence_threshold = 0.5
            high_conf_mask = pred_scores > confidence_threshold
            pred_boxes = pred_boxes[high_conf_mask]
            pred_labels = pred_labels[high_conf_mask]
            pred_scores = pred_scores[high_conf_mask]
            
        elif hasattr(outputs, "last_hidden_state"):
            # Handle models that return raw logits
            # This is a fallback for models with non-standard output formats
            pred_boxes = np.array([])
            pred_labels = np.array([])
            pred_scores = np.array([])
            
        else:
            # Try to extract from logits and boxes if available
            if hasattr(outputs, "logits") and hasattr(outputs, "pred_boxes"):
                pred_boxes = outputs.pred_boxes[0].cpu().numpy()
                pred_labels = outputs.logits[0].argmax(dim=-1).cpu().numpy()
                pred_scores = outputs.logits[0].softmax(dim=-1).max(dim=-1).values.cpu().numpy()
                
                # Filter out predictions with low confidence
                confidence_threshold = 0.5
                high_conf_mask = pred_scores > confidence_threshold
                pred_boxes = pred_boxes[high_conf_mask]
                pred_labels = pred_labels[high_conf_mask]
                pred_scores = pred_scores[high_conf_mask]
            else:
                # Default format or unknown format
                pred_boxes = outputs.boxes[0].cpu().numpy() if hasattr(outputs, "boxes") else np.array([])
                pred_labels = outputs.labels[0].cpu().numpy() if hasattr(outputs, "labels") else np.array([])
                pred_scores = outputs.scores[0].cpu().numpy() if hasattr(outputs, "scores") else np.array([])
        
        return {
            "boxes": pred_boxes.tolist(),
            "scores": pred_scores.tolist(),
            "labels": pred_labels.tolist()
        }
    
    async def evaluate(self, dataset_path: str, model_path: str = None, job_id: str = None) -> Dict[str, Any]:
        """
        Evaluate the model on a test dataset
        
        Args:
            dataset_path: Path to test dataset
            model_path: Path to saved model (if None, will use self.model)
            job_id: Optional job ID for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Load model if path is provided
            if model_path:
                checkpoint = torch.load(model_path)
                model = self.create_model()
                model.load_state_dict(checkpoint['model_state_dict'])
                class_to_idx = checkpoint.get('class_to_idx', {})
            else:
                # Ensure we have a model to evaluate
                raise ValueError("Model path must be provided for evaluation")
                
            # Set model to evaluation mode
            model.eval()
            
            # Get transforms for evaluation
            transform = self.get_transforms()
            
            # Check if we have a test directory
            test_dir = Path(dataset_path) / "test"
            if test_dir.exists():
                print(f"Using dedicated test directory: {test_dir}")
                # Use the test directory exclusively
                _, _, test_loader, classes = create_dataloaders(
                    str(test_dir), transform,
                    batch_size=self.config.batch_size,
                    val_split=0.0,  # No further splitting needed
                    test_split=0.0,  # No further splitting needed
                    shuffle=False    # No need to shuffle for evaluation
                )
            else:
                # No test directory, try to load saved splits
                try:
                    print(f"Looking for saved splits from training job: {job_id}")
                    # Create a loader using the saved test split
                    _, _, test_loader, classes = create_dataloaders(
                        dataset_path, transform,
                        batch_size=self.config.batch_size,
                        job_id=job_id,             # Use job_id to find splits
                        use_saved_splits=True,     # Use the persistent splits
                        shuffle=False
                    )
                    print(f"Using saved test split with {len(test_loader.dataset)} samples")
                except Exception as e:
                    print(f"Could not load saved splits: {e}")
                    print("Falling back to validation split approach")
                    # Fallback to using validation directory or creating new split
                    val_dir = Path(dataset_path) / "val"
                    if val_dir.exists():
                        print(f"Using validation directory: {val_dir}")
                        _, _, test_loader, classes = create_dataloaders(
                            str(val_dir), transform,
                            batch_size=self.config.batch_size,
                            val_split=0.0,
                            test_split=0.0,
                            shuffle=False
                        )
                    else:
                        # If no test or val directory, and no saved splits,
                        # create a new split and use test split
                        print(f"No dedicated test/val directory or saved splits. Creating a temporary test split.")
                        _, _, test_loader, classes = create_dataloaders(
                            dataset_path, transform,
                            batch_size=self.config.batch_size,
                            val_split=0.1,
                            test_split=0.1,
                            shuffle=False
                        )
            
            # Evaluate on the test set
            # Implementation for detection evaluation
            # Calculate mAP and other metrics
            
            # Clean up splits after evaluation if they won't be used again
            if job_id:
                from datasets_module.detection.dataloaders import ObjectDetectionDataset
                print("Cleaning up dataset split files...")
                ObjectDetectionDataset.cleanup_splits(job_id)
            
            return {
                "status": "completed",
                "mAP": 0.0,  # Placeholder
                "AP50": 0.0,  # Placeholder
                "AP75": 0.0,  # Placeholder
                "APs": 0.0,   # Small objects
                "APm": 0.0,   # Medium objects
                "APl": 0.0,   # Large objects
            }
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    @staticmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        return ["mAP", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]
        
    async def _update_job_log(self, job_id: str, message: str):
        """Update the job log with a message"""
        # This method should be implemented in the JobManager class
        # But we include it here for completeness
        print(f"Job {job_id}: {message}")

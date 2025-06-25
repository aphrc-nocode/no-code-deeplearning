"""
Pipeline for object detection tasks.
This module implements the BasePipeline interface for object detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import mlflow
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from pipelines.base_pipeline import BasePipeline
from models.detection import model_factory
from datasets_module.detection.dataloaders import create_dataloaders
from metrics.detection.metrics import calculate_detection_metrics
from mlflow_utils import log_model, start_run, log_metrics, log_batch_metrics, end_run

class ObjectDetectionPipeline(BasePipeline):
    """Pipeline for object detection tasks"""
    
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train the model with the given dataset"""
        try:
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.dict())
            
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
                
                # Use the directory as the return path
                local_model_path = str(model_output_path)
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
    
    def create_model(self) -> nn.Module:
        """Create a model based on the architecture specified in config"""
        return model_factory.create_model(
            self.config.architecture,
            num_classes=self.config.num_classes,
            pretrained=True
        ).to(self.device)
    
    def get_transforms(self):
        """Get data transforms for object detection"""
        from datasets_module.detection.transforms import get_detection_transforms
        return get_detection_transforms(train=True)
        
    async def predict(self, image, model=None) -> Dict[str, Any]:
        """Make a prediction using the trained model"""
        if model is None:
            raise ValueError("Model must be provided for prediction")
        
        # Apply transforms to the image
        transform = self.get_transforms()
        image_tensor = transform(image)[0].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            detections = model([image_tensor])
            
            # Format the results
            boxes = detections[0]['boxes'].cpu().numpy()
            scores = detections[0]['scores'].cpu().numpy()
            labels = detections[0]['labels'].cpu().numpy()
            
            return {
                "boxes": boxes.tolist(),
                "scores": scores.tolist(),
                "labels": labels.tolist()
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

"""
Pipeline for image segmentation tasks.
This module implements the BasePipeline interface for image segmentation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import mlflow
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from pipelines.base_pipeline import BasePipeline
from models.segmentation import model_factory
from datasets_module.segmentation.dataloaders import create_dataloaders
from metrics.segmentation.metrics import calculate_segmentation_metrics

class ImageSegmentationPipeline(BasePipeline):
    """Pipeline for image segmentation tasks (semantic and instance)"""
    
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train the model with the given dataset"""
        try:
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.dict())
            
            # Create model and setup training
            model = self.create_model()
            
            # Select appropriate loss function based on segmentation type
            if self.config.segmentation_type == "semantic":
                criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255 is often used as ignore index
            else:  # instance segmentation
                criterion = None  # Loss is computed inside the model
            
            # Create optimizer
            if self.config.segmentation_type == "semantic":
                # For semantic segmentation
                optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3
                )
            else:
                # For instance segmentation
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = optim.SGD(params, lr=self.config.learning_rate, 
                                    momentum=0.9, weight_decay=0.0005)
                scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=3, 
                                                    gamma=0.1)
            
            # Log optimizer configuration
            log_metrics({
                "initial_learning_rate": self.config.learning_rate
            })
            
            # Create datasets and data loaders
            transform = self.get_transforms()
            train_loader, val_loader, classes, dataset_num_classes = create_dataloaders(
                dataset_path, transform, 
                batch_size=self.config.batch_size,
                val_split=0.2 if self.config.early_stopping else 0.0,
                segmentation_type=self.config.segmentation_type
            )
            
            # Update num_classes from dataset if needed
            if dataset_num_classes != self.config.num_classes:
                self.config.num_classes = dataset_num_classes
                await self._update_job_log(job_id, f"Updated num_classes to {dataset_num_classes} based on dataset")
            
            # Store class mapping
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            
            # Log dataset information
            await self._update_job_log(job_id, f"Dataset loaded with {len(classes)} classes: {classes}")
            await self._update_job_log(job_id, f"Training on {len(train_loader.dataset)} samples")
            if val_loader:
                await self._update_job_log(job_id, f"Validating on {len(val_loader.dataset)} samples")
            
            # Initialize tracking variables
            best_loss = float('inf')
            best_miou = 0.0
            best_model_state = None
            patience_counter = 0
            
            # Track metrics for visualization
            train_losses = []
            train_mious = []
            val_losses = []
            val_mious = []
            
            # Training loop
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_outputs = []
                train_targets = []
                
                for batch_idx, batch_data in enumerate(train_loader):
                    if self.config.segmentation_type == "semantic":
                        images, targets = batch_data
                        images, targets = images.to(self.device), targets.to(self.device)
                        
                        # Forward pass
                        outputs = model(images)
                        
                        # Extract main output for segmentation
                        if isinstance(outputs, dict):
                            outputs = outputs['out']
                        
                        loss = criterion(outputs, targets)
                        
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        train_loss += loss.item()
                        
                        # Store predictions and targets for metric calculation
                        train_outputs.append(outputs.detach().cpu())
                        train_targets.append(targets.detach().cpu())
                    else:  # instance segmentation
                        images, targets = batch_data
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
                    
                    # Log batch progress every 10 batches
                    if batch_idx % 10 == 0:
                        await self._update_job_log(
                            job_id, 
                            f"Epoch {epoch+1}/{self.config.epochs} - "
                            f"Batch {batch_idx+1}/{len(train_loader)} - "
                            f"Loss: {loss.item() if self.config.segmentation_type == 'semantic' else losses.item():.4f}"
                        )
                
                train_loss /= len(train_loader)
                
                # Calculate IoU for semantic segmentation
                if self.config.segmentation_type == "semantic":
                    train_outputs = torch.cat(train_outputs, dim=0)
                    train_targets = torch.cat(train_targets, dim=0)
                    train_metrics = calculate_segmentation_metrics(
                        train_outputs, 
                        train_targets,
                        num_classes=self.config.num_classes
                    )
                    train_miou = train_metrics["miou"]
                else:
                    # For instance segmentation, we need a different evaluation approach
                    train_miou = 0.0  # Placeholder
                
                # Store metrics for visualization
                train_losses.append(train_loss)
                train_mious.append(train_miou)
                
                # Validation phase
                val_loss = 0.0
                val_miou = 0.0
                
                if val_loader:
                    model.eval()
                    val_outputs = []
                    val_targets = []
                    
                    with torch.no_grad():
                        for batch_data in val_loader:
                            if self.config.segmentation_type == "semantic":
                                images, targets = batch_data
                                images, targets = images.to(self.device), targets.to(self.device)
                                
                                # Forward pass
                                outputs = model(images)
                                
                                # Extract main output for segmentation
                                if isinstance(outputs, dict):
                                    outputs = outputs['out']
                                
                                loss = criterion(outputs, targets)
                                
                                val_loss += loss.item()
                                
                                # Store predictions and targets for metric calculation
                                val_outputs.append(outputs.detach().cpu())
                                val_targets.append(targets.detach().cpu())
                            else:  # instance segmentation
                                images, targets = batch_data
                                images = list(image.to(self.device) for image in images)
                                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                                
                                # Forward pass - compute losses
                                loss_dict = model(images, targets)
                                losses = sum(loss for loss in loss_dict.values())
                                
                                val_loss += losses.item()
                    
                    val_loss /= len(val_loader)
                    
                    # Calculate IoU for semantic segmentation
                    if self.config.segmentation_type == "semantic" and val_outputs:
                        val_outputs = torch.cat(val_outputs, dim=0)
                        val_targets = torch.cat(val_targets, dim=0)
                        val_metrics = calculate_segmentation_metrics(
                            val_outputs, 
                            val_targets,
                            num_classes=self.config.num_classes
                        )
                        val_miou = val_metrics["miou"]
                        
                        # Log detailed metrics
                        log_metrics({
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_miou": train_miou,
                            "val_loss": val_loss,
                            "val_miou": val_miou,
                            **{f"val_iou_class_{i}": val_metrics[f"iou_class_{i}"] for i in range(self.config.num_classes)}
                        })
                    else:
                        # For instance segmentation, we need a different evaluation approach
                        val_miou = 0.0  # Placeholder
                        log_metrics({
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss
                        })
                    
                    # Store metrics for visualization
                    val_losses.append(val_loss)
                    val_mious.append(val_miou)
                    
                    # Update learning rate based on validation loss
                    if self.config.segmentation_type == "semantic":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                    
                    # Early stopping based on validation metrics
                    if self.config.early_stopping:
                        if self.config.segmentation_type == "semantic" and val_miou > best_miou:
                            best_loss = val_loss
                            best_miou = val_miou
                            best_model_state = model.state_dict()
                            patience_counter = 0
                        elif val_loss < best_loss:  # For instance segmentation
                            best_loss = val_loss
                            best_model_state = model.state_dict()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.config.patience:
                                await self._update_job_log(job_id, f"Early stopping at epoch {epoch+1}")
                                break
                else:
                    # If no validation set, use training metrics
                    if self.config.segmentation_type == "semantic" and train_miou > best_miou:
                        best_loss = train_loss
                        best_miou = train_miou
                        best_model_state = model.state_dict()
                        patience_counter = 0
                    elif train_loss < best_loss:  # For instance segmentation
                        best_loss = train_loss
                        best_model_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience and self.config.early_stopping:
                            await self._update_job_log(job_id, f"Early stopping at epoch {epoch+1}")
                            break
            
            # Create visualization plots
            from mlflow_utils import log_training_visualizations
            
            # Log training curves - customize for segmentation metrics
            # TODO: Implement visualization for segmentation metrics
            
            # Log example segmentation masks
            # TODO: Log example segmentation predictions vs ground truth
            
            # Save best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Save the model
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{job_id}.pth"
            
            # Save necessary components
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {k: v for k, v in self.config.dict().items()},
                'class_to_idx': class_to_idx
            }, model_path)
            
            # Log final model to MLflow
            log_model(
                model=model,
                model_path=model_path,
                class_to_idx=class_to_idx,
                config=self.config.dict()
            )
            
            # End MLflow run
            end_run()
            
            await self._update_job_log(job_id, f"Training completed successfully. Model saved to {model_path}")
            
            # Return a dictionary with status and model path
            return {
                "status": "completed",
                "model_path": model_path,
                "metrics": {
                    "final_train_loss": train_loss,
                    "final_train_miou": train_miou if self.config.segmentation_type == "semantic" else None,
                    "final_val_loss": val_loss if val_loader else None,
                    "final_val_miou": val_miou if val_loader and self.config.segmentation_type == "semantic" else None
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
            segmentation_type=self.config.segmentation_type,
            pretrained=True
        ).to(self.device)
    
    def get_transforms(self):
        """Get data transforms for image segmentation"""
        from datasets_module.segmentation.transforms import get_segmentation_transforms
        return get_segmentation_transforms(
            train=True,
            segmentation_type=self.config.segmentation_type,
            image_size=self.config.image_size
        )
        
    async def predict(self, image, model=None) -> Dict[str, Any]:
        """Make a prediction using the trained model"""
        if model is None:
            raise ValueError("Model must be provided for prediction")
        
        # Apply transforms to the image
        transform = self.get_transforms()
        image_tensor, _ = transform(image, None)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            
            if self.config.segmentation_type == "semantic":
                outputs = model(image_tensor)
                
                # Extract main output for segmentation
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                # Convert to class probabilities and get predicted class
                probs = torch.softmax(outputs, dim=1)
                pred_mask = torch.argmax(probs, dim=1).cpu().numpy()[0]
                
                return {
                    "segmentation_mask": pred_mask.tolist(),
                    "probabilities": probs[0].cpu().tolist()
                }
            else:  # instance segmentation
                predictions = model(image_tensor)
                
                # Process instance segmentation results
                boxes = predictions[0]['boxes'].cpu().numpy()
                masks = predictions[0]['masks'].cpu().numpy()
                labels = predictions[0]['labels'].cpu().numpy()
                scores = predictions[0]['scores'].cpu().numpy()
                
                # Filter by confidence threshold
                conf_threshold = 0.5
                keep = scores > conf_threshold
                
                return {
                    "boxes": boxes[keep].tolist(),
                    "masks": masks[keep].tolist(),
                    "labels": labels[keep].tolist(),
                    "scores": scores[keep].tolist()
                }
    
    async def evaluate(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate the model on a test dataset"""
        # Implementation for evaluation on test dataset
        pass
    
    @staticmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        return [
            "miou",  # Mean IoU across all classes
            "pixel_accuracy",  # Percentage of correctly classified pixels
            "accuracy_per_class",  # Accuracy for each class
            "precision_per_class",  # Precision for each class
            "recall_per_class",  # Recall for each class
            "f1_per_class"  # F1 score for each class
        ]
        
    async def _update_job_log(self, job_id: str, message: str):
        """Update the job log with a message"""
        # This method should be implemented in the JobManager class
        # But we include it here for completeness
        print(f"Job {job_id}: {message}")

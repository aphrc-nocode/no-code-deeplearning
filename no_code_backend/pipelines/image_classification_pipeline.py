"""
Pipeline for image classification tasks.
This module implements the BasePipeline interface for image classification.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import mlflow
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from pipelines.base_pipeline import BasePipeline
from models.classification import model_factory
from datasets_module.classification.dataloaders import create_dataloaders
from metrics.classification.metrics import calculate_classification_metrics

class ImageClassificationPipeline(BasePipeline):
    """Pipeline for image classification tasks"""
    
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train the model with the given dataset"""
        try:
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.dict())
            
            # Create model and setup training
            model = self.create_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            
            # Log optimizer configuration
            log_metrics({
                "initial_learning_rate": self.config.learning_rate
            })
            
            # Create datasets and data loaders
            transform = self.get_transforms()
            train_loader, val_loader, classes = create_dataloaders(
                dataset_path, transform, 
                batch_size=self.config.batch_size,
                val_split=0.2 if self.config.early_stopping else 0.0
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
            best_acc = 0.0
            best_model_state = None
            patience_counter = 0
            
            # Track metrics for visualization
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            
            # Count class distribution for visualization
            class_counts = {}
            for _, targets in train_loader:
                for target in targets:
                    label = classes[target.item()]
                    class_counts[label] = class_counts.get(label, 0) + 1
            
            # Log class distribution as image
            from visualization_utils import create_class_distribution_plot
            class_dist_path = create_class_distribution_plot(class_counts)
            mlflow.log_artifact(class_dist_path, "visualizations")
            
            # Training loop
            for epoch in range(self.config.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # Log batch progress every 10 batches
                    if batch_idx % 10 == 0:
                        await self._update_job_log(
                            job_id, 
                            f"Epoch {epoch+1}/{self.config.epochs} - "
                            f"Batch {batch_idx+1}/{len(train_loader)} - "
                            f"Loss: {loss.item():.4f}"
                        )
                
                train_loss /= len(train_loader)
                train_acc = 100.0 * correct / total
                
                # Store metrics for visualization
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # Validation phase
                val_loss = 0.0
                val_acc = 0.0
                
                # Collect predictions and targets for confusion matrix at final epoch
                all_preds = []
                all_targets = []
                all_scores = []
                
                if val_loader:
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for data, targets in val_loader:
                            data, targets = data.to(self.device), targets.to(self.device)
                            outputs = model(data)
                            loss = criterion(outputs, targets)
                            
                            val_loss += loss.item()
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                            
                            # Store predictions and targets for final epoch
                            if epoch == self.config.epochs - 1:
                                all_preds.append(predicted.cpu())
                                all_targets.append(targets.cpu())
                                all_scores.append(torch.softmax(outputs, dim=1).cpu())
                    
                    val_loss /= len(val_loader)
                    val_acc = 100.0 * val_correct / val_total
                    
                    # Store metrics for visualization
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    
                    # Update learning rate based on validation loss
                    scheduler.step(val_loss)
                    
                    # Early stopping based on validation metrics
                    if self.config.early_stopping:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_acc = val_acc
                            best_model_state = model.state_dict()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.config.patience:
                                await self._update_job_log(job_id, f"Early stopping at epoch {epoch+1}")
                                break
                else:
                    # If no validation set, use training metrics
                    if train_loss < best_loss:
                        best_loss = train_loss
                        best_acc = train_acc
                        best_model_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config.patience and self.config.early_stopping:
                            await self._update_job_log(job_id, f"Early stopping at epoch {epoch+1}")
                            break
            
            # Create visualization plots
            from mlflow_utils import log_training_visualizations, log_evaluation_visualizations
            
            # Log training curves
            log_training_visualizations(train_losses, train_accs, val_losses, val_accs)
            
            # Create and log confusion matrix if validation data is available
            if val_loader and len(all_preds) > 0:
                # Concatenate all predictions and targets
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
                all_scores = torch.cat(all_scores)
                
                # Calculate additional classification metrics
                metrics_dict = calculate_classification_metrics(all_targets.numpy(), all_preds.numpy(), all_scores.numpy())
                log_metrics(metrics_dict)
                
                # Log evaluation visualizations
                log_evaluation_visualizations(
                    all_targets, 
                    all_preds, 
                    all_scores,
                    class_names=classes,
                    class_counts=class_counts
                )
            else:
                # Initialize metrics_dict as empty if no validation data
                metrics_dict = {}
            
            # Save best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Save the model
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{job_id}.pth"
            
            # Save only the necessary components
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
                    "final_train_accuracy": train_acc,
                    "final_val_loss": val_loss if val_loader else None,
                    "final_val_accuracy": val_acc if val_loader else None,
                    **({} if not val_loader else metrics_dict)  # Include additional metrics if we have validation data
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
            num_classes=self.config.num_classes
        ).to(self.device)
    
    def get_transforms(self):
        """Get data transforms based on configuration"""
        from torchvision import transforms
        
        base_transform = [
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        if self.config.augmentation_enabled:
            augment_transform = [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ] + base_transform
            return transforms.Compose(augment_transform)
        
        return transforms.Compose(base_transform)
        
    async def predict(self, image, model=None) -> Dict[str, Any]:
        """Make a prediction using the trained model"""
        if model is None:
            raise ValueError("Model must be provided for prediction")
        
        # Apply transforms to the image
        transform = self.get_transforms()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Get top-k predictions
            num_classes = len(probabilities)
            k = min(5, num_classes)
            top_p, top_class = torch.topk(probabilities, k)
            
            return {
                "probabilities": probabilities.cpu().tolist(),
                "predicted_classes": top_class.cpu().tolist(),
                "confidence_scores": top_p.cpu().tolist()
            }
    
    async def evaluate(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate the model on a test dataset"""
        # Implementation for evaluation on test dataset
        pass
    
    @staticmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        return ["accuracy", "precision", "recall", "f1_score"]
        
    async def _update_job_log(self, job_id: str, message: str):
        """Update the job log with a message"""
        # This method should be implemented in the JobManager class
        # But we include it here for completeness
        print(f"Job {job_id}: {message}")

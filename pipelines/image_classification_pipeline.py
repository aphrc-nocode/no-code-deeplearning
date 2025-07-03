"""
Pipeline for image classification tasks.
This module implements the BasePipeline interface for image classification.
"""
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import time
import os
import mlflow


from pipelines.base_pipeline import BasePipeline
from models.classification import model_factory
from datasets_module.classification.dataloaders import create_dataloaders
from metrics.classification.metrics import calculate_classification_metrics, get_confusion_matrix, generate_classification_report

class ImageClassificationPipeline(BasePipeline):
    """Pipeline for image classification tasks"""
    
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train the model with the given dataset"""
        try:
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.dict())
            
            # Create model with transfer learning approach
            model = self.create_model()
            
            # Setup training components
            criterion = nn.CrossEntropyLoss()
            
            # Handle different optimization strategies based on whether we want to fine-tune or use as feature extractor
            if self.config.feature_extraction_only:
                # Feature extraction: freeze all layers except the final layer
                for param in model.parameters():
                    param.requires_grad = False
                
                # Only optimize the final fully connected layer parameters
                if self.config.architecture.startswith('resnet'):
                    optimizer = optim.SGD(model.fc.parameters(), lr=self.config.learning_rate, momentum=0.9)
                elif self.config.architecture.startswith('vgg'):
                    optimizer = optim.SGD(model.classifier[6].parameters(), lr=self.config.learning_rate, momentum=0.9)
                else:
                    # Default for other models
                    optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
            else:
                # Fine-tuning: optimize all parameters
                optimizer = optim.SGD(model.parameters(), lr=self.config.learning_rate, momentum=0.9)
            
            # Learning rate scheduler
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Log optimizer configuration
            log_metrics({
                "initial_learning_rate": self.config.learning_rate
            })
            
            # Create datasets and data loaders with persistent splits
            transform = self.get_transforms()
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
            if test_loader:
                await self._update_job_log(job_id, f"Test set has {len(test_loader.dataset)} samples (reserved for evaluation)")
            
            # Initialize tracking variables
            best_loss = float('inf')
            best_acc = 0.0
            best_model_state = None
            patience_counter = 0
            
            # Initialize variables to prevent "referenced before assignment" errors
            epoch = 0
            all_preds = []
            all_targets = []
            all_scores = []
            
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
            
            # We'll use MLflow for model storage primarily
            # Local directory only used as fallback if MLflow fails
            model_dir = Path(f"models/{job_id}")
            model_dir.mkdir(exist_ok=True, parents=True)
            best_model_path = None
            
            # Training loop
            since = time.time()
            for epoch in range(self.config.epochs):
                await self._update_job_log(job_id, f"Epoch {epoch+1}/{self.config.epochs}")
                
                # Training phase
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with gradient tracking
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
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
                
                train_loss = train_loss / len(train_loader)
                train_acc = 100.0 * correct / total
                
                # Store metrics for visualization
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                
                # Log epoch metrics
                epoch_metrics = {
                    f"train_loss_epoch_{epoch}": train_loss,
                    f"train_acc_epoch_{epoch}": train_acc / 100.0,  # Convert to 0-1 range
                }
                
                # Validation phase
                val_loss = 0.0
                val_acc = 0.0
                
                if val_loader:
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for data, targets in val_loader:
                            data, targets = data.to(self.device), targets.to(self.device)
                            
                            # Forward pass without gradients
                            outputs = model(data)
                            loss = criterion(outputs, targets)
                            
                            # Update metrics
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                            
                            # Store for final confusion matrix
                            if epoch == self.config.epochs - 1:
                                all_preds.append(predicted.cpu())
                                all_targets.append(targets.cpu())
                                all_scores.append(torch.softmax(outputs, dim=1).cpu())
                    
                    val_loss = val_loss / len(val_loader)
                    val_acc = 100.0 * val_correct / val_total
                    
                    # Store metrics for visualization
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    
                    # Log validation metrics
                    epoch_metrics.update({
                        f"val_loss_epoch_{epoch}": val_loss,
                        f"val_acc_epoch_{epoch}": val_acc / 100.0,  # Convert to 0-1 range
                    })
                    
                    # Check for improvement
                    improved = False
                    if self.config.early_stopping:
                        if val_loss < best_loss:
                            best_loss = val_loss
                            improved = True
                        if val_acc > best_acc:
                            best_acc = val_acc
                            improved = True
                            
                        if improved:
                            patience_counter = 0
                            best_model_state = model.state_dict()
                            
                            # Log best model checkpoint to MLflow
                            mlflow.pytorch.log_state_dict(model.state_dict(), "best_model_checkpoint")
                            
                            # Also log metadata about this checkpoint
                            checkpoint_metadata = {
                                'epoch': epoch,
                                'loss': val_loss,
                                'accuracy': val_acc,
                            }
                            mlflow.log_dict(checkpoint_metadata, "best_model_metadata.json")
                            
                            await self._update_job_log(job_id, f"Model improved, saved checkpoint at epoch {epoch+1}")
                        else:
                            patience_counter += 1
                            
                        # Early stopping check
                        if patience_counter >= self.config.patience:
                            await self._update_job_log(job_id, f"Early stopping triggered at epoch {epoch+1}")
                            break
                else:
                    # If no validation, log the latest model state to MLflow
                    mlflow.pytorch.log_state_dict(model.state_dict(), f"checkpoint_epoch_{epoch}")
                    
                    # Also log metadata about this checkpoint
                    checkpoint_metadata = {
                        'epoch': epoch,
                        'loss': train_loss,
                        'accuracy': train_acc,
                    }
                    mlflow.log_dict(checkpoint_metadata, f"checkpoint_epoch_{epoch}_metadata.json")
                
                # Step the scheduler
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loader else train_loss)
                else:
                    scheduler.step()
                
                # Log metrics for this epoch
                log_metrics(epoch_metrics)
                
                # Log epoch summary
                await self._update_job_log(
                    job_id,
                    f"Epoch {epoch+1}/{self.config.epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    + (f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%" if val_loader else "")
                )
            
            # Training complete
            time_elapsed = time.time() - since
            await self._update_job_log(
                job_id,
                f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
            
            # Load best model if early stopping was used
            if self.config.early_stopping and best_model_state is not None:
                model.load_state_dict(best_model_state)
                await self._update_job_log(job_id, "Loaded best model from checkpoints")
            elif self.config.early_stopping:
                # Check if we actually created a checkpoint during training
                if val_loader and val_loader.dataset and len(val_loader.dataset) > 0:
                    # Try to get best model from MLflow
                    try:
                        best_model_dict = mlflow.pytorch.load_state_dict("best_model_checkpoint")
                        model.load_state_dict(best_model_dict)
                        await self._update_job_log(job_id, "Loaded best model from MLflow")
                    except Exception as e:
                        await self._update_job_log(job_id, f"Warning: Could not load best model from MLflow: {str(e)}")
                else:
                    await self._update_job_log(job_id, "No validation data was available for early stopping")
            
            # Final evaluation and visualizations
            if len(train_losses) > 0:
                from visualization_utils import create_loss_curve, create_accuracy_curve
                
                # Loss curve
                loss_curve_path = create_loss_curve(
                    train_losses, 
                    val_losses if val_loader else None
                )
                mlflow.log_artifact(loss_curve_path, "visualizations")
                
                # Accuracy curve
                acc_curve_path = create_accuracy_curve(
                    train_accs, 
                    val_accs if val_loader else None
                )
                mlflow.log_artifact(acc_curve_path, "visualizations")
            
            # Create final confusion matrix if validation data was used and we have predictions
            if val_loader and len(all_preds) > 0 and len(all_targets) > 0:
                from visualization_utils import create_confusion_matrix
                
                try:
                    # Concatenate lists
                    all_preds = torch.cat(all_preds).numpy()
                    all_targets = torch.cat(all_targets).numpy()
                    all_scores = torch.cat(all_scores).numpy()
                except RuntimeError as e:
                    await self._update_job_log(job_id, f"Warning: Could not concatenate predictions: {e}")
                    # Continue execution instead of returning which would abort the training
                    all_preds = None
                
                # Only create confusion matrix and calculate metrics if concatenation worked
                if all_preds is not None:
                    # Create and log confusion matrix
                    try:
                        cm_path = create_confusion_matrix(
                            all_targets, 
                            all_preds,
                            class_names=classes,
                            title="Validation Confusion Matrix"
                        )
                        mlflow.log_artifact(cm_path, "visualizations")
                        
                        # Calculate and log final metrics
                        final_metrics = calculate_classification_metrics(all_targets, all_preds, all_scores)
                        log_metrics({"final_" + k: v for k, v in final_metrics.items()})
                    except Exception as e:
                        await self._update_job_log(job_id, f"Warning: Error creating validation metrics: {e}")
            
            # Create the target directory for MLflow models
            mlflow_models_dir = Path("logs/mlflow/models")
            mlflow_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save class mapping separately for easier loading
            class_mapping_path = mlflow_models_dir / f"{job_id}_class_mapping.json"
            import json
            with open(class_mapping_path, 'w') as f:
                json.dump(class_to_idx, f)
            
            # Log model to MLflow
            log_model(model, "model", {
                'architecture': self.config.architecture,
                'num_classes': self.config.num_classes,
                'class_to_idx': class_to_idx,
                'input_size': self.config.image_size
            })
            
            # Get MLflow run info to use as model path
            run_info = mlflow.active_run()
            if run_info:
                # Use our custom MLflow models directory
                run_id = run_info.info.run_id
                model_output_path = mlflow_models_dir / run_id
                model_output_path.mkdir(exist_ok=True)
                
                # Save a copy of the model to our custom directory
                final_model_path = model_output_path / "model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'config': self.config.dict()
                }, final_model_path)
                
                # Log the local path as an artifact
                mlflow.log_artifact(str(final_model_path))
                
                # Keep final_model_path pointing to the actual .pth file
                # final_model_path already points to the correct file
            else:
                # Fallback to local storage only if MLflow run isn't active
                final_model_path = model_dir / "final_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'config': self.config.dict()
                }, final_model_path)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': class_to_idx,
                    'config': self.config.dict()
                }, final_model_path)
            
            # End MLflow run
            end_run()
            
            # Get MLflow run information for model reference
            run_info = mlflow.active_run()
            mlflow_run_id = run_info.info.run_id if run_info else None
            mlflow_model_uri = f"runs:/{mlflow_run_id}/model" if mlflow_run_id else None
            
            return {
                "status": "completed",
                "model_path": str(final_model_path),
                "mlflow_model_uri": mlflow_model_uri,
                "mlflow_run_id": mlflow_run_id,
                "training_time": time_elapsed,
                "epochs_completed": epoch + 1,
                "final_train_loss": train_loss,
                "final_train_accuracy": train_acc / 100.0,
                "final_val_loss": val_loss if val_loader else None,
                "final_val_accuracy": val_acc / 100.0 if val_loader else None,
                "class_mapping": class_to_idx
            }
                
        except Exception as e:
            await self._update_job_log(job_id, f"Error during training: {str(e)}")
            
            # End MLflow run on error
            try:
                from mlflow_utils import end_run
                end_run()
            except:
                pass
                
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
        
        # Set image size from config
        resize_dim = self.config.image_size
        
        # ImageNet normalization values
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        if self.config.augmentation_enabled:
            # Training transforms with data augmentation
            transform = transforms.Compose([
                transforms.RandomResizedCrop(resize_dim),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # Validation and testing transforms (no augmentation)
            transform = transforms.Compose([
                transforms.Resize(int(resize_dim * 1.14)),  # Slightly larger for center crop
                transforms.CenterCrop(resize_dim),
                transforms.ToTensor(),
                normalize,
            ])
        
        return transform
        
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
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top-k predictions
            topk = min(5, outputs.size(1))  # Get top-5 or fewer if less than 5 classes
            confidence, predicted = torch.topk(probabilities, topk)
            
            # Convert to lists
            confidence = confidence.squeeze().tolist()
            predicted = predicted.squeeze().tolist()
            
            # Ensure we handle scalars properly
            if isinstance(confidence, float):
                confidence = [confidence]
                predicted = [predicted]
                
            # Return predictions with confidence
            predictions = []
            for i, (pred, conf) in enumerate(zip(predicted, confidence)):
                predictions.append({
                    "rank": i + 1,
                    "class_id": int(pred),
                    "confidence": float(conf)
                })
                
            return {
                "predictions": predictions,
                "raw_output": outputs.cpu().numpy().tolist()
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
                idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else None
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
                # Use the test directory exclusively
                print(f"Using dedicated test directory: {test_dir}")
                test_loader, _, _, classes = create_dataloaders(
                    str(test_dir), transform, 
                    batch_size=self.config.batch_size,
                    val_split=0.0,  # Use all data for testing
                    test_split=0.0,  # No additional split needed
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
                    # Fallback to previous behavior - check for val directory
                    val_dir = Path(dataset_path) / "val"
                    if val_dir.exists():
                        print(f"Using validation directory: {val_dir}")
                        train_loader, test_loader, _, classes = create_dataloaders(
                            str(val_dir), transform,
                            batch_size=self.config.batch_size,
                            val_split=0.0,
                            test_split=0.0,
                            shuffle=False
                        )
                    else:
                        # If no test or val directory, and no saved splits,
                        # create a new 80/10/10 split and use the test split
                        print(f"No dedicated test/val directory or saved splits. Creating a temporary test split.")
                        train_loader, val_loader, test_loader, classes = create_dataloaders(
                            dataset_path, transform,
                            batch_size=self.config.batch_size,
                            val_split=0.1,
                            test_split=0.1,
                            shuffle=False
                        )
            
            # If we have a mapping from the training phase, use it for consistency
            if idx_to_class is not None:
                classes = [idx_to_class[i] for i in range(len(idx_to_class))]
            
            # Track metrics
            all_targets = []
            all_preds = []
            all_scores = []
            
            # Evaluate the model
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = model(data)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Store for metrics
                    all_preds.append(predicted.cpu())
                    all_targets.append(targets.cpu())
                    all_scores.append(probabilities.cpu())
                    
                    if job_id:
                        # Log progress occasionally
                        if len(all_preds) % 10 == 0:
                            await self._update_job_log(job_id, f"Evaluated {len(all_preds)} batches")
            
            # Concatenate all tensors
            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()
            all_scores = torch.cat(all_scores).numpy()
            
            # Calculate metrics
            metrics_dict = calculate_classification_metrics(all_targets, all_preds, all_scores)
            
            # Log the evaluation metrics
            if job_id:
                await self._update_job_log(job_id, f"Evaluation complete. Accuracy: {metrics_dict.get('accuracy', 0):.4f}")
                
                # If MLflow is being used
                try:
                    from mlflow_utils import start_run, log_metrics, end_run
                    run_id = start_run(f"{job_id}_eval", self.config.dict())
                    
                    log_metrics(metrics_dict)
                    
                    # Create confusion matrix visualization
                    from visualization_utils import create_confusion_matrix
                    cm_path = create_confusion_matrix(
                        all_targets, 
                        all_preds,
                        class_names=classes,
                        title="Confusion Matrix"
                    )
                    mlflow.log_artifact(cm_path, "visualizations")
                    
                    end_run()
                except ImportError:
                    # MLflow utils not available
                    pass
            
            # Generate and return the full evaluation results
            confusion = get_confusion_matrix(all_targets, all_preds)
            report = generate_classification_report(all_targets, all_preds, target_names=classes)
            
            return {
                "status": "completed",
                "metrics": metrics_dict,
                "confusion_matrix": confusion.tolist(),
                "classification_report": report,
                "class_names": classes
            }
            
        except Exception as e:
            if job_id:
                await self._update_job_log(job_id, f"Error during evaluation: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    @staticmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        return [
            'accuracy', 
            'precision_macro', 
            'recall_macro', 
            'f1_macro',
            'precision_weighted', 
            'recall_weighted', 
            'f1_weighted'
        ]
        
    async def _update_job_log(self, job_id: str, message: str):
        """Update the job log with a message"""
        # This method should be implemented in the JobManager class
        # But we include it here for completeness
        print(f"Job {job_id}: {message}")

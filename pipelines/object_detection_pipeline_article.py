"""
Object Detection Pipeline following the article's approach exactly.
Pipeline for Training Custom Faster-RCNN Object Detection models with PyTorch.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time

# Import the article-based components
from datasets_module.detection.coco_dataset_article import CocoDetectionDataset, get_transforms, collate_fn
from models.detection.model_factory import create_model
from visualization_utils_article import visualize_batch_from_dataloader, visualize_predictions
from pipelines.base_pipeline import BasePipeline

# Try to import PyTorch Vision reference training utilities
try:
    from torchvision.references.detection import train_one_epoch, evaluate
    from torchvision.references.detection.utils import MetricLogger, SmoothedValue
    PYTORCH_VISION_AVAILABLE = True
except ImportError:
    PYTORCH_VISION_AVAILABLE = False
    print("PyTorch Vision reference training utilities not available. Using custom training loop.")

# Import for COCO evaluation
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Warning: pycocotools not available. COCO evaluation will be limited.")


class ObjectDetectionPipeline(BasePipeline):
    """
    Object Detection Pipeline following the article's approach.
    Supports training custom Faster R-CNN models on COCO-format datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Object Detection Pipeline
        
        Args:
            config: Configuration dictionary containing all pipeline parameters
        """
        super().__init__(config)
        
        # Core configuration
        self.num_classes = config.get('num_classes', 2)  # Including background
        self.architecture = config.get('architecture', 'faster_rcnn')
        self.learning_rate = config.get('learning_rate', 0.005)
        self.batch_size = config.get('batch_size', 2)
        self.num_epochs = config.get('num_epochs', 10)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset paths
        self.train_image_dir = config.get('train_image_dir', '')
        self.train_annotation_path = config.get('train_annotation_path', '')
        self.val_image_dir = config.get('val_image_dir', '')
        self.val_annotation_path = config.get('val_annotation_path', '')
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Class names for visualization
        self.class_names = config.get('class_names', [f'class_{i}' for i in range(self.num_classes)])
        
        # Training configuration
        self.weight_decay = config.get('weight_decay', 0.0005)
        self.momentum = config.get('momentum', 0.9)
        self.step_size = config.get('step_size', 3)
        self.gamma = config.get('gamma', 0.1)
        
        # Paths for saving
        self.output_dir = config.get('output_dir', './outputs')
        self.model_save_path = config.get('model_save_path', './outputs/model.pth')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Initialized Object Detection Pipeline with {self.num_classes} classes")
        print(f"Using device: {self.device}")
        print(f"Architecture: {self.architecture}")
    
    def load_data(self) -> None:
        """
        Load and prepare datasets following the article's approach
        """
        print("Loading datasets...")
        
        # Get transforms for training and validation
        train_transforms = get_transforms(train=True)
        val_transforms = get_transforms(train=False)
        
        # Create training dataset
        if self.train_image_dir and self.train_annotation_path:
            train_dataset = CocoDetectionDataset(
                image_dir=self.train_image_dir,
                annotation_path=self.train_annotation_path,
                transforms=train_transforms
            )
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4
            )
            print(f"Training dataset loaded: {len(train_dataset)} images")
        
        # Create validation dataset
        if self.val_image_dir and self.val_annotation_path:
            val_dataset = CocoDetectionDataset(
                image_dir=self.val_image_dir,
                annotation_path=self.val_annotation_path,
                transforms=val_transforms
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4
            )
            print(f"Validation dataset loaded: {len(val_dataset)} images")
        
        print("Datasets loaded successfully!")
    
    def build_model(self) -> None:
        """
        Build the detection model following the article's approach
        """
        print("Building model...")
        
        # Create model using the factory
        self.model = create_model(
            architecture=self.architecture,
            num_classes=self.num_classes,
            pretrained=True
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up optimizer exactly as in the article
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )
        
        print("Model built successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch using custom training loop
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses.item(),
                'avg_loss': total_loss / num_batches
            })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate_epoch(self) -> Dict[str, float]:
        """
        Evaluate the model on validation set
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Evaluating"):
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'val_loss': avg_loss}
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop following the article's approach
        
        Returns:
            Dictionary with training results and metrics
        """
        print("Starting training...")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'num_classes': self.num_classes,
                'architecture': self.architecture,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'weight_decay': self.weight_decay,
                'momentum': self.momentum
            })
            
            # Training loop
            training_metrics = []
            best_val_loss = float('inf')
            
            for epoch in range(self.num_epochs):
                start_time = time.time()
                
                # Train for one epoch
                if PYTORCH_VISION_AVAILABLE and hasattr(self, 'train_loader'):
                    # Use PyTorch Vision's train_one_epoch if available
                    try:
                        train_metrics = train_one_epoch(
                            self.model, self.optimizer, self.train_loader, 
                            self.device, epoch, print_freq=10
                        )
                        train_loss = train_metrics.meters['loss'].global_avg
                        epoch_metrics = {'train_loss': train_loss}
                    except Exception as e:
                        print(f"Using custom training loop due to error: {e}")
                        epoch_metrics = self.train_epoch(epoch)
                else:
                    # Use custom training loop
                    epoch_metrics = self.train_epoch(epoch)
                
                # Evaluate
                val_metrics = self.evaluate_epoch()
                epoch_metrics.update(val_metrics)
                
                # Update learning rate
                self.lr_scheduler.step()
                
                # Log metrics
                mlflow.log_metrics(epoch_metrics, step=epoch)
                training_metrics.append(epoch_metrics)
                
                # Save best model
                if 'val_loss' in epoch_metrics and epoch_metrics['val_loss'] < best_val_loss:
                    best_val_loss = epoch_metrics['val_loss']
                    self.save_model(f"{self.output_dir}/best_model.pth")
                
                # Print epoch summary
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch + 1}/{self.num_epochs} completed in {epoch_time:.2f}s")
                for key, value in epoch_metrics.items():
                    print(f"  {key}: {value:.4f}")
            
            # Save final model
            self.save_model(self.model_save_path)
            
            # Log model to MLflow
            mlflow.pytorch.log_model(self.model, "model")
            
            print("Training completed!")
            
            return {
                'status': 'completed',
                'metrics': training_metrics,
                'best_val_loss': best_val_loss,
                'model_path': self.model_save_path
            }
    
    def predict(self, image_path: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Make predictions on a single image following the article's approach
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detections
            
        Returns:
            List of detections with boxes, scores, and labels
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call build_model() first.")
        
        # Load and preprocess image
        from PIL import Image
        import torchvision.transforms as T
        
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        pred = predictions[0]
        detections = []
        
        # Filter by threshold
        keep = pred['scores'] > threshold
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                'box': box.tolist(),  # [x1, y1, x2, y2]
                'score': float(score),
                'label': int(label),
                'class_name': self.class_names[label] if label < len(self.class_names) else f'class_{label}'
            })
        
        return detections
    
    def visualize_batch(self, num_samples: int = 2) -> None:
        """
        Visualize a batch of training data
        
        Args:
            num_samples: Number of samples to visualize
        """
        if not self.train_loader:
            print("Training data not loaded. Call load_data() first.")
            return
        
        visualize_batch_from_dataloader(self.train_loader, num_samples)
    
    def visualize_predictions(self, image_path: str, threshold: float = 0.8) -> None:
        """
        Visualize model predictions on an image
        
        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detections
        """
        if self.model is None:
            print("Model not loaded. Call build_model() first.")
            return
        
        visualize_predictions(
            image_path=image_path,
            model=self.model,
            device=self.device,
            label_list=self.class_names,
            threshold=threshold
        )
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'architecture': self.architecture,
                'class_names': self.class_names
            }
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer if available
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.num_classes = config.get('num_classes', self.num_classes)
            self.architecture = config.get('architecture', self.architecture)
            self.class_names = config.get('class_names', self.class_names)
        
        print(f"Model loaded from {path}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on validation set with detailed metrics
        
        Returns:
            Dictionary with evaluation results
        """
        if not self.val_loader:
            print("Validation data not loaded. Call load_data() first.")
            return {}
        
        if self.model is None:
            print("Model not loaded. Call build_model() first.")
            return {}
        
        print("Running evaluation...")
        
        # Use PyTorch Vision's evaluate function if available
        if PYTORCH_VISION_AVAILABLE:
            try:
                eval_result = evaluate(self.model, self.val_loader, device=self.device)
                return {
                    'status': 'completed',
                    'coco_eval': eval_result
                }
            except Exception as e:
                print(f"Using custom evaluation due to error: {e}")
        
        # Custom evaluation
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Evaluating"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                predictions = self.model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate basic metrics
        total_detections = sum(len(pred['boxes']) for pred in all_predictions)
        total_ground_truth = sum(len(target['boxes']) for target in all_targets)
        
        return {
            'status': 'completed',
            'total_predictions': total_detections,
            'total_ground_truth': total_ground_truth,
            'num_images': len(all_predictions)
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {'status': 'Model not loaded'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'class_names': self.class_names
        }


def create_detection_pipeline(config: Dict[str, Any]) -> ObjectDetectionPipeline:
    """
    Factory function to create an ObjectDetectionPipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured ObjectDetectionPipeline instance
    """
    return ObjectDetectionPipeline(config)


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration following the article's approach
    config = {
        'num_classes': 2,  # Including background
        'architecture': 'faster_rcnn',
        'learning_rate': 0.005,
        'batch_size': 2,
        'num_epochs': 10,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'step_size': 3,
        'gamma': 0.1,
        'train_image_dir': './datasets/coco/train2017',
        'train_annotation_path': './datasets/coco/annotations/instances_train2017.json',
        'val_image_dir': './datasets/coco/val2017',
        'val_annotation_path': './datasets/coco/annotations/instances_val2017.json',
        'output_dir': './outputs',
        'model_save_path': './outputs/faster_rcnn_model.pth',
        'class_names': ['background', 'person']  # Example classes
    }
    
    # Create and run pipeline
    pipeline = create_detection_pipeline(config)
    
    # Load data
    pipeline.load_data()
    
    # Build model
    pipeline.build_model()
    
    # Visualize some training samples
    pipeline.visualize_batch(num_samples=2)
    
    # Train the model
    results = pipeline.train()
    print("Training results:", results)
    
    # Evaluate the model
    eval_results = pipeline.evaluate()
    print("Evaluation results:", eval_results)
    
    # Make predictions on a test image
    # predictions = pipeline.predict('./test_image.jpg', threshold=0.5)
    # print("Predictions:", predictions)

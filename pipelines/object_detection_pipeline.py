"""
Object Detection Pipeline following the article's approach exactly.
Pipeline for Training Custom Faster-RCNN Object Detection models with PyTorch.
Compatible with the main.py architecture and dataset loading approach.
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
import time
from tqdm import tqdm

# Import the article-based components
from datasets_module.detection.coco_dataset_article import CocoDetectionDataset, get_transforms, collate_fn
from models.detection.model_factory import create_model
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
    Compatible with the main.py dataset loading approach.
    """
    
    def __init__(self, config):
        """
        Initialize the Object Detection Pipeline
        
        Args:
            config: Configuration object with pipeline parameters (from main.py)
        """
        super().__init__(config)
        
        # Extract configuration from the config object (maintains compatibility with main.py)
        self.num_classes = getattr(config, 'num_classes', 2)  # Including background
        # Always get the string value from Enum if present
        arch = getattr(config, 'architecture', 'faster_rcnn')
        if hasattr(arch, 'value'):
            self.architecture = arch.value.lower()
        else:
            self.architecture = str(arch).lower()
        self.learning_rate = getattr(config, 'learning_rate', 0.005)  # Article's default
        self.batch_size = getattr(config, 'batch_size', 2)  # Article's default
        self.num_epochs = getattr(config, 'epochs', 10)  # Note: uses 'epochs' from main.py
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Dataset paths (will be set when train() is called with dataset_path)
        self.dataset_path = None
        self.train_image_dir = None
        self.train_annotation_path = None
        self.val_image_dir = None
        self.val_annotation_path = None
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Class names for visualization (will be extracted from dataset)
        self.class_names = []
        
        # Training configuration following the article
        self.weight_decay = 0.0005  # Article's default
        self.momentum = 0.9  # Article's default
        self.step_size = 3  # Article's default
        self.gamma = 0.1  # Article's default
        
        # MLflow integration
        self.run_id = None
        
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
    
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """
        Main training method compatible with main.py approach
        
        Args:
            dataset_path: Path to the dataset directory
            job_id: Job ID for tracking and logging
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            print(f"Starting training for job {job_id}")
            
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.__dict__)
            
            # Set dataset path
            self.dataset_path = dataset_path
            
            # Load datasets following the article's approach but adapted for our structure
            await self._load_datasets_from_path(dataset_path, job_id)
            
            # Build model following the article's approach
            self._build_model()
            
            # Train the model
            training_results = await self._train_model(job_id)
            
            # End MLflow run
            end_run()
            
            return training_results
            
        except Exception as e:
            print(f"Training failed for job {job_id}: {str(e)}")
            if self.run_id:
                end_run()
            raise e

    async def _load_datasets_from_path(self, dataset_path: str, job_id: str) -> None:
        """
        Load datasets from the provided path, auto-detecting COCO format structure
        
        Args:
            dataset_path: Path to the dataset directory
            job_id: Job ID for logging
        """
        dataset_path = Path(dataset_path)
        print(f"Loading dataset from: {dataset_path}")
        
        # Try to detect COCO format structure
        # Look for common COCO dataset structures
        
        # Structure 1: train/val folders with images and annotations
        # Support common validation folder names: val, valid, validation, test
        train_dir = dataset_path / "train"
        val_dir = None
        
        # Look for validation folder with common names
        val_folder_names = ["val", "valid", "validation", "test"]
        for val_name in val_folder_names:
            potential_val_dir = dataset_path / val_name
            if potential_val_dir.exists():
                val_dir = potential_val_dir
                print(f"Found validation folder: {val_name}")
                break
        
        if train_dir.exists() and val_dir is not None:
            # Structure: dataset/train/images/, dataset/train/annotations.json
            # Or: dataset/train/, dataset/val/ with _annotations.coco.json
            
            # Check for annotations in train folder
            train_annotations = list(train_dir.glob("*annotations*.json")) + list(train_dir.glob("*.json"))
            val_annotations = list(val_dir.glob("*annotations*.json")) + list(val_dir.glob("*.json"))
            
            if train_annotations and val_annotations:
                self.train_image_dir = str(train_dir)
                self.train_annotation_path = str(train_annotations[0])
                self.val_image_dir = str(val_dir)
                self.val_annotation_path = str(val_annotations[0])
                
                # Check if images are in subdirectory
                if (train_dir / "images").exists():
                    self.train_image_dir = str(train_dir / "images")
                if (val_dir / "images").exists():
                    self.val_image_dir = str(val_dir / "images")
                    
                print(f"Found train/val structure with annotations")
                print(f"  Train dir: {self.train_image_dir}")
                print(f"  Train annotations: {self.train_annotation_path}")
                print(f"  Val dir: {self.val_image_dir}")
                print(f"  Val annotations: {self.val_annotation_path}")
        
        # Structure 2: images/ and annotations/ folders
        elif (dataset_path / "images").exists() and (dataset_path / "annotations").exists():
            images_dir = dataset_path / "images"
            annotations_dir = dataset_path / "annotations"
            
            # Look for train/val splits in annotations - support multiple val naming conventions
            train_ann = None
            val_ann = None
            
            # Look for training annotation file
            train_patterns = ["instances_train.json", "train.json", "_annotations.coco.json"]
            for pattern in train_patterns:
                potential_train = annotations_dir / pattern
                if potential_train.exists():
                    train_ann = potential_train
                    break
            
            # Look for validation annotation file with common naming patterns
            val_patterns = ["instances_val.json", "instances_valid.json", "instances_validation.json", 
                           "instances_test.json", "val.json", "valid.json", "validation.json", "test.json"]
            for pattern in val_patterns:
                potential_val = annotations_dir / pattern
                if potential_val.exists():
                    val_ann = potential_val
                    print(f"Found validation annotations: {pattern}")
                    break
            
            if train_ann and val_ann:
                # Check if images are split into train/val folders
                train_img_dirs = ["train", "training"]
                val_img_dirs = ["val", "valid", "validation", "test"]
                
                train_img_dir = None
                val_img_dir = None
                
                # Find training images directory
                for dir_name in train_img_dirs:
                    potential_dir = images_dir / dir_name
                    if potential_dir.exists():
                        train_img_dir = potential_dir
                        break
                
                # Find validation images directory  
                for dir_name in val_img_dirs:
                    potential_dir = images_dir / dir_name
                    if potential_dir.exists():
                        val_img_dir = potential_dir
                        print(f"Found validation images in: {dir_name}")
                        break
                
                if train_img_dir and val_img_dir:
                    self.train_image_dir = str(train_img_dir)
                    self.val_image_dir = str(val_img_dir)
                else:
                    # Images not split, use same directory for both
                    self.train_image_dir = str(images_dir)
                    self.val_image_dir = str(images_dir)
                
                self.train_annotation_path = str(train_ann)
                self.val_annotation_path = str(val_ann)
                
                print(f"Found images/annotations structure")
                print(f"  Train dir: {self.train_image_dir}")
                print(f"  Train annotations: {self.train_annotation_path}")
                print(f"  Val dir: {self.val_image_dir}")
                print(f"  Val annotations: {self.val_annotation_path}")
        
        # Structure 3: Single level with various folder names
        else:
            # Look for any combination of train/validation folders
            print(f"Checking directory contents: {list(dataset_path.iterdir())}")
            
            # Find any folders that might contain data
            all_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
            train_candidates = []
            val_candidates = []
            
            for folder in all_folders:
                folder_name = folder.name.lower()
                if 'train' in folder_name:
                    train_candidates.append(folder)
                elif any(val_name in folder_name for val_name in ['val', 'valid', 'validation', 'test']):
                    val_candidates.append(folder)
            
            # Try to find JSON files in these folders
            for train_folder in train_candidates:
                train_jsons = list(train_folder.glob("*.json"))
                if train_jsons:
                    for val_folder in val_candidates:
                        val_jsons = list(val_folder.glob("*.json"))
                        if val_jsons:
                            # Found a valid pair
                            self.train_image_dir = str(train_folder)
                            self.train_annotation_path = str(train_jsons[0])
                            self.val_image_dir = str(val_folder)
                            self.val_annotation_path = str(val_jsons[0])
                            
                            print(f"Found data structure:")
                            print(f"  Train folder: {train_folder.name}")
                            print(f"  Val folder: {val_folder.name}")
                            break
                if self.train_image_dir:  # Break outer loop if found
                    break
            
            # If still not found, look for any JSON file in root
            if not self.train_image_dir:
                root_jsons = list(dataset_path.glob("*.json"))
                if root_jsons and all_folders:
                    # Use first JSON and first folder with images
                    for folder in all_folders:
                        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
                        if images:
                            # Use same folder for both train and val
                            self.train_image_dir = str(folder)
                            self.train_annotation_path = str(root_jsons[0])
                            self.val_image_dir = str(folder)
                            self.val_annotation_path = str(root_jsons[0])
                            
                            print(f"Using single folder structure:")
                            print(f"  Images folder: {folder.name}")
                            print(f"  Annotation: {root_jsons[0].name}")
                            break
        
        # If no valid structure found, raise error
        if not all([self.train_image_dir, self.train_annotation_path, 
                   self.val_image_dir, self.val_annotation_path]):
            available_files = list(dataset_path.rglob("*"))[:10]  # Show first 10 files
            raise ValueError(
                f"Could not detect valid COCO dataset structure in {dataset_path}. "
                "Expected structures:\n"
                "1. dataset/train/ and dataset/val|valid|validation|test/ with annotations.json files\n"
                "2. dataset/images/ and dataset/annotations/ with instances_train.json and instances_val|valid|validation|test.json\n"
                "3. Any folder structure with train/val folders containing .json annotation files\n"
                f"Available files (first 10): {[str(f.relative_to(dataset_path)) for f in available_files]}"
            )
        
        print(f"Dataset structure detected:")
        print(f"  Train images: {self.train_image_dir}")
        print(f"  Train annotations: {self.train_annotation_path}")
        print(f"  Val images: {self.val_image_dir}")
        print(f"  Val annotations: {self.val_annotation_path}")
        
        # Load data using the article's approach
        await self._update_job_log(job_id, "Creating dataloaders...")
        
        try:
            from datasets_module.detection.coco_dataset_article import create_dataloaders
            
            self.train_loader, self.val_loader, detected_num_classes = create_dataloaders(
                train_image_dir=self.train_image_dir,
                train_annotation_path=self.train_annotation_path,
                val_image_dir=self.val_image_dir,
                val_annotation_path=self.val_annotation_path,
                batch_size=self.batch_size,
                num_workers=4
            )
            
            # Update num_classes if detected from dataset
            if detected_num_classes > 0:
                self.num_classes = detected_num_classes
                print(f"Updated num_classes to {self.num_classes} based on dataset")
            
            # Extract class names from COCO annotations using the same deduplication logic as the dataset
            from pycocotools.coco import COCO
            coco = COCO(self.train_annotation_path)
            
            # Get all categories and deduplicate by name (same logic as in dataset)
            category_ids = sorted(coco.getCatIds())
            categories = coco.loadCats(category_ids)
            
            # Get unique class names (same as in the dataset)
            unique_class_names = []
            seen_names = set()
            
            for cat in categories:
                class_name = cat['name']
                if class_name not in seen_names:
                    unique_class_names.append(class_name)
                    seen_names.add(class_name)
            
            # Debug: show category structure (can be commented out for production)
            # print(f"All categories: {[(cat['id'], cat['name']) for cat in categories]}")
            print(f"Detected {len(unique_class_names)} unique classes: {unique_class_names}")
            
            # Create class names list (background + unique categories)
            self.class_names = ['background'] + unique_class_names
            
            print(f"Final class configuration:")
            print(f"  - Classes: {self.class_names}")
            print(f"  - Total classes (including background): {self.num_classes}")
            
            await self._update_job_log(job_id, f"Loaded {len(unique_class_names)} unique classes: {unique_class_names}")
            
        except Exception as e:
            raise ValueError(f"Failed to load COCO datasets: {str(e)}")

    def _build_model(self) -> None:
        """
        Build the detection model following the article's approach
        """
        print("Building Faster R-CNN model...")
        
        # Ensure we're using Faster R-CNN (following the article)
        if self.architecture != 'faster_rcnn':
            print(f"Warning: Architecture '{self.architecture}' not supported. Using 'faster_rcnn' as per article.")
            self.architecture = 'faster_rcnn'
        
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
        
        # Learning rate scheduler (article uses StepLR)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )
        
        print("Model built successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    async def _train_model(self, job_id: str) -> Dict[str, Any]:
        """
        Train the model using the article's approach with PyTorch Vision utilities
        
        Args:
            job_id: Job ID for logging
            
        Returns:
            Dictionary with training results
        """
        print("Starting training loop...")
        
        # Log training parameters
        from mlflow_utils import log_metrics
        log_metrics({
            'num_classes': self.num_classes,
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
            
            await self._update_job_log(job_id, f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch using PyTorch Vision's utilities if available
            if PYTORCH_VISION_AVAILABLE:
                try:
                    # Import here to avoid issues if not available
                    from torchvision.references.detection.engine import train_one_epoch, evaluate
                    
                    # Train one epoch
                    train_one_epoch(self.model, self.optimizer, self.train_loader, 
                                  self.device, epoch, print_freq=10)
                    
                    # Evaluate
                    eval_results = evaluate(self.model, self.val_loader, device=self.device)
                    
                    # Extract metrics
                    train_loss = 0.0  # train_one_epoch doesn't return loss directly
                    val_loss = eval_results.coco_eval['bbox'].stats[0] if hasattr(eval_results, 'coco_eval') else 0.0
                    
                    epoch_metrics = {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'mAP': val_loss  # Use mAP as validation metric
                    }
                    
                except Exception as e:
                    print(f"PyTorch Vision utilities failed, using custom training: {e}")
                    epoch_metrics = await self._train_epoch_custom(epoch, job_id)
            else:
                # Use custom training loop
                epoch_metrics = await self._train_epoch_custom(epoch, job_id)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Log metrics
            log_metrics(epoch_metrics, step=epoch)
            training_metrics.append(epoch_metrics)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            await self._update_job_log(
                job_id, 
                f"Epoch {epoch + 1}/{self.num_epochs} completed in {epoch_time:.2f}s - "
                f"Loss: {epoch_metrics.get('train_loss', 0):.4f}"
            )
        
        # Save only the final model to logs folder
        model_path = await self._save_model(job_id, f"model_final.pth", save_dir="logs")
        return {
            'status': 'completed',
            'metrics': training_metrics,
            'best_val_loss': best_val_loss,
            'model_path': model_path,
            'num_epochs': self.num_epochs,
            'final_lr': self.optimizer.param_groups[0]['lr']
        }

    async def _train_epoch_custom(self, epoch: int, job_id: str) -> Dict[str, float]:
        """
        Custom training loop for one epoch
        
        Args:
            epoch: Current epoch number
            job_id: Job ID for logging
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move data to device with proper type checking
            images = [img.to(self.device) for img in images]
            
            # Move targets to device, handling different data types
            processed_targets = []
            for target in targets:
                processed_target = {}
                for k, v in target.items():
                    if torch.is_tensor(v):
                        processed_target[k] = v.to(self.device)
                    else:
                        # Convert non-tensors to appropriate tensor type
                        if k == 'image_id':
                            processed_target[k] = torch.tensor([v] if isinstance(v, int) else v, dtype=torch.int64).to(self.device)
                        else:
                            processed_target[k] = torch.tensor(v).to(self.device)
                processed_targets.append(processed_target)
            
            targets = processed_targets
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # Ensure model is in training mode
            self.model.train()
            outputs = self.model(images, targets)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # Training mode: outputs is a dict of losses
                loss_dict = outputs
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(outputs, list):
                # Evaluation mode: outputs is a list of predictions
                # This shouldn't happen during training, but handle it gracefully
                print("Warning: Model returned predictions instead of losses. Switching to training mode.")
                self.model.train()
                outputs = self.model(images, targets)
                if isinstance(outputs, dict):
                    loss_dict = outputs
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    raise RuntimeError("Model not returning losses in training mode")
            else:
                raise RuntimeError(f"Unexpected model output type: {type(outputs)}")
            
            # Backward pass
            losses.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses.item()
            num_batches += 1
            
            # Log every 10 batches
            if batch_idx % 10 == 0:
                await self._update_job_log(
                    job_id, 
                    f"Epoch {epoch + 1}, Batch {batch_idx}: Loss = {losses.item():.4f}"
                )
        
        # Evaluate on validation set
        val_loss = await self._evaluate_epoch_custom()
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        }

    async def _evaluate_epoch_custom(self) -> float:
        """
        Custom evaluation for one epoch
        
        Returns:
            Average validation loss
        """
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move data to device with proper type checking
                images = [img.to(self.device) for img in images]
                
                # Move targets to device, handling different data types
                processed_targets = []
                for target in targets:
                    processed_target = {}
                    for k, v in target.items():
                        if torch.is_tensor(v):
                            processed_target[k] = v.to(self.device)
                        else:
                            # Convert non-tensors to appropriate tensor type
                            if k == 'image_id':
                                processed_target[k] = torch.tensor([v] if isinstance(v, int) else v, dtype=torch.int64).to(self.device)
                            else:
                                processed_target[k] = torch.tensor(v).to(self.device)
                    processed_targets.append(processed_target)
                
                targets = processed_targets
                
                # Forward pass
                # Ensure model is in training mode for validation loss calculation
                self.model.train()
                outputs = self.model(images, targets)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    # Training mode: outputs is a dict of losses
                    loss_dict = outputs
                    losses = sum(loss for loss in loss_dict.values())
                elif isinstance(outputs, list):
                    # If we get predictions, skip this batch (shouldn't happen in training mode)
                    print("Warning: Model returned predictions instead of losses during evaluation.")
                    continue
                else:
                    print(f"Warning: Unexpected model output type during evaluation: {type(outputs)}")
                    continue
                
                total_loss += losses.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    async def _save_model(self, job_id: str, filename: str, save_dir: str = "logs") -> str:
        """
        Save the trained model
        
        Args:
            job_id: Job ID
            filename: Filename for the saved model
            save_dir: Directory to save the model (default: logs)
            
        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create logs directory structure
        models_dir = Path(save_dir) / job_id
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'num_classes': self.num_classes,
                'architecture': self.architecture,
                'class_names': self.class_names
            },
            'training_info': {
                'job_id': job_id,
                'device': str(self.device),
                'num_epochs': self.num_epochs
            }
        }, model_path)
        
        print(f"Model saved to {model_path}")
        return str(model_path)

    async def _update_job_log(self, job_id: str, message: str):
        """Update the job log with a message"""
        print(f"Job {job_id}: {message}")
        # This would integrate with the job management system in main.py
    
    async def predict(self, image, model=None) -> Dict[str, Any]:
        """
        Make predictions on a single image following the article's approach.
        Compatible with main.py interface.
        
        Args:
            image: PIL Image or image path
            model: Optional model (for compatibility with main.py)
            
        Returns:
            Dictionary with predictions
        """
        # Use provided model or default to self.model
        prediction_model = model if model is not None else self.model
        
        if prediction_model is None:
            raise ValueError("No model available for prediction. Train a model first.")
        
        # Handle different input types
        if isinstance(image, str):
            # Image path provided
            from PIL import Image as PILImage
            image = PILImage.open(image).convert("RGB")
        elif hasattr(image, 'convert'):
            # PIL Image provided
            image = image.convert("RGB")
        
        # Preprocess image following the article's approach
        from torchvision.transforms import ToTensor
        transform = ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        prediction_model.eval()
        with torch.no_grad():
            predictions = prediction_model(image_tensor)

        # Robustly handle prediction output
        detections = []
        # Check if predictions is a non-empty list
        if isinstance(predictions, list) and len(predictions) > 0:
            pred = predictions[0]
            # Check if pred is a dict and has required keys
            if isinstance(pred, dict) and all(k in pred for k in ['boxes', 'scores', 'labels']):
                # Apply threshold
                threshold = 0.5  # Default threshold
                keep = pred['scores'] > threshold
                if keep.any():
                    boxes = pred['boxes'][keep].cpu().numpy()
                    scores = pred['scores'][keep].cpu().numpy()
                    labels = pred['labels'][keep].cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        detection = {
                            'box': box.tolist(),  # [x1, y1, x2, y2]
                            'confidence': float(score * 100),  # Convert to percentage
                            'class_name': self.class_names[label] if label < len(self.class_names) else f'class_{label}',
                            'label': int(label)
                        }
                        detections.append(detection)
                # else: detections remains empty
            else:
                # pred is missing required keys, return empty detections
                pass
        # else: predictions is empty or not a list, return empty detections

        return {
            'predictions': detections,
            'status': 'success',
            'num_detections': len(detections)
        }

    async def evaluate(self, dataset_path: str = None, model_path: str = None, job_id: str = None) -> Dict[str, Any]:
        """
        Evaluate the model on a test dataset
        
        Args:
            dataset_path: Path to test dataset
            model_path: Path to saved model (optional)
            job_id: Optional job ID for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not self.model and not model_path:
                raise ValueError("No model available for evaluation")
            
            # Load model if path provided
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            
            # Set up evaluation dataset
            if not self.val_loader:
                await self._load_datasets_from_path(dataset_path, job_id or "eval")
            
            # Run evaluation
            val_loss = await self._evaluate_epoch_custom()
            eval_metrics = {'val_loss': val_loss}
            
            return {
                'status': 'completed',
                'metrics': eval_metrics
            }
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

    @staticmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        return [
            'train_loss',
            'val_loss',
            'learning_rate',
            'epoch_time'
        ]
    
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

    def create_model(self) -> nn.Module:
        """
        Create a model based on the architecture specified in config
        Following the article's approach for Faster R-CNN
        """
        # Create model using the factory (following the article)
        model = create_model(
            architecture=self.architecture,
            num_classes=self.num_classes,
            pretrained=True
        )
        
        # Move to device
        model.to(self.device)
        
        return model
    
    def get_transforms(self):
        """Get data transforms for object detection following the article's approach"""
        # Use the transforms from the article-based COCO dataset
        return get_transforms(train=True)  # Default to training transforms


# Factory function for compatibility with main.py
def create_detection_pipeline(config) -> ObjectDetectionPipeline:
    """
    Factory function to create an ObjectDetectionPipeline
    Compatible with main.py's PipelineFactory
    
    Args:
        config: Configuration object from main.py
        
    Returns:
        Configured ObjectDetectionPipeline instance
    """
    return ObjectDetectionPipeline(config)



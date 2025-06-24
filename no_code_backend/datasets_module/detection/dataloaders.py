"""
Dataloaders for object detection tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import glob

class ObjectDetectionDataset(Dataset):
    """Dataset for object detection tasks"""
    
    def __init__(self, images_dir: str, annotations_path: str, transform=None):
        """
        Initialize object detection dataset
        
        Args:
            images_dir: Path to the images directory
            annotations_path: Path to the annotations JSON file
            transform: Optional transform to apply to both image and targets
        
        The dataset structure is flexible:
        - Structured by split:
          datasets/detection_data/
          ├── images/
          │   ├── train/
          │   ├── val/
          │   └── test/
          ├── annotations/
          │   ├── instances_train.json
          │   ├── instances_val.json
          │   └── instances_test.json
          
        - Or annotations file can be in the same directory as images:
          datasets/detection_data/
          ├── train/
          │   ├── images/
          │   ├── annotations.json
          ├── val/
          │   ├── images/
          │   ├── annotations.json
          ├── test/
          │   ├── images/
          │   ├── annotations.json
        """
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        if not self.annotations_path.exists():
            raise ValueError(f"Annotations file not found: {self.annotations_path}")
        
        # Load annotations
        with open(self.annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Create class ID to name mapping
        self.categories = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        self.num_classes = len(self.categories)
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.annotations["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # Create image ID to image metadata mapping
        self.images = {img["id"]: img for img in self.annotations["images"]}
        
        # Create a list of image IDs that have annotations
        self.image_ids = list(self.image_annotations.keys())
        
    def get_categories(self):
        """Get category mapping"""
        return self.categories
    
    def get_class_names(self):
        """Get list of class names"""
        # Sort by ID to ensure consistent order
        return [self.categories[i] for i in sorted(self.categories.keys())]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image ID and metadata
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        # Check if the file exists directly in the images directory
        img_path = self.images_dir / img_info["file_name"]
        if not img_path.exists():
            # Try looking for the image in subdirectories
            potential_paths = list(self.images_dir.glob(f"**/{img_info['file_name']}"))
            if potential_paths:
                img_path = potential_paths[0]
            else:
                raise FileNotFoundError(f"Image not found: {img_info['file_name']}")
        
        image = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        img_anns = self.image_annotations[img_id]
        
        # Create target dictionary with boxes and labels
        boxes = []
        labels = []
        
        for ann in img_anns:
            # COCO bbox format is [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format for PyTorch
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            # Adjust category_id to be 0-indexed (COCO uses 1-indexed categories)
            labels.append(ann["category_id"] - 1)  # Convert to 0-indexed
        
        # Convert to tensors
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            # Empty annotations case
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
        
        # Add image_id for evaluation
        target["image_id"] = torch.tensor([img_id])
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target


def find_annotation_file(directory, pattern="*.json"):
    """Find annotation file in a directory"""
    json_files = list(Path(directory).glob(pattern))
    if not json_files:
        return None
    # Prefer files with 'instances' or 'annotations' in the name
    for keyword in ['instances', 'annotation']:
        for file in json_files:
            if keyword in file.name.lower():
                return file
    # If no preferred file found, return the first one
    return json_files[0]


def create_dataloaders(dataset_path: str, transform=None, batch_size: int = 4, 
                      val_split: float = 0.2, num_workers: int = 4, collate_fn=None):
    """
    Create train and validation dataloaders for object detection
    
    This function supports both pre-split datasets and auto-splitting:
    
    1. Pre-split structure:
       dataset_path/
       ├── images/
       │   ├── train/
       │   ├── val/
       │   └── test/
       ├── annotations/
       │   ├── instances_train.json
       │   ├── instances_val.json
       │   └── instances_test.json
    
    2. Alternative pre-split structure:
       dataset_path/
       ├── train/
       │   ├── images/
       │   ├── annotations.json
       ├── val/
       │   ├── images/
       │   ├── annotations.json
       
    3. Auto-split from single dataset:
       dataset_path/
       ├── images/
       ├── annotations.json
    
    Args:
        dataset_path: Path to dataset directory
        transform: Transforms to apply to images and targets
        batch_size: Batch size
        val_split: Validation split ratio (0 to 1)
        num_workers: Number of workers for data loading
        collate_fn: Custom collate function for batching
        
    Returns:
        train_loader, val_loader, class_names
    """
    dataset_path = Path(dataset_path)
    
    # Define a custom collate function if not provided
    if collate_fn is None:
        collate_fn = lambda batch: tuple(zip(*batch))
    
    # Check if this is a pre-split dataset
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    images_dir = dataset_path / "images"
    annotations_dir = dataset_path / "annotations"
    
    # Case 1: Standard COCO split structure with images/train + annotations/instances_train.json
    if images_dir.exists() and annotations_dir.exists():
        train_images_dir = images_dir / "train"
        val_images_dir = images_dir / "val"
        
        train_ann_path = annotations_dir / "instances_train.json"
        val_ann_path = annotations_dir / "instances_val.json"
        
        if not train_ann_path.exists():
            # Try to find any annotation file for train
            train_ann_path = find_annotation_file(annotations_dir, "*train*.json")
        
        if not val_ann_path.exists():
            # Try to find any annotation file for val
            val_ann_path = find_annotation_file(annotations_dir, "*val*.json")
        
        if train_images_dir.exists() and train_ann_path and train_ann_path.exists():
            print(f"Using pre-split dataset: {train_images_dir} with {train_ann_path}")
            train_dataset = ObjectDetectionDataset(train_images_dir, train_ann_path, transform)
            
            if val_images_dir.exists() and val_ann_path and val_ann_path.exists():
                val_dataset = ObjectDetectionDataset(val_images_dir, val_ann_path, transform)
            else:
                val_dataset = None
                
            class_names = train_dataset.get_class_names()
                
        else:
            # Fall back to case 3
            print("Standard COCO structure not found. Falling back to auto-split.")
            return _create_dataloaders_auto_split(dataset_path, transform, batch_size, val_split, num_workers, collate_fn)
    
    # Case 2: Alternative structure with train/images + train/annotations.json
    elif train_dir.exists():
        train_images_dir = train_dir / "images"
        if not train_images_dir.exists():
            # Check if images are directly in the train directory
            image_files = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.png"))
            if image_files:
                train_images_dir = train_dir
        
        train_ann_path = find_annotation_file(train_dir)
        if not train_ann_path:
            train_ann_path = train_dir / "annotations.json"
            
        if train_images_dir.exists() and train_ann_path and train_ann_path.exists():
            print(f"Using alternative pre-split dataset: {train_images_dir} with {train_ann_path}")
            train_dataset = ObjectDetectionDataset(train_images_dir, train_ann_path, transform)
            
            # Check for validation set
            val_dataset = None
            if val_dir.exists():
                val_images_dir = val_dir / "images"
                if not val_images_dir.exists():
                    # Check if images are directly in the val directory
                    image_files = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.jpeg")) + list(val_dir.glob("*.png"))
                    if image_files:
                        val_images_dir = val_dir
                
                val_ann_path = find_annotation_file(val_dir)
                if not val_ann_path:
                    val_ann_path = val_dir / "annotations.json"
                    
                if val_images_dir.exists() and val_ann_path and val_ann_path.exists():
                    val_dataset = ObjectDetectionDataset(val_images_dir, val_ann_path, transform)
            
            class_names = train_dataset.get_class_names()
        else:
            # Fall back to case 3
            print("Alternative structure not found. Falling back to auto-split.")
            return _create_dataloaders_auto_split(dataset_path, transform, batch_size, val_split, num_workers, collate_fn)
    
    # Case 3: Auto-split from single dataset
    else:
        print("Pre-split dataset not detected. Using auto-split.")
        return _create_dataloaders_auto_split(dataset_path, transform, batch_size, val_split, num_workers, collate_fn)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, class_names


def _create_dataloaders_auto_split(dataset_path, transform, batch_size, val_split, num_workers, collate_fn):
    """Helper function to create dataloaders with auto-split"""
    dataset_path = Path(dataset_path)
    
    # Check for images directory and annotations file
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        # Images might be directly in the dataset_path
        images_dir = dataset_path
    
    # Find annotations file
    annotations_path = find_annotation_file(dataset_path)
    if not annotations_path:
        # Try in an annotations subdirectory
        annotations_dir = dataset_path / "annotations"
        if annotations_dir.exists():
            annotations_path = find_annotation_file(annotations_dir)
    
    if not annotations_path:
        raise ValueError(f"Cannot find annotations file in {dataset_path}")
    
    print(f"Using single dataset with auto-split: {images_dir} with {annotations_path}")
    
    # Create dataset
    dataset = ObjectDetectionDataset(images_dir, annotations_path, transform)
    
    # Get class names
    class_names = dataset.get_class_names()
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    if val_size > 0:
        # Use random split if validation set is requested
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        # Use the entire dataset for training if no validation is requested
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        val_loader = None
    
    return train_loader, val_loader, class_names

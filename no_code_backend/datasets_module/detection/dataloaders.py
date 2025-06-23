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

class ObjectDetectionDataset(Dataset):
    """Dataset for object detection tasks"""
    
    def __init__(self, dataset_path: str, transform=None):
        """
        Initialize object detection dataset
        
        Args:
            dataset_path: Path to the dataset directory
            transform: Optional transform to apply to both image and targets
        
        The dataset directory should have the following structure:
        - dataset_path/
          - images/
            - image1.jpg
            - image2.jpg
            - ...
          - annotations.json
          
        The annotations.json file should follow COCO format or a simplified version:
        {
            "images": [
                {"id": 1, "file_name": "image1.jpg", "width": 800, "height": 600},
                {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600},
                ...
            ],
            "annotations": [
                {"image_id": 1, "bbox": [x, y, width, height], "category_id": 1},
                {"image_id": 1, "bbox": [x, y, width, height], "category_id": 2},
                ...
            ],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "car"},
                ...
            ]
        }
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load and parse annotations
        self.images_dir = self.dataset_path / "images"
        self.annotations_file = self.dataset_path / "annotations.json"
        
        if not self.images_dir.exists() or not self.annotations_file.exists():
            raise ValueError(f"Dataset directory structure is invalid. "
                             f"Expected 'images' folder and 'annotations.json' in {dataset_path}")
        
        # Load annotations
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create class ID to name mapping
        self.categories = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
        # No need for +1 as we're shifting category IDs down by 1 (0-indexed)
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
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image ID and metadata
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.images_dir / img_info["file_name"]
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
            labels.append(ann["category_id"] - 1)
        
        # Convert to tensors
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        else:
            # Empty annotations case
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target

def create_dataloaders(dataset_path: str, transform=None, batch_size: int = 4, 
                      val_split: float = 0.2, num_workers: int = 4, collate_fn=None):
    """
    Create train and validation dataloaders for object detection
    
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
    # Create dataset
    dataset = ObjectDetectionDataset(dataset_path, transform)
    
    # Get class names
    class_names = list(dataset.categories.values())
    
    # Define a custom collate function if not provided
    if collate_fn is None:
        collate_fn = lambda batch: tuple(zip(*batch))
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
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
    ) if val_size > 0 else None
    
    return train_loader, val_loader, class_names

"""
Dataloaders for image segmentation tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class SemanticSegmentationDataset(Dataset):
    """Dataset for semantic segmentation tasks"""
    
    def __init__(self, dataset_path: str, transform=None):
        """
        Initialize semantic segmentation dataset
        
        Args:
            dataset_path: Path to the dataset directory
            transform: Optional transform to apply to both image and masks
        
        The dataset directory should have the following structure:
        - dataset_path/
          - images/
            - image1.jpg
            - image2.jpg
            - ...
          - masks/
            - mask1.png
            - mask2.png
            - ...
          - classes.json (optional)
        
        The masks should be single-channel PNG images where pixel values represent class indices.
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load and parse data paths
        self.images_dir = self.dataset_path / "images"
        self.masks_dir = self.dataset_path / "masks"
        
        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise ValueError(f"Dataset directory structure is invalid. "
                           f"Expected 'images' and 'masks' folders in {dataset_path}")
        
        # Get all image files (both jpg and png)
        self.images = sorted([img for img in self.images_dir.glob("*.jpg")] + [img for img in self.images_dir.glob("*.png")])
        self.masks = sorted([mask for mask in self.masks_dir.glob("*.png")])
        
        # Check if we have matching number of images and masks
        if len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) doesn't match number of masks ({len(self.masks)})")
        
        # Load class information if available
        self.classes_file = self.dataset_path / "classes.json"
        if self.classes_file.exists():
            with open(self.classes_file, 'r') as f:
                self.class_info = json.load(f)
                self.class_names = self.class_info.get('class_names', [])
        else:
            # Determine class names from unique values in masks
            unique_values = set()
            for mask_path in self.masks:
                mask = np.array(Image.open(mask_path))
                unique_values.update(np.unique(mask))
            self.class_names = [f"class_{i}" for i in sorted(unique_values)]
        
        self.num_classes = len(self.class_names)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # Don't convert masks - they should be single channel
        
        # Apply transforms if specified
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            # Default conversion to tensor
            from torchvision import transforms
            image = transforms.ToTensor()(image)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        
        return image, mask

class InstanceSegmentationDataset(Dataset):
    """Dataset for instance segmentation tasks"""
    
    def __init__(self, dataset_path: str, transform=None):
        """
        Initialize instance segmentation dataset
        
        Args:
            dataset_path: Path to the dataset directory
            transform: Optional transform to apply to both image and masks
        
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
                ...
            ],
            "annotations": [
                {"image_id": 1, "bbox": [x, y, width, height], "category_id": 1, "segmentation": [...], "iscrowd": 0},
                ...
            ],
            "categories": [
                {"id": 1, "name": "person"},
                ...
            ]
        }
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        
        # Load and parse data paths
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
        
        # Create a mapping from original category IDs to consecutive zero-indexed IDs
        # This ensures category IDs are consecutive integers starting from 1 (0 is background)
        orig_ids = sorted(self.categories.keys())
        self.category_mapping = {orig_id: idx + 1 for idx, orig_id in enumerate(orig_ids)}  # Add +1 to map to 1-indexed
        self.class_names = [self.categories[orig_id] for orig_id in orig_ids]
        self.num_classes = len(self.categories) + 1  # +1 for background
        
        print(f"Category mapping: {self.category_mapping}")
        print(f"Number of classes (including background): {self.num_classes}")
        
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
        
        # Create target dictionary with boxes, masks, and labels
        boxes = []
        masks = []
        labels = []
        
        for ann in img_anns:
            # COCO bbox format is [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format for PyTorch
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            
            # Map the original category ID to zero-indexed consecutive IDs
            # This ensures category IDs start at 1 (0 is background) and are consecutive
            orig_category_id = ann["category_id"]
            mapped_category_id = self.category_mapping[orig_category_id]
            labels.append(mapped_category_id)
            
            # Convert segmentation to binary mask
            from pycocotools import mask as mask_utils
            if isinstance(ann["segmentation"], list):  # Polygon format
                h, w = img_info["height"], img_info["width"]
                rles = mask_utils.frPyObjects(ann["segmentation"], h, w)
                mask = mask_utils.decode(mask_utils.merge(rles))
            else:  # RLE format
                mask = mask_utils.decode(ann["segmentation"])
            
            masks.append(mask)
        
        # Convert to tensors
        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            
            # Check if the list is not empty
            if masks:
                # Stack along a new dimension if there are multiple masks
                target["masks"] = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            else:
                # Create an empty tensor with the right dimensions if no masks
                # Fix: Use img_info height and width instead of undefined variables
                target["masks"] = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
        else:
            # Empty annotations case
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
            target["masks"] = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
        
        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)
            
            # Ensure target is properly formatted after transforms
            if isinstance(target, dict):
                for k, v in target.items():
                    if not isinstance(v, torch.Tensor):
                        try:
                            # Handle different possible data types
                            if isinstance(v, (list, np.ndarray)):
                                if k == "masks" and isinstance(v, list):
                                    # Special handling for mask lists after transforms
                                    # Convert each mask to tensor first, then stack
                                    mask_tensors = []
                                    for mask in v:
                                        if isinstance(mask, (np.ndarray, torch.Tensor)):
                                            mask_tensors.append(torch.as_tensor(mask, dtype=torch.uint8))
                                        else:
                                            # Skip problematic masks or use zeros
                                            mask_tensors.append(torch.zeros((img_info["height"], img_info["width"]), 
                                                                           dtype=torch.uint8))
                                    
                                    if mask_tensors:
                                        target[k] = torch.stack(mask_tensors)
                                    else:
                                        target[k] = torch.zeros((0, img_info["height"], img_info["width"]), 
                                                              dtype=torch.uint8)
                                elif len(v) > 0 and isinstance(v[0], (list, np.ndarray)):
                                    # For nested structures like boxes
                                    target[k] = torch.as_tensor(v, dtype=torch.float32)
                                else:
                                    # For simple lists/arrays
                                    target[k] = torch.as_tensor(v)
                            elif v is None:
                                # Handle None values
                                if k == "masks":
                                    target[k] = torch.zeros((0, img_info["height"], img_info["width"]), dtype=torch.uint8)
                                elif k == "boxes":
                                    target[k] = torch.zeros((0, 4), dtype=torch.float32)
                                elif k == "labels":
                                    target[k] = torch.zeros((0), dtype=torch.int64)
                            else:
                                # Try direct conversion for other types
                                target[k] = torch.as_tensor(v)
                        except TypeError as e:
                            # If conversion fails, provide informative error
                            raise TypeError(f"Could not convert target['{k}'] of type {type(v)} to tensor: {e}")
        
        return image, target

def create_dataloaders(dataset_path: str, transform=None, batch_size: int = 4, 
                      val_split: float = 0.2, num_workers: int = 4, 
                      segmentation_type: str = "semantic"):
    """
    Create train and validation dataloaders for image segmentation
    
    Args:
        dataset_path: Path to dataset directory
        transform: Transforms to apply to images and masks
        batch_size: Batch size
        val_split: Validation split ratio (0 to 1)
        num_workers: Number of workers for data loading
        segmentation_type: Type of segmentation ('semantic' or 'instance')
        
    Returns:
        train_loader, val_loader, class_names, num_classes
    """
    # Select the appropriate dataset class based on segmentation type
    if segmentation_type == "semantic":
        dataset_class = SemanticSegmentationDataset
        collate_fn = None  # Default collate function is fine for semantic segmentation
    elif segmentation_type == "instance":
        dataset_class = InstanceSegmentationDataset
        # Custom collate function for instance segmentation
        collate_fn = lambda batch: tuple(zip(*batch))
    else:
        raise ValueError(f"Unsupported segmentation type: {segmentation_type}")
    
    # Create dataset
    dataset = dataset_class(dataset_path, transform)
    
    # Get class information
    if segmentation_type == "semantic":
        class_names = dataset.class_names
        num_classes = len(class_names)
    else:  # instance segmentation
        class_names = dataset.class_names
        num_classes = dataset.num_classes  # This includes background class
    
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
    
    return train_loader, val_loader, class_names, num_classes

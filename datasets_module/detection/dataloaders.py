"""
Dataloaders for object detection tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import time
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import glob
import importlib
import random


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
          ├── vali/
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
        
        # Debug: Print original COCO categories
        print(f"Original COCO categories in {self.annotations_path}:")
        for cat in self.annotations["categories"]:
            print(f"  ID: {cat['id']}, Name: '{cat['name']}'")
            if 'supercategory' in cat:
                print(f"    Supercategory: '{cat['supercategory']}'")
        
        # Create class ID to name mapping - filter out supercategories
        self.categories = {}
        for cat in self.annotations["categories"]:
            # Only include actual categories, not supercategories
            # Skip categories that might be supercategories (check if it's used in annotations)
            category_id = cat["id"]
            category_name = cat["name"]
            
            # Check if this category is actually used in annotations
            is_used = any(ann["category_id"] == category_id for ann in self.annotations["annotations"])
            
            if is_used:
                self.categories[category_id] = category_name
                print(f"Added category: ID {category_id} -> '{category_name}'")
            else:
                print(f"Skipping unused category: ID {category_id} -> '{category_name}'")
        
        self.num_classes = len(self.categories)
        print(f"Final categories: {self.categories}")
        print(f"Total classes: {self.num_classes}")
        
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
        """Get list of class names - only return classes that are actually used in annotations"""
        # Sort by ID to ensure consistent order, but only include used categories
        used_class_names = [self.categories[i] for i in sorted(self.categories.keys())]
        print(f"Returning class names: {used_class_names}")
        return used_class_names
    
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
        
        # Create mapping from category_id to 0-indexed labels
        category_id_to_label = {cat_id: idx for idx, cat_id in enumerate(sorted(self.categories.keys()))}
        
        for ann in img_anns:
            # COCO bbox format is [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format for PyTorch
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            
            # Map category_id to 0-indexed label using our filtered categories
            category_id = ann["category_id"]
            if category_id in category_id_to_label:
                labels.append(category_id_to_label[category_id])
            else:
                print(f"Warning: Found annotation with unused category_id {category_id}, skipping")
                # Skip this annotation since it's not in our filtered categories
                boxes.pop()  # Remove the box we just added
        
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
    
    def create_and_save_splits(self, val_split: float = 0.2, test_split: float = 0.1, job_id: str = None):
        """
        Create train/val/test splits and save them to a file for reproducibility
        
        Args:
            val_split: Proportion of data to use for validation
            test_split: Proportion of data to use for testing
            job_id: Unique job ID to identify the splits
            
        Returns:
            Dictionary with train/val/test indices
        """
        if job_id is None:
            # If no job ID, create random splits but don't persist them
            return self._create_splits(val_split, test_split)
        
        # Create dataset_splits directory if it doesn't exist
        splits_dir = Path("dataset_splits")
        splits_dir.mkdir(exist_ok=True)
        
        # Create job-specific directory
        job_splits_dir = splits_dir / job_id
        job_splits_dir.mkdir(exist_ok=True)
        
        # Define path for splits file
        splits_file = job_splits_dir / "dataset_splits.json"
        
        # Create and save splits
        splits = self._create_splits(val_split, test_split)
        
        # Add extra metadata for verification when reloading
        splits_data = {
            "dataset_path": str(self.images_dir.absolute()),
            "annotations_path": str(self.annotations_path.absolute()),
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"],
            "created_at": time.time()  # Add timestamp for reference
        }
        
        # Save to file
        with open(splits_file, 'w') as f:
            json.dump(splits_data, f)
            
        print(f"Dataset splits saved to {splits_file}")
        
        return splits
        
    def _create_splits(self, val_split: float = 0.2, test_split: float = 0.1):
        """
        Create train/val/test splits without persisting them
        
        Args:
            val_split: Proportion of data to use for validation
            test_split: Proportion of data to use for testing
            
        Returns:
            Dictionary with train/val/test indices
        """
        # Get total number of samples
        n_samples = len(self)
        
        # Calculate number of validation and test samples
        n_val = max(1, int(n_samples * val_split))
        n_test = max(1, int(n_samples * test_split))
        
        # If the dataset is too small for both validation and test, prioritize validation
        if n_val + n_test > n_samples - 1:
            if n_samples <= 2:
                # Extremely small dataset, use same samples for everything
                n_val = 1 if n_samples > 1 else 0
                n_test = 0
            else:
                # Give at least 1 sample for training
                n_val = max(1, int(n_samples * 0.5))
                n_test = max(0, min(n_samples - n_val - 1, int(n_samples * 0.25)))
        
        # Create list of indices and shuffle them
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        # Split into train, validation, and test indices
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test+n_val]
        train_indices = indices[n_test+n_val:]
        
        print(f"Created splits: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
        
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }
    
    @staticmethod
    def load_splits(job_id: str):
        """
        Load saved splits for a given job ID
        
        Args:
            job_id: Unique job ID to identify the splits
            
        Returns:
            Dictionary with train/val/test indices
            
        Raises:
            FileNotFoundError: If no splits file found for the job ID
        """
        # Define path for splits file
        splits_file = Path("dataset_splits") / job_id / "dataset_splits.json"
        
        # Check if file exists
        if not splits_file.exists():
            raise FileNotFoundError(f"No splits file found at {splits_file}")
        
        # Load from file
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
            
        return splits_data
    
    @staticmethod
    def cleanup_splits(job_id: str):
        """
        Clean up saved splits for a given job ID
        
        Args:
            job_id: Unique job ID to identify the splits to clean up
            
        Returns:
            True if splits were cleaned up, False otherwise
        """
        # Define path for job-specific splits directory
        job_splits_dir = Path("dataset_splits") / job_id
        
        # Check if directory exists
        if not job_splits_dir.exists():
            print(f"No splits directory found at {job_splits_dir}, nothing to clean up")
            return False
        
        # Remove the directory and its contents
        import shutil
        shutil.rmtree(job_splits_dir)
        print(f"Cleaned up splits directory: {job_splits_dir}")
        return True


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


def collate_fn(batch):
    """
    Custom collate function for object detection batches
    """
    return tuple(zip(*batch))


def create_dataloaders(dataset_path: str, transform, batch_size: int = 2, 
                      val_split: float = 0.2, test_split: float = 0.1, 
                      num_workers: int = 4, job_id: str = None,
                      use_saved_splits: bool = False,
                      shuffle: bool = True):
    """Create training, validation and test dataloaders for object detection"""
    # Add support for persistent splits when using auto-split approach
    # For existing pre-split datasets, we'll use the existing splits

    dataset_path = Path(dataset_path)
    print(f"Analyzing dataset structure at {dataset_path}...")
    
    # Define common paths that might be found in the dataset structure
    images_dir = dataset_path / "images"
    annotations_dir = dataset_path / "annotations"
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    test_dir = dataset_path / "test"
    
    # Track if we found a pre-split dataset
    pre_split_dataset_found = False
    
    # Case 1: Standard dataset with images and annotations directories
    if images_dir.exists() and annotations_dir.exists():
        print(f"Found standard dataset structure with images and annotations directories")
        
        # Look for pre-split annotations
        train_ann_path = annotations_dir / "instances_train.json"
        val_ann_path = annotations_dir / "instances_val.json"
        test_ann_path = annotations_dir / "instances_test.json"
        
        # Look for single annotation file
        ann_files = list(annotations_dir.glob("*.json"))
        
        if train_ann_path.exists():
            print(f"Using standard pre-split dataset: {images_dir} with {train_ann_path}")
            pre_split_dataset_found = True
            
            # Create train dataset
            train_dataset = ObjectDetectionDataset(images_dir, train_ann_path, transform)
            
            # Create validation dataset if available
            val_dataset = None
            if val_ann_path.exists():
                val_dataset = ObjectDetectionDataset(images_dir, val_ann_path, transform)
            
            # Create test dataset if available
            test_dataset = None
            if test_ann_path.exists():
                test_dataset = ObjectDetectionDataset(images_dir, test_ann_path, transform)
                
            class_names = train_dataset.get_class_names()
            
        elif len(ann_files) == 1:
            # Single annotation file with images directory
            ann_path = ann_files[0]
            # Check if annotations file has explicit train/val/test splits defined internally
            with open(ann_path, 'r') as f:
                try:
                    annotations = json.load(f)
                    # Check if annotations have split information
                    has_splits = False
                    for img in annotations.get('images', [])[:10]:  # Check first few images
                        if 'split' in img:
                            has_splits = True
                            break
                    
                    if has_splits:
                        print(f"Found single annotation file with internal splits defined: {ann_path}")
                        pre_split_dataset_found = True
                        # Use the annotations file directly with ObjectDetectionDataset
                        # but filter by split when creating the datasets
                        # This would require extending the ObjectDetectionDataset class
                        # For now, we'll fall back to auto-split
                    
                except Exception as e:
                    print(f"Error reading annotation file {ann_path}: {e}")
    
    # Case 2: Alternative dataset structure with train/val/test directories
    if not pre_split_dataset_found and train_dir.exists():
        print(f"Found alternative dataset structure with train directory: {train_dir}")
        
        # Check for images directory in train directory
        train_images_dir = train_dir / "images"
        train_ann_path = None
        
        if not train_images_dir.exists():
            # Check if images are directly in the train directory
            image_files = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.png"))
            if image_files:
                train_images_dir = train_dir
        
        # Look for annotations file in train directory
        if train_images_dir.exists():
            train_ann_path = find_annotation_file(train_dir)
            if not train_ann_path:
                train_ann_path = train_dir / "annotations.json"
                if not train_ann_path.exists():
                    print(f"No annotation file found in {train_dir}")
                    train_ann_path = None
            
        if train_ann_path and train_ann_path.exists():
            print(f"Using alternative pre-split dataset: {train_images_dir} with {train_ann_path}")
            pre_split_dataset_found = True
            
            # Create train dataset
            train_dataset = ObjectDetectionDataset(train_images_dir, train_ann_path, transform)
            
            # Check for validation set
            val_dataset = None
            if val_dir.exists():
                val_images_dir = val_dir / "images"
                val_ann_path = None
                
                if not val_images_dir.exists():
                    # Check if images are directly in the val directory
                    image_files = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.jpeg")) + list(val_dir.glob("*.png"))
                    if image_files:
                        val_images_dir = val_dir
                
                if val_images_dir.exists():
                    val_ann_path = find_annotation_file(val_dir)
                    if not val_ann_path:
                        val_ann_path = val_dir / "annotations.json"
                        if not val_ann_path.exists():
                            print(f"No annotation file found in {val_dir}")
                            val_ann_path = None
                    
                    if val_ann_path and val_ann_path.exists():
                        val_dataset = ObjectDetectionDataset(val_images_dir, val_ann_path, transform)
                        print(f"Found validation dataset: {val_images_dir} with {val_ann_path}")
            
            # Check for test set
            test_dataset = None
            if test_dir.exists():
                test_images_dir = test_dir / "images"
                test_ann_path = None
                
                if not test_images_dir.exists():
                    # Check if images are directly in the test directory
                    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
                    if image_files:
                        test_images_dir = test_dir
                
                if test_images_dir.exists():
                    test_ann_path = find_annotation_file(test_dir)
                    if not test_ann_path:
                        test_ann_path = test_dir / "annotations.json"
                        if not test_ann_path.exists():
                            print(f"No annotation file found in {test_dir}")
                            test_ann_path = None
                    
                    if test_ann_path and test_ann_path.exists():
                        test_dataset = ObjectDetectionDataset(test_images_dir, test_ann_path, transform)
                        print(f"Found test dataset: {test_images_dir} with {test_ann_path}")
                        
            class_names = train_dataset.get_class_names()
    
    # Case 3: Direct structure with single annotations.json file at root and images directory
    # Check if we have a dataset with direct structure: single annotations file and images directory with images directly in it
    if not pre_split_dataset_found:
        annotations_file = dataset_path / "annotations.json"
        images_dir = dataset_path / "images"
        
        if annotations_file.exists() and images_dir.exists():
            # Check if the images directory contains images directly (not in subdirectories)
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
            
            if len(image_files) > 0:
                print(f"Found direct dataset structure with annotations.json and {len(image_files)} images in images directory")
                
                # Consider this a pre-split dataset - all data is considered training
                pre_split_dataset_found = True
                
                # Create a dataset using all available images
                train_dataset = ObjectDetectionDataset(images_dir, annotations_file, transform)
                class_names = train_dataset.get_class_names()
                
                # No val or test datasets in this simple case
                val_dataset = None
                test_dataset = None
                
                print(f"Direct structure dataset has {len(train_dataset)} samples and {len(class_names)} classes: {class_names}")
    
    # Auto-split if no pre-split dataset was found
    if not pre_split_dataset_found:
        print("Pre-split dataset not detected. Using auto-split.")
        return _create_dataloaders_auto_split(
            dataset_path, transform, batch_size, val_split, test_split, 
            num_workers, collate_fn, job_id, use_saved_splits
        )
    
    # If we get here, we've found a pre-split dataset
    print("Using pre-split dataset - no need to create splits")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader, class_names


def _create_dataloaders_auto_split(dataset_path, transform, batch_size, val_split, test_split, 
                               num_workers, collate_fn, job_id=None, use_saved_splits=False):
    """Helper function to create dataloaders with auto-split and support for persistent splits"""
    dataset_path = Path(dataset_path)
    
    print(f"Using auto-split dataloaders for {dataset_path}")
    
    # Check for images directory and annotations file
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        # Images might be directly in the dataset_path
        images_dir = dataset_path
        print(f"No 'images' directory found, using {images_dir} as images directory")
    
    # Find annotations file
    annotations_path = find_annotation_file(dataset_path)
    if not annotations_path:
        # Try in an annotations subdirectory
        annotations_dir = dataset_path / "annotations"
        if annotations_dir.exists():
            annotations_path = find_annotation_file(annotations_dir)
            if annotations_path:
                print(f"Found annotations file in annotations directory: {annotations_path}")
    
    if not annotations_path:
        raise ValueError(f"Cannot find annotations file in {dataset_path}")
    
    print(f"Using single dataset with auto-split: {images_dir} with {annotations_path}")
    
    # Create dataset
    dataset = ObjectDetectionDataset(images_dir, annotations_path, transform)
    
    # Get class names
    class_names = dataset.get_class_names()
    
    # Try to load saved splits if requested and job_id is provided
    splits = None
    if job_id:  # Only try to load/save splits if job_id is provided
        if use_saved_splits:
            try:
                print(f"Attempting to load saved splits for job ID {job_id}...")
                splits = ObjectDetectionDataset.load_splits(job_id)
                if (str(Path(images_dir).absolute()) != str(Path(splits['dataset_path']).absolute()) or
                    str(Path(annotations_path).absolute()) != str(Path(splits['annotations_path']).absolute())):
                    print(f"Warning: Saved splits are for a different dataset. Creating new splits.")
                    splits = None
                else:
                    print(f"Successfully loaded saved splits for job ID {job_id}")
            except FileNotFoundError:
                print(f"No saved splits found for job ID {job_id}. Creating new splits.")
        else:
            print(f"Not using saved splits. Creating new splits for job ID {job_id}")
    else:
        print(f"No job ID provided. Creating temporary splits (will not be saved)")
    
    # Create and save splits if not loaded
    if splits is None:
        if job_id:
            print(f"Creating and saving new splits for job ID {job_id}")
            splits = dataset.create_and_save_splits(
                val_split=val_split,
                test_split=test_split,
                job_id=job_id
            )
        else:
            print(f"Creating temporary splits (not saving to disk)")
            splits = dataset._create_splits(
                val_split=val_split,
                test_split=test_split
            )
    
    # Create dataset subsets
    print(f"Creating dataset subsets: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test samples")
    train_dataset = Subset(dataset, splits['train'])
    val_dataset = Subset(dataset, splits['val']) if len(splits['val']) > 0 else None
    test_dataset = Subset(dataset, splits['test']) if len(splits['test']) > 0 else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    test_loader = None
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader, test_loader, class_names

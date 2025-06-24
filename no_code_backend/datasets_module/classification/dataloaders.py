"""
Dataloaders for image classification tasks.
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class ImageClassificationDataset(Dataset):
    """Dataset for image classification tasks"""
    
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.classes = self._find_classes()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _find_classes(self) -> List[str]:
        """Find class folders in dataset"""
        classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        classes.sort()
        return classes
    
    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Create a list of (image_path, class_index) tuples"""
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            class_dir = self.dataset_path / target_class
            
            for img_path in class_dir.glob("*.*"):
                # Only include image files
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    samples.append((str(img_path), class_index))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Load image
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            
        # Apply transform if specified
        if self.transform:
            img = self.transform(img)
            
        return img, target
    
    def create_and_save_splits(self, val_split: float = 0.2, test_split: float = 0.1,
                              job_id: str = None, random_seed: int = 42) -> Dict[str, List[int]]:
        """Create and save train/val/test splits to a file for reproducibility
        
        Args:
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            job_id: Optional job ID to use in the filename
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split indices
        """
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Generate random indices for the entire dataset
        indices = np.random.permutation(len(self.samples))
        
        # Calculate split sizes
        test_size = int(test_split * len(indices))
        val_size = int(val_split * len(indices))
        train_size = len(indices) - val_size - test_size
        
        # Create splits
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size+val_size].tolist()
        test_indices = indices[train_size+val_size:].tolist()
        
        splits = {
            "train": train_indices,
            "val": val_indices, 
            "test": test_indices,
            "random_seed": random_seed,
            "dataset_path": str(self.dataset_path)
        }
        
        # Save splits to file if job_id is provided
        if job_id:
            # Create directory to store splits if it doesn't exist
            splits_dir = Path("dataset_splits") / job_id
            splits_dir.mkdir(exist_ok=True, parents=True)
            
            # Save splits to file
            splits_path = splits_dir / "dataset_splits.json"
            with open(splits_path, 'w') as f:
                json.dump(splits, f)
            
            print(f"Dataset splits saved to {splits_path}")
            
        return splits
    
    @staticmethod
    def load_splits(job_id: str) -> Dict[str, List[int]]:
        """Load train/val/test splits from a file
        
        Args:
            job_id: Job ID to load splits for
            
        Returns:
            Dictionary with split indices
        """
        splits_path = Path("dataset_splits") / job_id / "dataset_splits.json"
        if not splits_path.exists():
            # For backward compatibility, also check old location
            old_splits_path = Path("models") / job_id / "splits" / "dataset_splits.json"
            if old_splits_path.exists():
                print(f"Found splits in old location: {old_splits_path}")
                splits_path = old_splits_path
            else:
                raise FileNotFoundError(f"No dataset splits found for job ID {job_id}")
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        return splits

def create_dataloaders(dataset_path: str, transform, batch_size: int = 32, 
                      val_split: float = 0.2, test_split: float = 0.1,
                      num_workers: int = 4, job_id: str = None, 
                      use_saved_splits: bool = False,
                      shuffle: bool = True):
    """Create training, validation and test dataloaders
    
    Args:
        dataset_path: Path to dataset
        transform: Transforms to apply to images
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        num_workers: Number of workers for dataloaders
        job_id: Optional job ID to use for saving/loading splits
        use_saved_splits: Whether to use saved splits if available
        shuffle: Whether to shuffle the training dataloader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, classes)
    """
    # Create full dataset with transform
    full_dataset = ImageClassificationDataset(dataset_path, transform)
    
    # Try to load saved splits if requested
    splits = None
    if use_saved_splits and job_id:
        try:
            splits = ImageClassificationDataset.load_splits(job_id)
            if str(Path(dataset_path).absolute()) != str(Path(splits['dataset_path']).absolute()):
                print(f"Warning: Saved splits are for a different dataset path. Creating new splits.")
                splits = None
        except FileNotFoundError:
            print(f"No saved splits found for job ID {job_id}. Creating new splits.")
    
    # Create and save splits if not loaded
    if splits is None:
        splits = full_dataset.create_and_save_splits(
            val_split=val_split,
            test_split=test_split,
            job_id=job_id
        )
    
    # Create dataset subsets
    train_dataset = Subset(full_dataset, splits['train'])
    val_dataset = Subset(full_dataset, splits['val']) if splits['val'] else None
    test_dataset = Subset(full_dataset, splits['test']) if splits['test'] else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if val_dataset and len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if test_dataset and len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader, full_dataset.classes

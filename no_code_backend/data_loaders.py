import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

def create_dataloaders(dataset_path: str, transform, batch_size: int = 32, 
                      val_split: float = 0.2, num_workers: int = 4, task_type: str = "image_classification", 
                      segmentation_type: str = "semantic"):
    """Create training and validation dataloaders based on task type"""
    
    # Import the specific dataloader based on task type
    if task_type == "image_classification":
        from datasets_module.classification.dataloaders import create_dataloaders as create_classification_dataloaders
        return create_classification_dataloaders(dataset_path, transform, batch_size, val_split, num_workers)
    
    elif task_type == "object_detection":
        from datasets_module.detection.dataloaders import create_dataloaders as create_detection_dataloaders
        return create_detection_dataloaders(dataset_path, transform, batch_size, val_split, num_workers)
    
    elif task_type == "image_segmentation":
        from datasets_module.segmentation.dataloaders import create_dataloaders as create_segmentation_dataloaders
        return create_segmentation_dataloaders(dataset_path, transform, batch_size, val_split, num_workers, segmentation_type)
    
    else:
        # Fall back to the original implementation for backward compatibility
        # Create full dataset with transform
        full_dataset = ImageClassificationDataset(dataset_path, transform)
        
        # Split dataset into train and validation
        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        ) if val_size > 0 else None
        
        return train_loader, val_loader, full_dataset.classes

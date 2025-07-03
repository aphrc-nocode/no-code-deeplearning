"""
COCO Dataset implementation following the article's approach exactly.
Custom PyTorch Dataset to load COCO-format annotations and images
"""
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor


class CocoDetectionDataset(Dataset):
    """COCO Dataset for object detection - following the article implementation"""
    
    def __init__(self, image_dir, annotation_path, transforms=None):
        """
        Init function: loads annotation file and prepares list of image IDs
        
        Args:
            image_dir: Path to directory containing images
            annotation_path: Path to COCO annotation JSON file
            transforms: Optional transforms to apply to images
        """
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        
        # Create category ID mapping for sequential class indices
        # Handle duplicate class names by creating a unique mapping
        category_ids = sorted(self.coco.getCatIds())
        categories = self.coco.loadCats(category_ids)
        
        # Get unique class names and create mapping
        unique_class_names = []
        seen_names = set()
        category_id_to_unique_name = {}
        
        for cat in categories:
            class_name = cat['name']
            if class_name not in seen_names:
                unique_class_names.append(class_name)
                seen_names.add(class_name)
            
            # Map all category IDs with the same name to the same unique index
            unique_index = unique_class_names.index(class_name)
            category_id_to_unique_name[cat['id']] = unique_index
        
        # Create mapping from COCO category ID to PyTorch label (1-indexed)
        self.category_id_to_label = {cat_id: idx + 1 for cat_id, idx in category_id_to_unique_name.items()}
        self.unique_class_names = unique_class_names
        
        # Debug info (can be commented out for production)
        # print(f"COCO category IDs: {category_ids}")
        # print(f"Unique class names: {unique_class_names}")
        # print(f"Category ID to label mapping: {self.category_id_to_label}")
        print(f"Loaded COCO dataset with {len(unique_class_names)} unique classes: {unique_class_names}")

    def __len__(self):
        """Returns total number of images"""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Fetches a single image and its annotations
        
        Args:
            idx: Index of the image to fetch
            
        Returns:
            Tuple of (image, target) where target contains annotations
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Map category ID to sequential label
            category_id = obj['category_id']
            label = self.category_id_to_label[category_id]
            labels.append(label)

        # Convert annotations to PyTorch tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
            iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)
        else:
            # Handle case with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)

        # Package everything into a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64),  # Convert to tensor
            "area": area,
            "iscrowd": iscrowd
        }

        # Apply transforms if any were passed
        if self.transforms:
            image = self.transforms(image)

        return image, target


def get_transforms(train=True):
    """
    Get transforms for training and validation following the article's approach
    
    Args:
        train: Whether to return training transforms (with augmentation) or validation transforms
        
    Returns:
        Transform pipeline
    """
    if train:
        # Training transforms with basic augmentation
        return ToTensor()
    else:
        # Validation transforms (no augmentation)
        return ToTensor()


def get_transform():
    """Legacy function for compatibility - returns a simple transform"""
    return ToTensor()


def collate_fn(batch):
    """Custom collate function for object detection batches"""
    return tuple(zip(*batch))


def create_dataloaders(train_image_dir, train_annotation_path, 
                      val_image_dir, val_annotation_path,
                      batch_size=2, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        train_image_dir: Path to training images
        train_annotation_path: Path to training annotations
        val_image_dir: Path to validation images  
        val_annotation_path: Path to validation annotations
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    from torch.utils.data import DataLoader
    
    # Load training dataset with transform applied
    train_dataset = CocoDetectionDataset(
        image_dir=train_image_dir,
        annotation_path=train_annotation_path,
        transforms=get_transforms(train=True)
    )

    # Load validation dataset with validation transforms
    val_dataset = CocoDetectionDataset(
        image_dir=val_image_dir,
        annotation_path=val_annotation_path,
        transforms=get_transforms(train=False)
    )

    # Get number of classes (including background)
    # Use the number of unique class names + 1 for background
    num_unique_classes = len(train_dataset.unique_class_names)
    num_classes = num_unique_classes + 1  # +1 for background
    
    # Summary of loaded datasets
    print(f"Dataset loaded with {num_unique_classes} unique classes: {train_dataset.unique_class_names}")
    print(f"Total classes for model (including background): {num_classes}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, num_classes

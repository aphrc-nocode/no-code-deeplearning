"""
Transforms for image segmentation tasks.
This module provides transforms for both semantic and instance segmentation.
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image

class Compose:
    """Composes transforms for segmentation tasks"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    """Convert PIL Image and mask to tensors"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            if isinstance(target, Image.Image):
                target = torch.as_tensor(np.array(target), dtype=torch.long)
            elif isinstance(target, dict):  # For instance segmentation
                if "masks" in target and len(target["masks"]) > 0:
                    target["masks"] = [torch.as_tensor(np.array(mask), dtype=torch.uint8) 
                                      for mask in target["masks"]]
                if "boxes" in target and len(target["boxes"]) > 0:
                    target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
                if "labels" in target:
                    target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
        return image, target

class Normalize:
    """Normalize tensor image with mean and standard deviation"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Resize:
    """Resize the image and mask/masks"""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def __call__(self, image, target):
        image = F.resize(image, self.size)
        
        if target is not None:
            if isinstance(target, torch.Tensor):  # Semantic segmentation
                target = F.resize(target.unsqueeze(0), self.size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            elif isinstance(target, Image.Image):
                target = target.resize(self.size, Image.NEAREST)
            elif isinstance(target, dict):  # Instance segmentation
                if "masks" in target and len(target["masks"]) > 0:
                    target["masks"] = [F.resize(mask.unsqueeze(0), self.size, 
                                              interpolation=F.InterpolationMode.NEAREST).squeeze(0)
                                      for mask in target["masks"]]
                
                # Adjust bounding boxes
                if "boxes" in target and len(target["boxes"]) > 0:
                    # Get original dimensions - handle both PIL Image and Tensor
                    if isinstance(image, Image.Image):
                        w, h = image.size
                    else:  # Tensor
                        h, w = image.shape[-2:]
                    
                    boxes = target["boxes"]
                    scale_x = self.size[1] / w
                    scale_y = self.size[0] / h
                    
                    boxes[:, 0] *= scale_x  # x1
                    boxes[:, 1] *= scale_y  # y1
                    boxes[:, 2] *= scale_x  # x2
                    boxes[:, 3] *= scale_y  # y2
                    target["boxes"] = boxes
                    
        return image, target

class RandomHorizontalFlip:
    """Randomly flip the image and mask horizontally"""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            
            if target is not None:
                if isinstance(target, torch.Tensor):  # Semantic segmentation
                    target = F.hflip(target)
                elif isinstance(target, Image.Image):
                    target = target.transpose(Image.FLIP_LEFT_RIGHT)
                elif isinstance(target, dict):  # Instance segmentation
                    if "masks" in target and len(target["masks"]) > 0:
                        target["masks"] = [F.hflip(mask) for mask in target["masks"]]
                    
                    # Flip bounding boxes
                    if "boxes" in target and len(target["boxes"]) > 0:
                        # Get width based on image type
                        if isinstance(image, torch.Tensor):
                            width = image.size(-1)
                        else:  # PIL Image
                            width = image.width
                        
                        boxes = target["boxes"]
                        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                        target["boxes"] = boxes
                        
        return image, target

class RandomVerticalFlip:
    """Randomly flip the image and mask vertically"""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            
            if target is not None:
                if isinstance(target, torch.Tensor):  # Semantic segmentation
                    target = F.vflip(target)
                elif isinstance(target, Image.Image):
                    target = target.transpose(Image.FLIP_TOP_BOTTOM)
                elif isinstance(target, dict):  # Instance segmentation
                    if "masks" in target and len(target["masks"]) > 0:
                        target["masks"] = [F.vflip(mask) for mask in target["masks"]]
                    
                    # Flip bounding boxes
                    if "boxes" in target and len(target["boxes"]) > 0:
                        # Get height based on image type
                        if isinstance(image, torch.Tensor):
                            height = image.size(-2)
                        else:  # PIL Image
                            height = image.height
                            
                        boxes = target["boxes"]
                        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                        target["boxes"] = boxes
                        
        return image, target

class RandomCrop:
    """Randomly crop the image and mask"""
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def __call__(self, image, target):
        # Get crop parameters
        i, j, h, w = T.RandomCrop.get_params(image, self.size)
        
        # Apply crop to image
        image = F.crop(image, i, j, h, w)
        
        if target is not None:
            if isinstance(target, torch.Tensor):  # Semantic segmentation
                target = F.crop(target, i, j, h, w)
            elif isinstance(target, Image.Image):
                target = F.crop(target, i, j, h, w)
            elif isinstance(target, dict):  # Instance segmentation
                # Adjust masks
                if "masks" in target and len(target["masks"]) > 0:
                    target["masks"] = [F.crop(mask, i, j, h, w) for mask in target["masks"]]
                
                # Adjust bounding boxes
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"]
                    boxes[:, 0] = boxes[:, 0] - j  # x1
                    boxes[:, 1] = boxes[:, 1] - i  # y1
                    boxes[:, 2] = boxes[:, 2] - j  # x2
                    boxes[:, 3] = boxes[:, 3] - i  # y2
                    
                    # Clip boxes to new image boundaries
                    boxes[:, 0].clamp_(min=0, max=w)
                    boxes[:, 1].clamp_(min=0, max=h)
                    boxes[:, 2].clamp_(min=0, max=w)
                    boxes[:, 3].clamp_(min=0, max=h)
                    
                    # Remove boxes with no area
                    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
                    if not keep.all():
                        for k in target.keys():
                            if k == "boxes":
                                target[k] = boxes[keep]
                            elif k == "masks":
                                target[k] = [target[k][i] for i in range(len(target[k])) if keep[i]]
                            elif k == "labels":
                                target[k] = target[k][keep]
                    else:
                        target["boxes"] = boxes
                        
        return image, target

class ColorJitter:
    """Apply color jitter to the image but not the mask"""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.jitter(image)
        return image, target

class RandomRotation:
    """Randomly rotate the image and mask"""
    def __init__(self, degrees, prob=0.5):
        self.degrees = degrees
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Apply rotation to image
            image = F.rotate(image, angle, interpolation=F.InterpolationMode.BILINEAR)
            
            if target is not None:
                if isinstance(target, torch.Tensor):  # Semantic segmentation
                    target = F.rotate(target, angle, interpolation=F.InterpolationMode.NEAREST)
                elif isinstance(target, Image.Image):
                    target = target.rotate(angle, resample=Image.NEAREST)
                elif isinstance(target, dict):  # Instance segmentation
                    # For instance segmentation, rotation of boxes is complex
                    # Here we only rotate the masks and recalculate boxes
                    if "masks" in target and len(target["masks"]) > 0:
                        rotated_masks = [F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST)
                                     for mask in target["masks"]]
                        target["masks"] = rotated_masks
                        
                        # Recalculate bounding boxes from rotated masks
                        if "boxes" in target and len(target["boxes"]) > 0:
                            new_boxes = []
                            for mask in rotated_masks:
                                pos = torch.where(mask)
                                if pos[0].numel() > 0:
                                    xmin = pos[1].min().item()
                                    xmax = pos[1].max().item()
                                    ymin = pos[0].min().item()
                                    ymax = pos[0].max().item()
                                    new_boxes.append([xmin, ymin, xmax, ymax])
                                else:
                                    new_boxes.append([0, 0, 0, 0])
                            target["boxes"] = torch.tensor(new_boxes, dtype=torch.float32)
                        
        return image, target

def get_segmentation_transforms(train: bool = True, 
                               segmentation_type: str = "semantic", 
                               image_size: Union[int, Tuple[int, int]] = 224) -> Compose:
    """
    Get transforms for image segmentation
    
    Args:
        train: Whether to use training transforms with data augmentation
        segmentation_type: Type of segmentation ('semantic' or 'instance')
        image_size: Size to resize images to (int or tuple)
        
    Returns:
        Compose transform for image segmentation
    """
    # Ensure size is a tuple
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size
    
    if train:
        # Calculate crop size (80% of the smallest dimension)
        crop_size = int(0.8 * min(size))
        
        # Training transforms with augmentation
        transforms = [
            Resize(size),
            RandomHorizontalFlip(prob=0.5),
            RandomCrop(crop_size),  # Crop to 80% then resize back
            Resize(size),  # Resize back to original size after crop
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ToTensor(),
            Normalize()
        ]
    else:
        # Evaluation/inference transforms
        transforms = [
            Resize(size),
            ToTensor(),
            Normalize()
        ]
    
    return Compose(transforms)

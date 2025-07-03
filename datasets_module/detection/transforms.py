"""
Transforms for object detection tasks.
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from typing import Dict, List, Tuple, Optional, Any

class Compose:
    """Composes transforms for object detection"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class InferenceCompose:
    """Composes transforms for inference only (no target required)"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ToTensor:
    """Convert PIL Image to tensor and normalize"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class InferenceToTensor:
    """Convert PIL Image to tensor for inference"""
    def __call__(self, image):
        return F.to_tensor(image)

class Normalize:
    """Normalize tensor image with mean and standard deviation"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class InferenceNormalize:
    """Normalize tensor image for inference"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, image):
        return F.normalize(image, mean=self.mean, std=self.std)

class Resize:
    """Resize the image and bounding boxes"""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image, target):
        original_size = image.size
        image = F.resize(image, self.size)
        
        # Adjust bounding boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            scale_x = self.size[1] / original_size[0]
            scale_y = self.size[0] / original_size[1]
            
            boxes = target["boxes"]
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            target["boxes"] = boxes
            
        return image, target

class InferenceResize:
    """Resize image for inference"""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, image):
        return F.resize(image, self.size)

class RandomHorizontalFlip:
    """Randomly flip the image and bounding boxes horizontally"""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            
            if "boxes" in target and len(target["boxes"]) > 0:
                width = image.size[0]
                boxes = target["boxes"]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
                
        return image, target

class RandomRotation:
    """Randomly rotate the image and bounding boxes"""
    def __init__(self, degrees, prob=0.5):
        self.degrees = degrees
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            angle = random.uniform(-self.degrees, self.degrees)
            image = F.rotate(image, angle)
            
            # Bounding box rotation is complex and can cause issues
            # For simplicity, we'll skip transforming the boxes for rotation
            # In a production system, proper box rotation should be implemented
                
        return image, target

class ColorJitter:
    """Apply color jitter to the image"""
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

def get_detection_transforms(train=True, image_size=(800, 800)):
    """
    Get transforms for object detection
    
    Args:
        train: Whether to include data augmentation transforms
        image_size: Image size to resize to
        
    Returns:
        Composed transforms
    """
    transforms = []
    
    # Resize
    transforms.append(Resize(image_size))
    
    # Data augmentation for training
    if train:
        transforms.append(RandomHorizontalFlip())
        transforms.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    # Convert to tensor and normalize
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return Compose(transforms)

def get_inference_transforms(image_size=(800, 800)):
    """
    Get transforms for inference (prediction) only
    
    Args:
        image_size: Image size to resize to
        
    Returns:
        Composed transforms for inference
    """
    transforms = []
    
    # Resize
    transforms.append(InferenceResize(image_size))
    
    # Convert to tensor and normalize
    transforms.append(InferenceToTensor())
    transforms.append(InferenceNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    return InferenceCompose(transforms)

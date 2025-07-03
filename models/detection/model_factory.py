"""
Factory for creating object detection models following the article's approach.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Optional


def create_faster_rcnn_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a Faster R-CNN model exactly as shown in the article
    
    Args:
        num_classes: Number of output classes (including background)
        pretrained: Whether to use pre-trained weights on COCO dataset
        
    Returns:
        Faster R-CNN model
    """
    # Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the classifier head with a new one for the custom dataset's classes
    # Number of classes must be equal to your label number
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    print(f"Created Faster R-CNN with {num_classes} classes")
    print(f"Input features for classifier: {in_features}")

    return model


def create_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create an object detection model based on architecture name
    
    Args:
        architecture: Name of the architecture (e.g., 'faster_rcnn')
        num_classes: Number of output classes (including background)
        pretrained: Whether to use pre-trained weights on COCO dataset
        
    Returns:
        Object detection model
    """
    if architecture == "faster_rcnn":
        return create_faster_rcnn_model(num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Only 'faster_rcnn' is supported in this implementation.")
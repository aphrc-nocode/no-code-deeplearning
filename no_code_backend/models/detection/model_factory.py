"""
Factory for creating object detection models.
This module provides functionality to create various pre-trained models for object detection.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Optional

def create_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create an object detection model based on architecture name
    
    Args:
        architecture: Name of the architecture (e.g., 'faster_rcnn', 'mask_rcnn', 'yolo')
        num_classes: Number of output classes (including background)
        pretrained: Whether to use pre-trained weights on COCO dataset
        
    Returns:
        Object detection model
    """
    weights = 'DEFAULT' if pretrained else None
    
    if architecture == "faster_rcnn":
        # Load Faster R-CNN pre-trained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif architecture == "mask_rcnn":
        # Load Mask R-CNN pre-trained model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        
        # Get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Get the number of input features for the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
        
    elif architecture == "ssd":
        # Load Single Shot Detector pre-trained model
        model = torchvision.models.detection.ssd300_vgg16(weights=weights)
        
        # Replace the classification head for the new number of classes
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
    elif architecture == "retinanet":
        # Load RetinaNet pre-trained model
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)
        
        # Replace the classification head for the new number of classes
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes
        cls_logits = torch.nn.Conv2d(model.head.classification_head.conv[0].out_channels,
                                    num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        model.head.classification_head.cls_logits = cls_logits
        
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
        
    return model

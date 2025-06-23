"""
Factory for creating image segmentation models.
This module provides functionality to create various pre-trained models for image segmentation.
"""
import torch
import torch.nn as nn
import torchvision
from typing import Optional, Dict, Any

def create_model(architecture: str, num_classes: int, segmentation_type: str = "semantic", pretrained: bool = True) -> nn.Module:
    """
    Create a segmentation model based on architecture name
    
    Args:
        architecture: Name of the architecture (e.g., 'fcn', 'deeplabv3')
        num_classes: Number of output classes
        segmentation_type: Type of segmentation task ('semantic' or 'instance')
        pretrained: Whether to use pre-trained weights (default: True)
        
    Returns:
        Image segmentation model
    """
    weights = 'DEFAULT' if pretrained else None
    
    if segmentation_type == "semantic":
        if architecture == "fcn":
            model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
            model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        elif architecture == "fcn_resnet50":
            model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
            model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        elif architecture == "fcn_resnet101":
            model = torchvision.models.segmentation.fcn_resnet101(weights=weights)
            model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        elif architecture == "deeplabv3" or architecture == "deeplabv3_resnet50":
            model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        elif architecture == "deeplabv3_resnet101":
            model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        elif architecture == "lraspp_mobilenet_v3_large":
            model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
            model.classifier.low_classifier = nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.classifier.high_classifier = nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        elif architecture == "unet":
            # Import a simple UNet implementation or use a library like segmentation_models_pytorch
            try:
                import segmentation_models_pytorch as smp
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3,
                    classes=num_classes,
                )
            except ImportError:
                raise ValueError("UNet requires segmentation_models_pytorch package. Please install it with: pip install segmentation-models-pytorch")
            
        else:
            raise ValueError(f"Unsupported semantic segmentation architecture: {architecture}")
            
    elif segmentation_type == "instance":
        if architecture == "mask_rcnn" or architecture == "mask_rcnn_resnet50_fpn":
            # Load Mask R-CNN with ResNet-50-FPN backbone
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
            
            # Get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            
            # Replace the pre-trained head with a new one
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            
            # Get the number of input features for the mask predictor
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            
            # Replace the mask predictor with a new one
            model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                num_classes
            )
            
        elif architecture == "maskrcnn_resnet50_fpn_v2":
            # Load Mask R-CNN v2 with ResNet-50-FPN backbone
            model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
            
            # Get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            
            # Replace the pre-trained head with a new one
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            
            # Get the number of input features for the mask predictor
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            
            # Replace the mask predictor with a new one
            model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask,
                hidden_layer,
                num_classes
            )
            
        else:
            raise ValueError(f"Unsupported instance segmentation architecture: {architecture}")
    
    else:
        raise ValueError(f"Unsupported segmentation type: {segmentation_type}. Choose 'semantic' or 'instance'.")
        
    return model

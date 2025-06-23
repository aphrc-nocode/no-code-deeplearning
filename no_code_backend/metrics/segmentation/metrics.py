"""
Metrics for image segmentation tasks.
This module provides functions to calculate common metrics for image segmentation.
"""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple

def calculate_confusion_matrix(pred_mask, gt_mask, num_classes):
    """
    Calculate confusion matrix for semantic segmentation
    
    Args:
        pred_mask: Predicted segmentation mask (B, H, W)
        gt_mask: Ground truth segmentation mask (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    hist = torch.bincount(
        num_classes * gt_mask[mask].int() + pred_mask[mask].int(),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def calculate_segmentation_metrics(outputs, targets, num_classes) -> Dict[str, float]:
    """
    Calculate common segmentation metrics like IoU, pixel accuracy
    
    Args:
        outputs: Model outputs (B, C, H, W) or (B, H, W)
        targets: Target masks (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Process model outputs if they're in logits form
    if outputs.dim() == 4:  # (B, C, H, W)
        pred_mask = torch.argmax(outputs, dim=1)  # (B, H, W)
    else:
        pred_mask = outputs
    
    # Flatten predictions and targets
    pred_mask = pred_mask.view(-1)
    gt_mask = targets.view(-1)
    
    # Calculate confusion matrix
    hist = calculate_confusion_matrix(pred_mask, gt_mask, num_classes)
    
    # Calculate accuracy
    acc = torch.diag(hist).sum() / hist.sum()
    
    # Calculate per-class accuracy
    acc_per_class = torch.diag(hist) / hist.sum(1)
    
    # Calculate IoU per class
    union = hist.sum(1) + hist.sum(0) - torch.diag(hist)
    iou = torch.diag(hist) / union
    
    # Calculate mean IoU
    miou = torch.mean(iou[~torch.isnan(iou)])
    
    # Calculate frequency weighted IoU
    freq = hist.sum(1) / hist.sum()
    fwiou = (freq * iou).sum()
    
    # Calculate precision, recall, and F1 per class
    precision = torch.diag(hist) / hist.sum(0)
    recall = torch.diag(hist) / hist.sum(1)
    f1 = 2 * precision * recall / (precision + recall)
    
    # Compile metrics dictionary
    metrics = {
        'accuracy': acc.item(),
        'miou': miou.item(),
        'fwiou': fwiou.item()
    }
    
    # Add per-class metrics
    for i in range(num_classes):
        if not torch.isnan(iou[i]):
            metrics[f'iou_class_{i}'] = iou[i].item()
        if not torch.isnan(acc_per_class[i]):
            metrics[f'acc_class_{i}'] = acc_per_class[i].item()
        if not torch.isnan(precision[i]):
            metrics[f'precision_class_{i}'] = precision[i].item()
        if not torch.isnan(recall[i]):
            metrics[f'recall_class_{i}'] = recall[i].item()
        if not torch.isnan(f1[i]):
            metrics[f'f1_class_{i}'] = f1[i].item()
    
    return metrics

def calculate_panoptic_quality(instance_preds, instance_targets, num_classes) -> Dict[str, float]:
    """
    Calculate Panoptic Quality metric for instance segmentation
    
    Args:
        instance_preds: List of prediction dictionaries with 'masks', 'labels', 'scores'
        instance_targets: List of target dictionaries with 'masks', 'labels'
        num_classes: Number of classes
        
    Returns:
        Dictionary with PQ, SQ, RQ metrics
    """
    # Implementation for Panoptic Quality
    # This is a simplified placeholder - full implementation requires matching instances
    # between predictions and ground truth
    metrics = {
        'PQ': 0.0,  # Panoptic Quality
        'SQ': 0.0,  # Segmentation Quality
        'RQ': 0.0   # Recognition Quality
    }
    
    # For a proper implementation, we would:
    # 1. Match predicted instances to ground truth instances
    # 2. Calculate IoU for each matched pair
    # 3. Count true positives (IoU > threshold), false positives, false negatives
    # 4. Calculate PQ = TP / (TP + 0.5*FP + 0.5*FN)
    # 5. Calculate SQ = average IoU of matched pairs
    # 6. Calculate RQ = TP / (TP + 0.5*FP + 0.5*FN) (same as PQ but without IoU weighting)
    
    return metrics

def calculate_mask_ap(instance_preds, instance_targets, iou_thresholds=[0.5, 0.75]) -> Dict[str, float]:
    """
    Calculate Average Precision for instance segmentation masks
    
    Args:
        instance_preds: List of prediction dictionaries with 'masks', 'labels', 'scores'
        instance_targets: List of target dictionaries with 'masks', 'labels'
        iou_thresholds: List of IoU thresholds to evaluate AP at
        
    Returns:
        Dictionary with AP metrics
    """
    # Implementation for instance segmentation AP
    # This is a simplified placeholder - full implementation is complex
    metrics = {
        'mask_mAP': 0.0,  # mean Average Precision for masks
    }
    
    # For each IoU threshold
    for iou_threshold in iou_thresholds:
        metrics[f'mask_AP{int(iou_threshold*100)}'] = 0.0
    
    return metrics

def calculate_instance_segmentation_metrics(preds, targets) -> Dict[str, float]:
    """
    Calculate metrics for instance segmentation
    
    Args:
        preds: List of prediction dictionaries with 'masks', 'labels', 'scores'
        targets: List of target dictionaries with 'masks', 'labels'
        
    Returns:
        Dictionary with instance segmentation metrics
    """
    # Combine results from different metrics
    metrics = {}
    
    # Get number of classes
    all_labels = set()
    for target in targets:
        all_labels.update(target['labels'].cpu().numpy())
    num_classes = max(all_labels) + 1 if all_labels else 0
    
    # Calculate mask AP
    mask_ap_metrics = calculate_mask_ap(preds, targets)
    metrics.update(mask_ap_metrics)
    
    # Calculate panoptic quality
    pq_metrics = calculate_panoptic_quality(preds, targets, num_classes)
    metrics.update(pq_metrics)
    
    return metrics

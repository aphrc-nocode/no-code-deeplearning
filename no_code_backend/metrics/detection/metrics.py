"""
Metrics for object detection tasks.
This module provides functions to calculate common metrics for object detection.
"""
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from torchvision.ops import box_iou

def calculate_detection_metrics(detections, targets, iou_threshold=0.5) -> Dict[str, float]:
    """
    Calculate object detection metrics such as mAP, precision, recall
    
    Args:
        detections: List of detection dictionaries with 'boxes', 'labels', 'scores'
        targets: List of target dictionaries with 'boxes', 'labels'
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Dictionary of metric name to value
    """
    metrics = {}
    
    # Get all unique classes across all targets
    all_classes = set()
    for target in targets:
        # Handle empty targets case
        if 'labels' in target and len(target['labels']) > 0:
            all_classes.update(target['labels'].cpu().numpy())
    all_classes = sorted(list(all_classes))
    
    # Calculate AP for each class
    APs = {}
    for class_id in all_classes:
        try:
            APs[class_id] = calculate_average_precision(
                detections, targets, class_id, iou_threshold
            )
        except Exception as e:
            print(f"Error calculating AP for class {class_id}: {e}")
            APs[class_id] = 0.0
    
    # Calculate mean AP
    if APs:
        metrics['mAP'] = sum(APs.values()) / len(APs)
    else:
        metrics['mAP'] = 0.0
    
    # Add per-class AP
    for class_id, ap in APs.items():
        metrics[f'AP_class_{class_id}'] = ap
    
    # Calculate AP at different IoU thresholds
    iou_thresholds = [0.5, 0.75]  # AP50, AP75
    for iou_t in iou_thresholds:
        AP_iou = {}
        for class_id in all_classes:
            AP_iou[class_id] = calculate_average_precision(
                detections, targets, class_id, iou_t
            )
        
        threshold_name = f"AP{int(iou_t*100)}"
        if AP_iou:
            metrics[threshold_name] = sum(AP_iou.values()) / len(AP_iou)
        else:
            metrics[threshold_name] = 0.0
    
    return metrics

def calculate_average_precision(detections, targets, class_id, iou_threshold=0.5) -> float:
    """
    Calculate Average Precision for a specific class
    
    Args:
        detections: List of detection dictionaries
        targets: List of target dictionaries
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Average Precision value
    """
    # Extract all detections and targets for this class
    all_detections = []
    all_targets = []
    
    for i, (detection, target) in enumerate(zip(detections, targets)):
        try:
            # Get detections for this class
            if 'labels' not in detection or len(detection['labels']) == 0:
                continue
                
            detection_indices = (detection['labels'] == class_id).nonzero(as_tuple=True)[0]
            if len(detection_indices) == 0:
                continue
                
            detection_boxes = detection['boxes'][detection_indices].cpu()
            detection_scores = detection['scores'][detection_indices].cpu()
        except (KeyError, IndexError) as e:
            print(f"Error processing detection {i}: {e}")
            continue
        
        # Sort by scores
        if len(detection_scores) > 0:
            score_sort = torch.argsort(detection_scores, descending=True)
            detection_boxes = detection_boxes[score_sort]
            detection_scores = detection_scores[score_sort]
        
        # Get targets for this class
        try:
            if 'labels' not in target or len(target['labels']) == 0:
                continue
                
            target_indices = (target['labels'] == class_id).nonzero(as_tuple=True)[0]
            if len(target_indices) == 0:
                continue
                
            target_boxes = target['boxes'][target_indices].cpu()
        except (KeyError, IndexError) as e:
            print(f"Error processing target {i}: {e}")
            continue
        
        # Add image index to track which image each detection/target belongs to
        for box, score in zip(detection_boxes, detection_scores):
            all_detections.append({
                'box': box,
                'score': score,
                'image_id': i
            })
        
        for box in target_boxes:
            all_targets.append({
                'box': box,
                'image_id': i
            })
    
    # Sort all detections by score
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    
    # Initialize precision/recall variables
    TP = torch.zeros(len(all_detections))
    FP = torch.zeros(len(all_detections))
    total_targets = len(all_targets)
    
    if total_targets == 0:
        return 0.0  # No targets for this class
    
    # Mark detected targets to avoid multiple detections
    detected_targets = [False] * len(all_targets)
    
    # Iterate through detections and calculate TP/FP
    for i, detection in enumerate(all_detections):
        img_id = detection['image_id']
        det_box = detection['box']
        
        # Get targets from same image
        img_targets = [t for t in range(len(all_targets)) 
                      if all_targets[t]['image_id'] == img_id and not detected_targets[t]]
        
        if not img_targets:  # No targets in this image
            FP[i] = 1
            continue
        
        # Calculate IoU with all targets in the same image
        target_boxes = torch.stack([all_targets[t]['box'] for t in img_targets])
        ious = box_iou(det_box.unsqueeze(0), target_boxes)[0]
        
        # Get max IoU and corresponding target index
        max_iou, max_idx = torch.max(ious, dim=0)
        max_idx = img_targets[max_idx]
        
        if max_iou >= iou_threshold:
            if not detected_targets[max_idx]:  # If not detected yet
                TP[i] = 1
                detected_targets[max_idx] = True
            else:
                FP[i] = 1
        else:
            FP[i] = 1
    
    # Calculate cumulative TP and FP
    cumul_TP = torch.cumsum(TP, dim=0)
    cumul_FP = torch.cumsum(FP, dim=0)
    
    # Calculate precision and recall
    precision = cumul_TP / (cumul_TP + cumul_FP)
    recall = cumul_TP / total_targets
    
    # Add a start point (0,1) and end point (1,0) for AP calculation
    precision = torch.cat([torch.tensor([1]), precision])
    recall = torch.cat([torch.tensor([0]), recall])
    
    # Ensure precision is decreasing
    for i in range(len(precision)-1, 0, -1):
        precision[i-1] = max(precision[i-1], precision[i])
    
    # Find points where recall increases
    indices = torch.where(recall[1:] != recall[:-1])[0] + 1
    
    # Calculate AP as area under precision-recall curve
    ap = torch.sum((recall[indices] - recall[indices-1]) * precision[indices])
    
    return ap.item()

def calculate_map_by_size(detections, targets, small_threshold=32*32, medium_threshold=96*96):
    """
    Calculate mAP for different object sizes (small, medium, large)
    
    Args:
        detections: List of detection dictionaries
        targets: List of target dictionaries
        small_threshold: Maximum area for small objects (in pixels)
        medium_threshold: Maximum area for medium objects (in pixels)
        
    Returns:
        Dictionary with mAP for small, medium, and large objects
    """
    # Filter targets by size
    small_targets = []
    medium_targets = []
    large_targets = []
    
    for target in targets:
        boxes = target['boxes']
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        small_mask = areas < small_threshold
        medium_mask = (areas >= small_threshold) & (areas < medium_threshold)
        large_mask = areas >= medium_threshold
        
        small_targets.append({
            'boxes': boxes[small_mask],
            'labels': target['labels'][small_mask]
        })
        
        medium_targets.append({
            'boxes': boxes[medium_mask],
            'labels': target['labels'][medium_mask]
        })
        
        large_targets.append({
            'boxes': boxes[large_mask],
            'labels': target['labels'][large_mask]
        })
    
    # Calculate mAP for each size
    small_metrics = calculate_detection_metrics(detections, small_targets)
    medium_metrics = calculate_detection_metrics(detections, medium_targets)
    large_metrics = calculate_detection_metrics(detections, large_targets)
    
    return {
        'mAP_small': small_metrics['mAP'],
        'mAP_medium': medium_metrics['mAP'],
        'mAP_large': large_metrics['mAP']
    }

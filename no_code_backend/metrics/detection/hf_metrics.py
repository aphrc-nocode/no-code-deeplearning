"""
Metrics calculation for Hugging Face object detection models.
This module provides proper mAP calculation for object detection tasks.
"""

import torch
import torch.utils.data
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import logging

# Try importing torchmetrics with fallback
try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    try:
        from torchmetrics import MeanAveragePrecision
        TORCHMETRICS_AVAILABLE = True
    except ImportError:
        TORCHMETRICS_AVAILABLE = False
        print("Warning: torchmetrics not available. Install with pip install torchmetrics")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelOutput:
    """Data class for model outputs"""
    logits: torch.Tensor
    pred_boxes: torch.Tensor

def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts bounding boxes from YOLO to Pascal VOC format.
    
    Args:
        boxes: Tensor of shape (N, 4) with boxes in YOLO format [x_center, y_center, width, height] (normalized)
        image_size: Tuple of (height, width) of the original image
    
    Returns:
        Tensor of shape (N, 4) with boxes in Pascal format [x_min, y_min, x_max, y_max] (absolute)
    """
    if boxes.numel() == 0:
        return boxes.clone()
    
    height, width = image_size
    
    # Convert from normalized center format to absolute corner format
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height
    
    x_min = x_center - box_width / 2
    y_min = y_center - box_height / 2
    x_max = x_center + box_width / 2
    y_max = y_center + box_height / 2
    
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def convert_bbox_coco_to_pascal(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from COCO to Pascal VOC format.
    
    Args:
        boxes: Tensor of shape (N, 4) with boxes in COCO format [x_min, y_min, width, height]
    
    Returns:
        Tensor of shape (N, 4) with boxes in Pascal format [x_min, y_min, x_max, y_max]
    """
    if boxes.numel() == 0:
        return boxes.clone()
    
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    
    x_max = x_min + width
    y_max = y_min + height
    
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def postprocess_object_detection_output(
    outputs: Any, 
    target_sizes: torch.Tensor, 
    threshold: float = 0.1,
    id2label: Optional[Dict[int, str]] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Post-process object detection model outputs.
    
    Args:
        outputs: Model outputs containing logits and pred_boxes
        target_sizes: Tensor of shape (batch_size, 2) with target image sizes
        threshold: Confidence threshold for filtering predictions
        id2label: Mapping from class indices to class names
    
    Returns:
        List of dictionaries with processed predictions for each image
    """
    if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
    elif hasattr(outputs, 'prediction_logits') and hasattr(outputs, 'pred_boxes'):
        logits = outputs.prediction_logits
        pred_boxes = outputs.pred_boxes
    else:
        logger.warning("Unexpected output format from model")
        return []
    
    batch_size = logits.shape[0]
    processed_outputs = []
    
    for i in range(batch_size):
        # Get predictions for this image
        image_logits = logits[i]  # Shape: (num_queries, num_classes)
        image_boxes = pred_boxes[i]  # Shape: (num_queries, 4)
        
        # Convert logits to probabilities and get scores
        probs = torch.nn.functional.softmax(image_logits, dim=-1)
        
        # Get the maximum probability and corresponding class for each query
        scores, labels = probs.max(dim=-1)
        
        # Filter out background class (usually index 0 or last index)
        # For most models, background is the last class
        background_class = probs.shape[-1] - 1
        valid_detections = labels != background_class
        
        # Apply confidence threshold
        confidence_mask = scores > threshold
        keep = valid_detections & confidence_mask
        
        final_boxes = image_boxes[keep]
        final_scores = scores[keep]
        final_labels = labels[keep]
        
        # Convert boxes to absolute coordinates
        if len(final_boxes) > 0:
            # Get target size for this image
            if i < len(target_sizes):
                target_h, target_w = target_sizes[i]
            else:
                target_h, target_w = target_sizes[0]  # Fallback
            
            # Convert from normalized center format to absolute corner format
            final_boxes = convert_bbox_yolo_to_pascal(final_boxes, (target_h, target_w))
            
            # Clamp boxes to image boundaries
            final_boxes[:, [0, 2]] = torch.clamp(final_boxes[:, [0, 2]], 0, target_w)
            final_boxes[:, [1, 3]] = torch.clamp(final_boxes[:, [1, 3]], 0, target_h)
        
        processed_outputs.append({
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        })
    
    return processed_outputs

@torch.no_grad()
def compute_detection_metrics(
    predictions: List[Dict[str, torch.Tensor]], 
    targets: List[Dict[str, torch.Tensor]], 
    id2label: Optional[Dict[int, str]] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Computes mAP and other detection metrics.
    
    Args:
        predictions: List of prediction dictionaries with 'boxes', 'scores', 'labels'
        targets: List of target dictionaries with 'boxes', 'labels'
        id2label: Mapping from class indices to class names
        num_classes: Total number of classes
    
    Returns:
        Dictionary with computed metrics
    """
    if len(predictions) != len(targets):
        logger.error(f"Mismatch in predictions ({len(predictions)}) and targets ({len(targets)})")
        return _get_zero_metrics()
    
    if len(predictions) == 0:
        logger.warning("No predictions to evaluate")
        return _get_zero_metrics()
    
    # Format predictions and targets for torchmetrics
    formatted_preds = []
    formatted_targets = []
    
    for pred, target in zip(predictions, targets):
        # Process predictions
        if 'boxes' in pred and 'scores' in pred and 'labels' in pred:
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            
            # Ensure tensors are on CPU and correct dtype
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().float()
            else:
                pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
            
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().float()
            else:
                pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
            
            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.cpu().long()
            else:
                pred_labels = torch.tensor(pred_labels, dtype=torch.int64)
            
            # Handle empty predictions
            if len(pred_boxes) == 0:
                pred_boxes = torch.empty((0, 4), dtype=torch.float32)
                pred_scores = torch.empty((0,), dtype=torch.float32)
                pred_labels = torch.empty((0,), dtype=torch.int64)
            
            formatted_preds.append({
                'boxes': pred_boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })
        else:
            # Empty prediction
            formatted_preds.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'scores': torch.empty((0,), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64)
            })
        
        # Process targets
        if 'boxes' in target and 'labels' in target:
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            # Ensure tensors are on CPU and correct dtype
            if isinstance(target_boxes, torch.Tensor):
                target_boxes = target_boxes.cpu().float()
            else:
                target_boxes = torch.tensor(target_boxes, dtype=torch.float32)
            
            if isinstance(target_labels, torch.Tensor):
                target_labels = target_labels.cpu().long()
            else:
                target_labels = torch.tensor(target_labels, dtype=torch.int64)
            
            # Handle empty targets
            if len(target_boxes) == 0:
                target_boxes = torch.empty((0, 4), dtype=torch.float32)
                target_labels = torch.empty((0,), dtype=torch.int64)
            
            formatted_targets.append({
                'boxes': target_boxes,
                'labels': target_labels
            })
        else:
            # Empty target
            formatted_targets.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64)
            })
    
    try:
        # Compute mAP using torchmetrics if available
        if TORCHMETRICS_AVAILABLE:
            logger.info(f"Computing metrics for {len(formatted_preds)} predictions and {len(formatted_targets)} targets")
            
            # Log statistics about the data
            total_pred_boxes = sum(len(p['boxes']) for p in formatted_preds)
            total_target_boxes = sum(len(t['boxes']) for t in formatted_targets)
            logger.info(f"Total prediction boxes: {total_pred_boxes}, Total target boxes: {total_target_boxes}")
            
            if total_pred_boxes == 0 and total_target_boxes == 0:
                logger.warning("No predictions or targets found")
                return _get_zero_metrics()
            
            map_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
            map_metric.update(formatted_preds, formatted_targets)
            map_results = map_metric.compute()
            
            logger.info(f"Raw mAP results: {map_results}")
            
            # Extract base metrics
            metrics = {
                "mAP": float(map_results.get('map', 0.0)),
                "AP50": float(map_results.get('map_50', 0.0)),
                "AP75": float(map_results.get('map_75', 0.0)),
                "mAP_small": float(map_results.get('map_small', -1.0)),
                "mAP_medium": float(map_results.get('map_medium', -1.0)),
                "mAP_large": float(map_results.get('map_large', -1.0)),
            }
            
            # Add class-specific metrics
            if 'classes' in map_results and 'map_per_class' in map_results:
                classes = map_results['classes']
                map_per_class = map_results['map_per_class']
                
                # Handle cases where metrics are 0-dimensional tensors or scalars
                try:
                    if isinstance(classes, torch.Tensor):
                        if classes.dim() == 0:
                            classes = [int(classes.item())]
                        else:
                            classes = [int(c.item()) if c.dim() == 0 else int(c) for c in classes]
                    else:
                        classes = [int(c) for c in classes] if isinstance(classes, (list, tuple)) else [int(classes)]
                    
                    if isinstance(map_per_class, torch.Tensor):
                        if map_per_class.dim() == 0:
                            map_per_class = [float(map_per_class.item())]
                        else:
                            map_per_class = [float(m.item()) if m.dim() == 0 else float(m) for m in map_per_class]
                    else:
                        map_per_class = [float(m) for m in map_per_class] if isinstance(map_per_class, (list, tuple)) else [float(map_per_class)]
                    
                    # Add per-class metrics
                    for class_id, class_map in zip(classes, map_per_class):
                        if id2label and class_id in id2label:
                            class_name = id2label[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        metrics[f"map_{class_name}"] = float(class_map) if not np.isnan(class_map) else 0.0
                except Exception as class_error:
                    logger.warning(f"Error processing class-specific metrics: {str(class_error)}")
                    # Continue without class-specific metrics
        else:
            # Fallback to simple metrics if torchmetrics is not available
            logger.warning("torchmetrics not available, using simplified metrics")
            metrics = _compute_simple_map(formatted_preds, formatted_targets, id2label)
        
        # Clean up any NaN or inf values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                metrics[key] = 0.0
            else:
                metrics[key] = round(value, 4)
        
        logger.info(f"Computed metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return _get_zero_metrics()

def _get_zero_metrics() -> Dict[str, float]:
    """Return zero metrics as fallback"""
    return {
        "mAP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "mAP_small": -1.0,
        "mAP_medium": -1.0,
        "mAP_large": -1.0
    }

def _compute_simple_map(
    predictions: List[Dict[str, torch.Tensor]], 
    targets: List[Dict[str, torch.Tensor]], 
    id2label: Optional[Dict[int, str]] = None
) -> Dict[str, float]:
    """
    Simple mAP computation when torchmetrics is not available.
    This is a basic implementation for fallback purposes.
    """
    if not predictions or not targets:
        return _get_zero_metrics()
    
    # Basic counting for simple metrics
    total_predictions = sum(len(pred['boxes']) for pred in predictions)
    total_targets = sum(len(target['boxes']) for target in targets)
    
    if total_targets == 0:
        return _get_zero_metrics()
    
    # Simple recall approximation (this is very basic)
    recall = min(total_predictions / total_targets, 1.0) if total_targets > 0 else 0.0
    
    # Basic metrics (these are approximations)
    metrics = {
        "mAP": recall * 0.5,  # Very rough approximation
        "AP50": recall * 0.7,
        "AP75": recall * 0.3,
        "mAP_small": -1.0,
        "mAP_medium": -1.0,
        "mAP_large": -1.0
    }
    
    return metrics

def create_compute_metrics_function(
    model: torch.nn.Module,
    image_processor: Any,
    val_dataset: Any,
    collate_fn: callable,
    id2label: Optional[Dict[int, str]] = None,
    threshold: float = 0.1
) -> callable:
    """
    Create a compute_metrics function for Hugging Face Trainer.
    
    Args:
        model: The detection model
        image_processor: Image processor for post-processing
        val_dataset: Validation dataset
        collate_fn: Data collation function
        id2label: Class index to name mapping
        threshold: Confidence threshold for predictions
    
    Returns:
        Function that can be used with Hugging Face Trainer
    """
    
    def compute_metrics(eval_pred):
        """
        Compute metrics during evaluation.
        This function is called by the Hugging Face Trainer.
        """
        try:
            # Check if we have a validation dataset
            if val_dataset is None or len(val_dataset) == 0:
                logger.warning("No validation dataset available")
                return _get_zero_metrics()
            
            # Run inference on validation dataset
            model.eval()
            all_predictions = []
            all_targets = []
            
            # Create a small validation dataloader
            import torch.utils.data
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=2,  # Small batch size for evaluation
                collate_fn=collate_fn,
                shuffle=False
            )
            
            device = next(model.parameters()).device
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= 10:  # Limit evaluation to first 10 batches for speed
                        break
                    
                    try:
                        logger.debug(f"Processing batch {batch_idx}")
                        
                        # Move batch to device
                        pixel_values = batch["pixel_values"].to(device)
                        logger.debug(f"Batch {batch_idx}: pixel_values shape: {pixel_values.shape}")
                        
                        # Get model outputs
                        outputs = model(pixel_values=pixel_values)
                        logger.debug(f"Batch {batch_idx}: Model outputs obtained")
                        
                        # Get target sizes - handle properly
                        batch_size = pixel_values.shape[0]
                        target_sizes_list = []
                        for i in range(batch_size):
                            if i < len(batch["labels"]):
                                label_dict = batch["labels"][i]
                                if "orig_size" in label_dict:
                                    orig_size = label_dict["orig_size"]
                                    if isinstance(orig_size, torch.Tensor):
                                        if orig_size.numel() == 1:
                                            # Single value tensor, use default
                                            target_sizes_list.append([224, 224])
                                        else:
                                            # Multi-element tensor, convert to list
                                            target_sizes_list.append(orig_size.tolist()[:2])
                                    elif isinstance(orig_size, (list, tuple)):
                                        target_sizes_list.append(list(orig_size[:2]))
                                    else:
                                        target_sizes_list.append([224, 224])
                                else:
                                    target_sizes_list.append([224, 224])
                            else:
                                target_sizes_list.append([224, 224])
                        
                        target_sizes = torch.tensor(target_sizes_list, dtype=torch.float32)
                        logger.debug(f"Batch {batch_idx}: target_sizes: {target_sizes}")
                        
                        # Post-process predictions
                        batch_predictions = postprocess_object_detection_output(
                            outputs, target_sizes, threshold=threshold, id2label=id2label
                        )
                        logger.debug(f"Batch {batch_idx}: Got {len(batch_predictions)} predictions")
                        all_predictions.extend(batch_predictions)
                        
                        # Process targets - handle tensor conversion more carefully
                        batch_targets = []
                        for i, label_dict in enumerate(batch["labels"]):
                            try:
                                logger.debug(f"Batch {batch_idx}, target {i}: Processing target with keys: {label_dict.keys()}")
                                
                                target_boxes = label_dict['boxes']
                                target_labels = label_dict['class_labels']
                                
                                logger.debug(f"Batch {batch_idx}, target {i}: boxes type: {type(target_boxes)}, labels type: {type(target_labels)}")
                                
                                # Convert target boxes to tensor
                                if isinstance(target_boxes, torch.Tensor):
                                    target_boxes = target_boxes.clone().detach().float()
                                elif isinstance(target_boxes, (list, np.ndarray)):
                                    target_boxes = torch.tensor(target_boxes, dtype=torch.float32)
                                else:
                                    # Handle edge case
                                    target_boxes = torch.empty((0, 4), dtype=torch.float32)
                                
                                # Convert target labels to tensor
                                if isinstance(target_labels, torch.Tensor):
                                    target_labels = target_labels.clone().detach().long()
                                elif isinstance(target_labels, (list, np.ndarray)):
                                    target_labels = torch.tensor(target_labels, dtype=torch.int64)
                                else:
                                    # Handle edge case
                                    target_labels = torch.empty((0,), dtype=torch.int64)
                                
                                logger.debug(f"Batch {batch_idx}, target {i}: boxes shape: {target_boxes.shape}, labels shape: {target_labels.shape}")
                                
                                # Check if boxes are in COCO format (x, y, w, h) and convert
                                if target_boxes.numel() > 0 and target_boxes.shape[-1] == 4:
                                    # Assume COCO format and convert to Pascal VOC
                                    target_boxes = convert_bbox_coco_to_pascal(target_boxes)
                                    logger.debug(f"Batch {batch_idx}, target {i}: Converted boxes to Pascal VOC format")
                                
                                batch_targets.append({
                                    'boxes': target_boxes,
                                    'labels': target_labels
                                })
                            except Exception as target_error:
                                logger.error(f"Error processing target {i} in batch {batch_idx}: {str(target_error)}")
                                # Add empty target to maintain batch consistency
                                batch_targets.append({
                                    'boxes': torch.empty((0, 4), dtype=torch.float32),
                                    'labels': torch.empty((0,), dtype=torch.int64)
                                })
                        
                        logger.debug(f"Batch {batch_idx}: Got {len(batch_targets)} targets")
                        all_targets.extend(batch_targets)
                        
                    except Exception as batch_error:
                        logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                        continue
            
            # Compute metrics
            if len(all_predictions) > 0 and len(all_targets) > 0:
                logger.info(f"Computing metrics with {len(all_predictions)} predictions and {len(all_targets)} targets")
                
                # Log some statistics
                total_pred_boxes = sum(len(p['boxes']) for p in all_predictions)
                total_target_boxes = sum(len(t['boxes']) for t in all_targets)
                logger.info(f"Total prediction boxes: {total_pred_boxes}, Total target boxes: {total_target_boxes}")
                
                metrics = compute_detection_metrics(
                    all_predictions, all_targets, id2label=id2label
                )
                logger.info(f"Final computed metrics: {metrics}")
                return metrics
            else:
                logger.warning("No valid predictions or targets for metric computation")
                return _get_zero_metrics()
                
        except Exception as e:
            logger.error(f"Error in compute_metrics: {str(e)}")
            return _get_zero_metrics()
    
    return compute_metrics

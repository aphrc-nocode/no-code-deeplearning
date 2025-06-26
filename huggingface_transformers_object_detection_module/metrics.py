import torch
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

def convert_bbox_yolo_to_pascal(boxes, image_size):
    """Converts bounding boxes from YOLO to Pascal VOC format."""
    boxes = center_to_corners_format(boxes)
    height, width = image_size
    boxes *= torch.tensor([[width, height, width, height]])
    return boxes

@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """Computes mAP and mAR for object detection."""
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    
    image_sizes, post_processed_targets, post_processed_predictions = [], [], []

    for batch in targets:
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)
    
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Pop class-specific metrics
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")

    # Handle cases where metrics are 0-dimensional tensors
    if isinstance(classes, torch.Tensor) and classes.dim() == 0:
        classes = [classes]
    if isinstance(map_per_class, torch.Tensor) and map_per_class.dim() == 0:
         map_per_class = [map_per_class]
    if isinstance(mar_100_per_class, torch.Tensor) and mar_100_per_class.dim() == 0:
         mar_100_per_class = [mar_100_per_class]

    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label.get(class_id.item(), class_id.item())
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar
    
    return {k: round(v.item(), 4) for k, v in metrics.items()}

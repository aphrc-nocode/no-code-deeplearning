"""
Visualization utilities for model analysis and evaluation.
These functions create common plots for model evaluation and log them to MLflow.
"""
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
import mlflow
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional


def get_text_size(draw, text, font):
    """
    Get text size using the appropriate method based on Pillow version.
    Compatible with both old (textsize) and new (textbbox) Pillow versions.
    """
    try:
        # New method for Pillow 10.0.0+
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        return draw.textsize(text, font=font)


def draw_bounding_boxes(
    image: Image.Image, 
    detections: List[Dict[str, Any]], 
    confidence_threshold: float = 0.5,
    box_thickness: int = 2,
    font_size: int = 12
) -> Image.Image:
    """
    Draw bounding boxes on an image for object detection results.
    
    Args:
        image: PIL Image to draw on
        detections: List of detection dictionaries with 'box', 'class_name', 'confidence'
        confidence_threshold: Minimum confidence to display detection
        box_thickness: Thickness of bounding box lines
        font_size: Font size for labels
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy of the image
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Get image dimensions
    img_width, img_height = image.size
    print(f"Image size for bounding box drawing: {img_width}x{img_height}")
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    # Define colors for different classes (cycling through a predefined palette)
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
        "#FFA500", "#FFC0CB", "#A52A2A", "#DDA0DD", "#98FB98", "#F0E68C"
    ]
    
    # Keep track of class colors
    class_colors = {}
    color_index = 0
    
    for detection in detections:
        confidence = detection.get("confidence", 0)
        if confidence < confidence_threshold * 100:  # confidence is in percentage
            continue
            
        box = detection["box"]
        class_name = detection.get("class_name", "Unknown")
        
        print(f"Original box coordinates: {box}")
        
        # Handle coordinate scaling and clamping
        x1, y1, x2, y2 = box
        
        # If coordinates are way outside image bounds, they might be scaled for a different size
        # Common model output sizes are 800x800, so check if we need to scale down
        if max(x1, x2) > img_width * 1.5 or max(y1, y2) > img_height * 1.5:
            # Assume coordinates are for 800x800 and scale to actual image size
            scale_x = img_width / 800.0
            scale_y = img_height / 800.0
            
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            
            print(f"Scaled coordinates: [{x1}, {y1}, {x2}, {y2}]")
        
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # Ensure box has valid dimensions
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid box dimensions after clamping: [{x1}, {y1}, {x2}, {y2}] - skipping")
            continue
            
        print(f"Final clamped coordinates: [{x1}, {y1}, {x2}, {y2}]")
        
        # Assign color to class
        if class_name not in class_colors:
            class_colors[class_name] = colors[color_index % len(colors)]
            color_index += 1
        
        color = class_colors[class_name]
        
        # Draw bounding box with multiple lines for thickness
        for i in range(box_thickness):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color, width=1)
        
        # Prepare label text
        label = f"{class_name}: {confidence:.1f}%"
        
        # Get text size
        text_width, text_height = get_text_size(draw, label, font)
        
        # Calculate label background rectangle
        label_bg_coords = [
            x1, 
            y1 - text_height - 4, 
            x1 + text_width + 4, 
            y1
        ]
        
        # Ensure label background is within image bounds
        if label_bg_coords[1] < 0:
            # If label would be above image, place it below the box
            label_bg_coords[1] = y2
            label_bg_coords[3] = y2 + text_height + 4
        
        # Draw label background
        draw.rectangle(label_bg_coords, fill=color)
        
        # Draw label text
        text_x = x1 + 2
        text_y = label_bg_coords[1] + 2
        draw.text((text_x, text_y), label, fill="white", font=font)
        
        print(f"Drew bounding box for {class_name} at [{x1}, {y1}, {x2}, {y2}]")
    
    return img_with_boxes


def draw_segmentation_mask(
    image: Image.Image,
    segmentation_mask: np.ndarray,
    class_mapping: Dict[int, str],
    alpha: float = 0.5
) -> Image.Image:
    """
    Draw segmentation mask overlay on an image.
    
    Args:
        image: PIL Image to draw on
        segmentation_mask: Numpy array with class indices
        class_mapping: Mapping from class indices to class names
        alpha: Transparency of the overlay
    
    Returns:
        PIL Image with segmentation overlay
    """
    # Create a copy of the image
    img_with_mask = image.copy()
    
    # Convert to RGBA for transparency support
    if img_with_mask.mode != 'RGBA':
        img_with_mask = img_with_mask.convert('RGBA')
    
    # Create color mapping for classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 165, 0), (255, 192, 203), (165, 42, 42), (221, 160, 221), (152, 251, 152)
    ]
    
    # Create overlay image
    overlay = Image.new('RGBA', img_with_mask.size, (0, 0, 0, 0))
    overlay_data = np.array(overlay)
    
    # Apply colors to mask
    for class_idx, class_name in class_mapping.items():
        if class_idx == 0:  # Skip background
            continue
        
        mask_coords = np.where(segmentation_mask == class_idx)
        if len(mask_coords[0]) > 0:
            color = colors[class_idx % len(colors)]
            overlay_data[mask_coords[0], mask_coords[1]] = (*color, int(255 * alpha))
    
    # Convert back to PIL and composite
    overlay = Image.fromarray(overlay_data, 'RGBA')
    result = Image.alpha_composite(img_with_mask, overlay)
    
    return result.convert('RGB')


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return img_base64


def base64_to_pil(base64_string: str) -> Image.Image:
    """
    Convert a base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        PIL Image
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


def log_prediction_results(
    job_id: str,
    task_type: str,
    prediction_result: Dict[str, Any],
    processing_time: float = None
):
    """
    Log prediction results for monitoring and debugging.
    
    Args:
        job_id: Job identifier
        task_type: Type of ML task
        prediction_result: The prediction results
        processing_time: Time taken for prediction in seconds
    """
    print(f"\n=== Prediction Results for Job {job_id} ===")
    print(f"Task Type: {task_type}")
    
    if processing_time:
        print(f"Processing Time: {processing_time:.3f}s")
    
    if task_type == "object_detection":
        detections = prediction_result.get("detections", [])
        print(f"Number of detections: {len(detections)}")
        
        for i, detection in enumerate(detections):
            class_name = detection.get("class_name", "Unknown")
            confidence = detection.get("confidence", 0)
            box = detection.get("box", [])
            print(f"  Detection {i+1}: {class_name} ({confidence:.1f}%) at {box}")
            
    elif task_type == "image_classification":
        predictions = prediction_result.get("predictions", [])
        print(f"Number of classes predicted: {len(predictions)}")
        
        for i, pred in enumerate(predictions[:5]):  # Show top 5
            class_name = pred.get("class_name", "Unknown")
            confidence = pred.get("confidence", 0)
            print(f"  Prediction {i+1}: {class_name} ({confidence:.1f}%)")
            
    elif task_type == "image_segmentation":
        segmentation_type = prediction_result.get("segmentation_type", "semantic")
        if segmentation_type == "semantic":
            class_mapping = prediction_result.get("class_mapping", {})
            print(f"Semantic segmentation with {len(class_mapping)} classes")
        else:
            instances = prediction_result.get("instances", [])
            print(f"Instance segmentation with {len(instances)} instances")
    
    print("=" * 50)


def create_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix", normalize=False):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Path to the saved confusion matrix image
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Get all unique labels from both true and predicted labels
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    
    # If class_names are provided, ensure we have labels for all classes
    if class_names:
        # Create labels for all possible classes (0 to len(class_names)-1)
        all_labels = list(range(len(class_names)))
        # Compute confusion matrix with explicit labels parameter
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    else:
        # Use found unique labels
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    
    # Generate heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd', 
        cmap='Blues',
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto"
    )
    
    plt.title(title, fontsize=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
    # Save plot to a temporary file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    save_path = temp_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def create_loss_curve(train_losses, val_losses=None, title="Training and Validation Loss"):
    """
    Create and save a loss curve visualization.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        title: Plot title
        
    Returns:
        Path to the saved loss curve image
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(title, fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save plot to a temporary file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    save_path = temp_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def create_accuracy_curve(train_accuracies, val_accuracies=None, title="Training and Validation Accuracy"):
    """
    Create and save an accuracy curve visualization.
    
    Args:
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch (optional)
        title: Plot title
        
    Returns:
        Path to the saved accuracy curve image
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
    if val_accuracies:
        plt.plot(epochs, val_accuracies, 'y-', label='Validation Accuracy')
    
    plt.title(title, fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save plot to a temporary file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    save_path = temp_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def create_roc_curve(y_true, y_scores, class_names=None, title="ROC Curve"):
    """
    Create and save a ROC curve visualization.
    For multi-class, creates one-vs-rest ROC curves.
    
    Args:
        y_true: Ground truth labels (one-hot encoded for multi-class)
        y_scores: Predicted scores/probabilities
        class_names: List of class names
        title: Plot title
        
    Returns:
        Path to the saved ROC curve image
    """
    plt.figure(figsize=(10, 8))
    
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Check if we have a single class scenario
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        plt.text(0.5, 0.5, f"All samples belong to class {unique_classes[0]}\nROC curve not applicable",
                ha='center', va='center', fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=15)
    # Handle multi-class case
    elif len(unique_classes) > 2:
        n_classes = max(len(unique_classes), y_scores.shape[1] if y_scores.ndim > 1 else 2)
        
        # Convert to one-hot encoding if not already
        if y_true.ndim == 1:
            y_true_onehot = np.eye(n_classes)[y_true]
        else:
            y_true_onehot = y_true
            
        # For each class
        for i in range(n_classes):
            # Skip if no scores for this class
            if y_scores.shape[1] <= i:
                continue
                
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')
    else:
        # Binary case
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1] if y_scores.ndim > 1 else y_scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        except (ValueError, IndexError):
            plt.text(0.5, 0.5, "Insufficient data for ROC curve\nNeed both positive and negative samples",
                    ha='center', va='center', fontsize=14)
            
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    
    # Save plot to a temporary file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    save_path = temp_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def create_precision_recall_curve(y_true, y_scores, class_names=None, title="Precision-Recall Curve"):
    """
    Create and save a precision-recall curve visualization.
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores/probabilities
        class_names: List of class names
        title: Plot title
        
    Returns:
        Path to the saved precision-recall curve image
    """
    plt.figure(figsize=(10, 8))
    
    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Check if we have a single class scenario
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        plt.text(0.5, 0.5, f"All samples belong to class {unique_classes[0]}\nPrecision-Recall curve not applicable",
                ha='center', va='center', fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=15)
    # Handle multi-class case
    elif len(unique_classes) > 2:
        n_classes = max(len(unique_classes), y_scores.shape[1] if y_scores.ndim > 1 else 2)
        
        # For each class
        for i in range(n_classes):
            # Skip if no scores for this class
            if y_scores.shape[1] <= i:
                continue
                
            try:
                precision, recall, _ = precision_recall_curve(y_true == i, y_scores[:, i])
                pr_auc = auc(recall, precision)
                class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                plt.plot(recall, precision, lw=2, label=f'{class_label} (AUC = {pr_auc:.2f})')
            except ValueError as e:
                # Skip if class has too few samples
                print(f"Skipping PR curve for class {i}: {e}")
    else:
        # Binary case
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1] if y_scores.ndim > 1 else y_scores)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        except (ValueError, IndexError):
            plt.text(0.5, 0.5, "Insufficient data for Precision-Recall curve\nNeed both positive and negative samples",
                    ha='center', va='center', fontsize=14)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend(loc="best", fontsize=10)
    
    # Save plot to a temporary file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    save_path = temp_dir / "precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def create_class_distribution_plot(class_counts, class_names=None, title="Class Distribution"):
    """
    Create and save a bar chart showing class distribution.
    
    Args:
        class_counts: Dictionary or list of class counts
        class_names: List of class names (optional if class_counts is a dict)
        title: Plot title
        
    Returns:
        Path to the saved class distribution plot
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(class_counts, dict):
        labels = list(class_counts.keys())
        values = list(class_counts.values())
    else:
        values = class_counts
        labels = class_names if class_names else [f"Class {i}" for i in range(len(class_counts))]
    
    plt.bar(range(len(values)), values, align='center')
    plt.xticks(range(len(values)), labels, rotation=45, ha='right')
    
    plt.title(title, fontsize=15)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, str(v), ha='center')
    
    # Save plot to a temporary file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    save_path = temp_dir / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return str(save_path)

def log_all_visualizations_to_mlflow(
    train_losses, 
    train_accuracies, 
    val_losses=None, 
    val_accuracies=None, 
    y_true=None, 
    y_pred=None, 
    y_scores=None, 
    class_counts=None, 
    class_names=None
):
    """
    Create and log all visualizations to MLflow at once.
    
    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses (optional)
        val_accuracies: List of validation accuracies (optional)
        y_true: Ground truth labels (optional)
        y_pred: Predicted labels (optional)
        y_scores: Predicted scores/probabilities (optional)
        class_counts: Dictionary or list of class counts (optional)
        class_names: List of class names (optional)
    """
    # Create a temporary directory for all visualizations
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Dictionary to track all created plots
    plots = {}
    
    # Create and log loss curve
    if train_losses:
        loss_curve_path = create_loss_curve(train_losses, val_losses)
        mlflow.log_artifact(loss_curve_path, "visualizations")
        plots['loss_curve'] = loss_curve_path
    
    # Create and log accuracy curve
    if train_accuracies:
        accuracy_curve_path = create_accuracy_curve(train_accuracies, val_accuracies)
        mlflow.log_artifact(accuracy_curve_path, "visualizations")
        plots['accuracy_curve'] = accuracy_curve_path
    
    # Create and log confusion matrix
    if y_true is not None and y_pred is not None:
        confusion_matrix_path = create_confusion_matrix(y_true, y_pred, class_names)
        mlflow.log_artifact(confusion_matrix_path, "visualizations")
        plots['confusion_matrix'] = confusion_matrix_path
    
    # Create and log ROC curve
    if y_true is not None and y_scores is not None:
        roc_curve_path = create_roc_curve(y_true, y_scores, class_names)
        mlflow.log_artifact(roc_curve_path, "visualizations")
        plots['roc_curve'] = roc_curve_path
        
        # Also create and log precision-recall curve
        pr_curve_path = create_precision_recall_curve(y_true, y_scores, class_names)
        mlflow.log_artifact(pr_curve_path, "visualizations")
        plots['precision_recall_curve'] = pr_curve_path
    
    # Create and log class distribution
    if class_counts:
        class_dist_path = create_class_distribution_plot(class_counts, class_names)
        mlflow.log_artifact(class_dist_path, "visualizations")
        plots['class_distribution'] = class_dist_path
    
    return plots

def cleanup_temp_files(file_paths):
    """
    Remove temporary visualization files after they've been logged to MLflow.
    
    Args:
        file_paths: List or dictionary of file paths to clean up
    """
    if isinstance(file_paths, dict):
        paths = file_paths.values()
    else:
        paths = file_paths
    
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            
    # Try to remove temp directory if it's empty
    try:
        os.rmdir("temp")
    except:
        pass  # Directory not empty or doesn't exist
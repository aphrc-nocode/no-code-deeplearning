"""
Metrics for image classification tasks.
This module provides functions to calculate common metrics for image classification.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict, Any, List, Tuple

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """
    Calculate common classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Model scores/probabilities for each class
        
    Returns:
        Dictionary of metric name to value
    """
    # Check that we have more than one class for multi-class metrics
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_classes)
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # For multi-class classification
    if n_classes > 2:
        # Use macro averaging for multi-class
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Use weighted averaging for multi-class
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics for each class
        for i, class_idx in enumerate(unique_classes):
            metrics[f'precision_class_{class_idx}'] = precision_score(y_true, y_pred, 
                                                                    labels=[class_idx], 
                                                                    average='micro',
                                                                    zero_division=0)
            metrics[f'recall_class_{class_idx}'] = recall_score(y_true, y_pred, 
                                                             labels=[class_idx], 
                                                             average='micro',
                                                             zero_division=0)
            metrics[f'f1_class_{class_idx}'] = f1_score(y_true, y_pred, 
                                                     labels=[class_idx], 
                                                     average='micro',
                                                     zero_division=0)
    else:
        # Binary classification metrics
        # For binary classification, positive class is 1
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    return metrics

def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None) -> str:
    """
    Generate a text report showing the main classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: List of target class names
        
    Returns:
        Text report
    """
    return classification_report(y_true, y_pred, target_names=target_names)

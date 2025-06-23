"""
MLflow integration utilities for the no-code backend.
This module provides functions to track experiments, log metrics, and manage model artifacts.
"""

import mlflow
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from pydantic import BaseModel
import json
import numpy as np

# Import visualization utilities
from visualization_utils import (
    log_all_visualizations_to_mlflow,
    create_confusion_matrix,
    create_loss_curve,
    create_accuracy_curve,
    cleanup_temp_files
)

# Configure MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./logs/mlflow")
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "no-code-ml-experiments")

def setup_mlflow():
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create mlflow directory if it doesn't exist
    Path("logs/mlflow").mkdir(parents=True, exist_ok=True)
    
    # Set or create the experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    
    mlflow.set_experiment(EXPERIMENT_NAME)

def start_run(job_id: str, config: Dict[str, Any]) -> str:
    """Start an MLflow run with the given configuration"""
    run = mlflow.start_run(run_name=f"job_{job_id}")
    
    # Log parameters from config
    for key, value in config.items():
        # Convert non-primitive types to strings
        if isinstance(value, (int, float, str, bool)):
            mlflow.log_param(key, value)
        else:
            mlflow.log_param(key, str(value))
    
    return run.info.run_id

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """Log metrics to the current MLflow run"""
    for name, value in metrics.items():
        if value is not None and not np.isnan(value) and isinstance(value, (int, float)):
            mlflow.log_metric(name, value, step=step)

def log_batch_metrics(metrics: Dict[str, Any], step: int):
    """Log batch metrics to the current MLflow run"""
    log_metrics(metrics, step)

def log_model(model, model_path: str, class_to_idx: Dict = None, config: Dict = None):
    """Log model and associated metadata to MLflow"""
    # Log the PyTorch model
    mlflow.pytorch.log_model(model, "model")
    
    # Log class mapping if provided
    if class_to_idx:
        class_mapping_path = Path("class_mapping.json")
        with open(class_mapping_path, "w") as f:
            json.dump(class_to_idx, f)
        mlflow.log_artifact(str(class_mapping_path), "metadata")
        
        # Clean up temporary file
        if class_mapping_path.exists():
            class_mapping_path.unlink()
    
    # Log model configuration
    if config:
        config_path = Path("model_config.json")
        with open(config_path, "w") as f:
            json.dump({k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                      for k, v in config.items()}, f)
        mlflow.log_artifact(str(config_path), "metadata")
        
        # Clean up temporary file
        if config_path.exists():
            config_path.unlink()

def log_training_visualizations(
    train_losses: List[float], 
    train_accuracies: List[float], 
    val_losses: List[float] = None, 
    val_accuracies: List[float] = None
):
    """Create and log training curve visualizations to MLflow"""
    # Create loss curve
    loss_curve_path = create_loss_curve(train_losses, val_losses)
    mlflow.log_artifact(loss_curve_path, "visualizations")
    
    # Create accuracy curve
    acc_curve_path = create_accuracy_curve(train_accuracies, val_accuracies)
    mlflow.log_artifact(acc_curve_path, "visualizations")
    
    # Clean up temporary files
    cleanup_temp_files([loss_curve_path, acc_curve_path])

def log_evaluation_visualizations(
    y_true, 
    y_pred, 
    y_scores, 
    class_names=None, 
    class_counts=None
):
    """Create and log model evaluation visualizations to MLflow"""
    # Use the comprehensive visualization logging function
    plots = log_all_visualizations_to_mlflow(
        train_losses=None,  # We pass these separately in log_training_visualizations
        train_accuracies=None,
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        class_names=class_names,
        class_counts=class_counts
    )
    
    # Clean up temporary files
    cleanup_temp_files(plots)

def log_image(image_path: str, image_name: str):
    """Log an image artifact to MLflow"""
    mlflow.log_artifact(image_path, f"images/{image_name}")

def end_run():
    """End the current MLflow run"""
    if mlflow.active_run():
        mlflow.end_run()

def get_run_info(run_id: str) -> Dict[str, Any]:
    """Get information about a specific MLflow run"""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "metrics": run.data.metrics,
        "params": run.data.params
    }

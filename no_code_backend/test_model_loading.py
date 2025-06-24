"""
Script to test model loading from both local path and MLflow.
"""
import os
import sys
import torch
from pathlib import Path
import mlflow
import json

def test_load_from_mlflow(run_id):
    """Test loading model from MLflow"""
    try:
        mlflow.set_tracking_uri("file:./logs/mlflow")
        # Load model from MLflow
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
        print(f"Successfully loaded model from MLflow run: {run_id}")
        return model
    except Exception as e:
        print(f"Failed to load model from MLflow: {str(e)}")
        return None

def test_load_from_local(model_path):
    """Test loading model from local path"""
    try:
        # Load model from local path
        checkpoint = torch.load(model_path, weights_only=False)
        print(f"✅ Successfully loaded checkpoint from: {model_path}")
        
        # Print model contents
        print("Checkpoint contains:", list(checkpoint.keys()))
        
        # If model_state_dict is available, we could create a model and load the weights
        if 'model_state_dict' in checkpoint:
            print("Model state dict is available!")
            
            # Print config if available
            if 'config' in checkpoint:
                print("Model config:", checkpoint['config'])
            
            return checkpoint
        else:
            print("❌ No model_state_dict in checkpoint")
            return None
    except Exception as e:
        print(f"❌ Failed to load local model: {str(e)}")
        return None

def main():
    """Test model loading from both sources"""
    # Check if MLflow directory exists and print runs
    mlflow_dir = Path("logs/mlflow")
    if not mlflow_dir.exists():
        print("MLflow directory doesn't exist")
    else:
        print(f"MLflow directory found at {mlflow_dir}")
        # Find the latest run ID
        mlflow.set_tracking_uri("file:./logs/mlflow")
        client = mlflow.tracking.MlflowClient()
        runs = list(client.search_runs(experiment_ids=["1"]))
        if runs:
            latest_run = runs[0]
            print(f"Latest run ID: {latest_run.info.run_id}")
            # Try to load the model
            test_load_from_mlflow(latest_run.info.run_id)
            
            # Check if we have a local model path in MLflow models directory
            local_path = mlflow_dir / "models" / latest_run.info.run_id / "model.pth"
            if local_path.exists():
                print(f"Local MLflow model found at {local_path}")
                test_load_from_local(local_path)
            else:
                print(f"No local MLflow model found at {local_path}")
        else:
            print("No runs found in MLflow")
    
    # Also check models directory
    models_dir = Path("models")
    if not models_dir.exists():
        print("Models directory doesn't exist")
    else:
        print(f"Models directory found at {models_dir}")
        # Check if we have example_detection.pth
        model_path = models_dir / "example_detection.pth"
        if model_path.exists():
            print(f"Found model at {model_path}")
            test_load_from_local(model_path)
        else:
            print(f"No model found at {model_path}")

if __name__ == "__main__":
    main()

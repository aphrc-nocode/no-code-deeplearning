# No-Code AI Platform Backend with FastAPI
# Core architecture for reusable ML pipelines with image support

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import asyncio
import uuid
import json
import sys
from datetime import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import io
import numpy as np
from pathlib import Path
import logging
from mlflow_server import start_mlflow_server, get_mlflow_ui_url
import os
import mlflow
import zipfile
import shutil
import traceback
import glob

# Import pipeline base class
from pipelines.base_pipeline import BasePipeline
from pipelines.image_classification_pipeline import ImageClassificationPipeline
from pipelines.object_detection_pipeline import ObjectDetectionPipeline
from pipelines.image_segmentation_pipeline import ImageSegmentationPipeline
from pipelines.speech_recognition_pipeline import SpeechRecognitionPipeline

# Import MLflow utilities
from mlflow_utils import (
    setup_mlflow, start_run, log_metrics, log_model,
    end_run, log_batch_metrics
)

from data_loaders import create_dataloaders


# ==================== Models & Schemas ====================

class TaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    SPEECH_RECOGNITION = "speech_recognition"
    STYLE_TRANSFER = "style_transfer"

class SegmentationType(str, Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"

class ModelArchitecture(str, Enum):
    # Classification architectures
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    EFFICIENTNET = "efficientnet"
    MOBILENET = "mobilenet"
    
    # Object detection architectures
    FASTER_RCNN = "faster_rcnn"
    SSD = "ssd"
    RETINANET = "retinanet"
    YOLO = "yolo"
    
    # Hugging Face transformer object detection architectures
    DETR_RESNET50 = "detr_resnet50"
    DETR_RESNET101 = "detr_resnet101"
    YOLOS_SMALL = "yolos_small"
    YOLOS_BASE = "yolos_base"
    OWLV2_BASE = "owlv2_base"
    
    # Segmentation architectures
    FCN = "fcn"
    DEEPLABV3 = "deeplabv3"
    MASK_RCNN = "mask_rcnn"
    UNET = "unet"
    
    # Speech recognition architectures
    WAV2VEC2_BERT = "wav2vec2_bert"
    WHISPER_SMALL = "whisper_small"
    WHISPER_BASE = "whisper_base"
    WHISPER_LARGE = "whisper_large"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class SegmentationType(str, Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"

class DatasetSource(str, Enum):
    LOCAL = "local"

    

class PipelineConfig(BaseModel):
    name: str   
    task_type: TaskType
    architecture: ModelArchitecture
    num_classes: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    image_size: tuple = (224, 224)
    augmentation_enabled: bool = True
    early_stopping: bool = True
    patience: int = 5  # Early stopping patience
    
    # Hugging Face specific configuration
    use_hf_transformers: bool = False
    hf_model_checkpoint: str = None  # Will be set based on architecture selection
    feature_extraction_only: bool = False
    patience: int = 5
    segmentation_type: Optional[SegmentationType] = SegmentationType.SEMANTIC
    dataset_source: Optional[DatasetSource] = DatasetSource.LOCAL
    
    # Speech recognition specific parameters
    language_code: str = "en"
    language: str = "english"
    target_sampling_rate: int = 16000
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0
    min_transcript_len: int = 10
    max_transcript_len: int = 300
    outlier_std_devs: float = 2.0
    apply_outlier_filtering: bool = True
    alphabet: str = " 'abcdefghijklmnopqrstuvwxyz"
    audio_column: str = "audio"
    text_column: str = "sentence"
    normalized_text_column: str = "normalized_text"
    duration_column: str = "duration"
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-3

class TrainingJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_config: PipelineConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Dict[str, float] = {}
    model_path: Optional[str] = None
    logs: List[str] = []    
    linked_dataset_id: Optional[str] = None  # <-- Add this field


# ==================== Pipeline Factory ====================

class PipelineFactory:
    """Factory to create appropriate pipeline based on task type"""
    
    # Define mappings from architecture enums to HF model checkpoints
    HF_MODEL_MAPPING = {
        ModelArchitecture.DETR_RESNET50: "facebook/detr-resnet-50",
        ModelArchitecture.DETR_RESNET101: "facebook/detr-resnet-101",
        ModelArchitecture.YOLOS_SMALL: "hustvl/yolos-small",
        ModelArchitecture.YOLOS_BASE: "hustvl/yolos-base",
        ModelArchitecture.OWLV2_BASE: "owlv2-base-patch16-ensemble",
        # Speech recognition models
        ModelArchitecture.WAV2VEC2_BERT: "facebook/w2v-bert-2.0",
        ModelArchitecture.WHISPER_SMALL: "openai/whisper-small",
        ModelArchitecture.WHISPER_BASE: "openai/whisper-base",
        ModelArchitecture.WHISPER_LARGE: "openai/whisper-large",
    }
    
    @staticmethod
    def create_pipeline(config: PipelineConfig) -> BasePipeline:
        # Check if using a Hugging Face transformer architecture
        if config.architecture in PipelineFactory.HF_MODEL_MAPPING:
            # Set HF-specific configuration
            config.use_hf_transformers = True
            config.hf_modezzl_checkpoint = PipelineFactory.HF_MODEL_MAPPING[config.architecture]
        
        if config.task_type == TaskType.IMAGE_CLASSIFICATION:
            from pipelines.image_classification_pipeline import ImageClassificationPipeline
            return ImageClassificationPipeline(config)
        elif config.task_type == TaskType.OBJECT_DETECTION:
            from pipelines.object_detection_pipeline import ObjectDetectionPipeline
            return ObjectDetectionPipeline(config)
        elif config.task_type == TaskType.IMAGE_SEGMENTATION:
            from pipelines.image_segmentation_pipeline import ImageSegmentationPipeline
            return ImageSegmentationPipeline(config)
        elif config.task_type == TaskType.SPEECH_RECOGNITION:
            from pipelines.speech_recognition_pipeline import SpeechRecognitionPipeline
            return SpeechRecognitionPipeline(config)
        else:
            raise ValueError(f"Unsupported task type: {config.task_type}")

# ==================== Utility Functions ====================

def smart_extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Smart ZIP extraction that handles nested folder structures.
    If ZIP contains a single root folder, extracts its contents directly.
    Otherwise, extracts everything as-is.
    """
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of all files/folders in the ZIP
        all_items = zip_ref.namelist()
        
        # Check if there's a common root folder
        if all_items:
            # Get the first level directories/files
            top_level_items = set()
            for item in all_items:
                # Split path and get first component
                parts = item.split('/')
                if parts[0]:  # Ignore empty strings
                    top_level_items.add(parts[0])
            
            # If there's only one top-level item and it's a directory
            if len(top_level_items) == 1:
                root_folder = list(top_level_items)[0]
                
                # Check if this root folder contains all the files
                all_in_root = all(item.startswith(root_folder + '/') or item == root_folder for item in all_items)
                
                if all_in_root:
                    # Extract to a temporary directory first
                    temp_extract = extract_to.parent / f"{extract_to.name}_temp"
                    temp_extract.mkdir(exist_ok=True)
                    
                    try:
                        # Extract everything to temp directory
                        zip_ref.extractall(temp_extract)
                        
                        # Move contents of root folder to final destination
                        root_folder_path = temp_extract / root_folder
                        if root_folder_path.exists() and root_folder_path.is_dir():
                            # Move all contents from root folder to extract_to
                            for item in root_folder_path.iterdir():
                                destination = extract_to / item.name
                                if destination.exists():
                                    if destination.is_dir():
                                        shutil.rmtree(destination)
                                    else:
                                        destination.unlink()
                                shutil.move(str(item), str(destination))
                        else:
                            # Fallback: normal extraction
                            zip_ref.extractall(extract_to)
                    finally:
                        # Clean up temp directory
                        if temp_extract.exists():
                            shutil.rmtree(temp_extract)
                    return
        
        # Normal extraction if no nested folder issue
        zip_ref.extractall(extract_to)

# ==================== Storage & Job Management ====================

class JobManager:
    """Manages training jobs and their lifecycle"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.loaded_models: Dict[str, tuple[nn.Module, Dict]] = {}  # Cache for loaded models
        
        # Initialize MLflow
        setup_mlflow()
    
    def create_job(self, config: PipelineConfig) -> TrainingJob:
        """Create a new training job"""
        job = TrainingJob(pipeline_config=config)
        self.jobs[job.id] = job
        return job
    
    async def start_job(self, job_id: str, dataset_path: str):
        """Start a training job with the given dataset"""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Ensure models directory exists
        Path("models").mkdir(exist_ok=True)
        
        # Create model directory for this job
        job_model_dir = Path(f"models/{job_id}")
        job_model_dir.mkdir(exist_ok=True)
        
        # Set job status to running
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        # Create pipeline for this job
        pipeline = PipelineFactory.create_pipeline(job.pipeline_config)
        
        # Start training in the background
        asyncio.create_task(self._run_training(pipeline, job_id, dataset_path))
        
        return {"message": "Training started", "job_id": job_id}
    
    async def _run_training(self, pipeline: BasePipeline, job_id: str, dataset_path: str):
        """Run the training pipeline"""
        try:
            result = await pipeline.train(dataset_path, job_id)
            
            job = self.jobs[job_id]
            job.status = TrainingStatus.COMPLETED if result["status"] == "completed" else TrainingStatus.FAILED
            job.completed_at = datetime.now()
            
            if result["status"] == "completed":
                job.model_path = result["model_path"]
                if "metrics" in result:
                    job.metrics.update(result["metrics"])
                job.logs.append(f"Training completed successfully. Model saved to {result['model_path']}")
                
                # Clear model from cache if it exists
                if job_id in self.loaded_models:
                    del self.loaded_models[job_id]
            else:
                job.logs.append(f"Training failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            job = self.jobs[job_id]
            job.status = TrainingStatus.FAILED
            job.completed_at = datetime.now()
            job.logs.append(f"Training failed with exception: {str(e)}")
        
        finally:
            # Clean up
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> List[TrainingJob]:
        """List all jobs"""
        return list(self.jobs.values())
    
    def _load_model(self, model_path: str, pipeline_config: PipelineConfig) -> nn.Module:
        """Load a model from a saved checkpoint"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to load from MLflow if available
        try:
            import mlflow
            # Check if model is in MLflow and load from there
            try:
                # Find the run_id from the job_id (model_path contains job_id)
                job_id = Path(model_path).stem
                # Try to load model from the MLflow artifacts
                model = mlflow.pytorch.load_model(f"models:/{job_id}/production", map_location=device)
                model.eval()
                return model
            except Exception:
                # Fall back to local file if MLflow loading fails
                pass
        except ImportError:
            # MLflow not available, use local file
            pass
    
        # Create pipeline to get model architecture
        pipeline = PipelineFactory.create_pipeline(pipeline_config)
        model = pipeline.create_model()
        
        # Load saved weights from local file
        # Since this is our own checkpoint, we can safely use weights_only=False
        try:
            # First try with weights_only=True for security
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model
        except Exception as e:
            # Fall back to weights_only=False since this is our own trusted checkpoint
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                model.eval()
                return model
            except Exception as nested_e:
                raise ValueError(f"Failed to load model: {str(e)} | {str(nested_e)}")
    
    def _get_transform(self, pipeline_config: PipelineConfig) -> transforms.Compose:
        """Get the transforms for a model"""
        pipeline = PipelineFactory.create_pipeline(pipeline_config)
        return pipeline.get_transforms()
    
    def _get_class_map(self, model_path: str) -> Dict:
        """Get class mapping from saved model"""
        # Try to get class mapping from MLflow
        try:
            import mlflow
            job_id = Path(model_path).stem
            
            # Try to get class mapping from MLflow artifacts
            try:
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(
                    experiment_ids=[mlflow.get_experiment_by_name("no-code-ml-experiments").experiment_id],
                    filter_string=f"attributes.run_name = 'job_{job_id}'"
                )
                
                if runs:
                    run_id = runs[0].info.run_id
                    artifact_path = client.download_artifacts(run_id, "class_to_idx.json")
                    
                    if artifact_path:
                        with open(artifact_path, 'r') as f:
                            import json
                            class_to_idx = json.load(f)
                            # Create a reverse mapping from index to class name
                            return {str(idx): cls for cls, idx in class_to_idx.items()}
            except Exception:
                # Fall back to local file
                pass
        except ImportError:
            # MLflow not available
            pass
            
        # Fall back to local file - use weights_only=False since this is our own checkpoint
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and 'class_to_idx' in checkpoint:
                # Create a reverse mapping from index to class name
                return {str(idx): cls for cls, idx in checkpoint['class_to_idx'].items()}
        except Exception as e:
            print(f"Warning: Failed to load class mapping from {model_path}: {e}")
        return {}
    
    async def predict(self, job_id: str, image: Image.Image) -> Dict:
        """Make prediction using a trained model"""
        job = self.get_job(job_id)
        if not job or job.status != TrainingStatus.COMPLETED:
            raise ValueError("Model not ready for prediction")
            
        if not job.model_path or not Path(job.model_path).exists():
            raise ValueError(f"Model file not found: {job.model_path}")
        
        # Load model if not already in cache
        if job_id not in self.loaded_models:
            try:
                model = self._load_model(job.model_path, job.pipeline_config)
                class_map = self._get_class_map(job.model_path)
                self.loaded_models[job_id] = (model, class_map)
            except Exception as e:
                raise ValueError(f"Failed to load model: {str(e)}")
        
        model, class_map = self.loaded_models[job_id]
        
        # Create pipeline based on task type
        pipeline = PipelineFactory.create_pipeline(job.pipeline_config)
        
        # Use the pipeline's predict method to get predictions
        try:
            result = await pipeline.predict(image, model)
            print(f"Pipeline prediction result keys: {list(result.keys())}")
            
            # Process the prediction result based on task type
            if job.pipeline_config.task_type == TaskType.IMAGE_CLASSIFICATION:
                # Format the classification results - handle the actual pipeline output format
                if "predictions" in result:
                    # New format from pipeline
                    pipeline_predictions = result["predictions"]
                    formatted_predictions = []
                    
                    for pred in pipeline_predictions:
                        class_idx = pred["class_id"]
                        confidence = pred["confidence"] * 100  # Convert to percentage
                        class_name = class_map.get(str(class_idx), f"Class {class_idx}")
                        
                        formatted_predictions.append({
                            "class_id": class_idx,
                            "class_name": class_name,
                            "confidence": confidence
                        })
                    
                    return {
                        "predictions": formatted_predictions,
                        "top_prediction": formatted_predictions[0] if formatted_predictions else None,
                        "task_type": "image_classification"
                    }
                else:
                    # Fallback for old format (if any)
                    probabilities = result.get("probabilities", [])
                    predicted_classes = result.get("predicted_classes", [])
                    confidence_scores = result.get("confidence_scores", []);
                    
                    predictions = []
                    for i, (class_idx, score) in enumerate(zip(predicted_classes, confidence_scores)):
                        class_name = class_map.get(str(class_idx), f"Class {class_idx}")
                        confidence = score * 100
                        predictions.append({
                            "class_id": class_idx,
                            "class_name": class_name,
                            "confidence": confidence
                        })
                    
                    return {
                        "predictions": predictions,
                        "top_prediction": predictions[0] if predictions else None,
                        "all_probabilities": probabilities
                    }
            
            elif job.pipeline_config.task_type == TaskType.OBJECT_DETECTION:
                # Handle new format from ObjectDetectionPipeline
                if "predictions" in result:
                    # New format: result contains ['predictions', 'status', 'num_detections']
                    detections = result.get("predictions", [])
                    # detections is already in the correct format with keys: 'box', 'confidence', 'class_name', 'label'
                    
                    # Convert to the expected API format
                    formatted_detections = []
                    for detection in detections:
                        formatted_detections.append({
                            "box": detection["box"],  # [x1, y1, x2, y2]
                            "class_id": detection["label"],
                            "class_name": detection["class_name"],
                            "confidence": detection["confidence"]
                        })
                    
                    return {
                        "detections": formatted_detections,
                        "count": len(formatted_detections)
                    }
                else:
                    # Old format (fallback): result contains ['boxes', 'scores', 'labels']
                    boxes = result["boxes"]
                    scores = result["scores"]
                    labels = result["labels"]
                    
                    detections = []
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        class_name = class_map.get(str(label), f"Class {label}")
                        confidence = score * 100
                        detections.append({
                            "box": box,  # [x1, y1, x2, y2]
                            "class_id": label,
                            "class_name": class_name,
                            "confidence": confidence
                        })
                    
                    return {
                        "detections": detections,
                        "count": len(detections)
                    }
            
            elif job.pipeline_config.task_type == TaskType.IMAGE_SEGMENTATION:
                # Format the segmentation results
                if job.pipeline_config.segmentation_type == SegmentationType.SEMANTIC:
                    # For semantic segmentation
                    segmentation_mask = result["segmentation_mask"]
                    probabilities = result.get("probabilities", [])
                    
                    # Convert class indices to class names for visualization
                    class_mapping = {int(idx): name for name, idx in class_map.items()}
                    
                    return {
                        "segmentation_mask": segmentation_mask,
                        "class_mapping": class_mapping,
                        "probabilities": probabilities
                    }
                else:
                    # For instance segmentation
                    masks = result.get("masks", [])
                    boxes = result.get("boxes", [])
                    scores = result.get("scores", [])
                    labels = result.get("labels", [])
                    
                    instances = []
                    for i, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
                        class_name = class_map.get(str(label), f"Class {label}")
                        confidence = score * 100
                        instances.append({
                            "mask": mask,
                            "box": box,
                            "class_id": label,
                            "class_name": class_name,
                            "confidence": confidence
                        })
                    
                    return {
                        "instances": instances,
                        "count": len(instances)
                    }
            else:
                return result
                
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its associated files"""
        if job_id not in self.jobs:
            return False
            
        # Stop running job if exists
        if job_id in self.running_jobs:
            self.running_jobs[job_id].cancel()
            del self.running_jobs[job_id]
        
        # Remove model from cache if loaded
        if job_id in self.loaded_models:
            del self.loaded_models[job_id]
            
        # Delete model file if exists
        job = self.jobs[job_id]
        if job.model_path and Path(job.model_path).exists():
            Path(job.model_path).unlink()
            
        # Delete dataset if exists
        dataset_path = Path(f"datasets/{job_id}")
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)
            
        # Remove from jobs dict
        del self.jobs[job_id]
        return True

# ==================== FastAPI Application ====================

app = FastAPI(title="No-Code AI Platform", version="1.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import MLflow server utilities
from mlflow_server import (
    start_mlflow_server, stop_mlflow_server, 
    get_mlflow_ui_url, get_experiment_details
)

# Global job manager instance
job_manager = JobManager()

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "No-Code AI Platform API", 
        "version": "1.0.0",
        "pytorch_version": torch.__version__,
        "gpu_available": torch.cuda.is_available()
    }

# MLflow integration endpoints
@app.post("/mlflow/start-server")
async def start_mlflow():
    """Start the MLflow UI server"""
    result = start_mlflow_server()
    return {"message": result}

@app.post("/mlflow/stop-server")
async def stop_mlflow():
    """Stop the MLflow UI server"""
    result = stop_mlflow_server()
    return {"message": result}

@app.get("/mlflow/ui-url")
async def get_ui_url():
    """Get the URL for the MLflow UI"""
    return {"url": get_mlflow_ui_url()}

@app.get("/mlflow/experiments")
async def get_experiments():
    """Get details about MLflow experiments"""
    return get_experiment_details()

@app.get("/jobs/{job_id}/mlflow")
async def get_job_mlflow_info(job_id: str):
    """Get MLflow information for a specific job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    from mlflow_utils import EXPERIMENT_NAME
    
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return {"message": "No MLflow experiment found"}
        
        # Search for runs with this job ID
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attributes.run_name = 'job_{job_id}'"
        )
        
        if not runs:
            return {"message": f"No MLflow runs found for job {job_id}"}
        
        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "mlflow_ui_url": f"{get_mlflow_ui_url()}/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}"
        }
    except Exception as e:
        return {"message": f"Error retrieving MLflow information: {str(e)}"}

@app.post("/pipelines", response_model=TrainingJob)
async def create_pipeline(config: PipelineConfig):
    """Create a new training pipeline"""
    try:
        job = job_manager.create_job(config)
        return job
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/pipelines/{job_id}/train")
async def start_training(job_id: str, background_tasks: BackgroundTasks):
    """Start training for a specific job"""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if not job.linked_dataset_id:
            raise HTTPException(status_code=400, detail="No dataset linked to this job. Please link a dataset first.")
        dataset_path = f"datasets/{job.linked_dataset_id}"
        result = await job_manager.start_job(job_id, dataset_path)
        return result  
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/pipelines/{job_id}", response_model=TrainingJob)
async def get_pipeline_status(job_id: str):
    """Get the status of a training job"""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/pipelines", response_model=List[TrainingJob])
async def list_pipelines():
    """List all training jobs"""
    return job_manager.list_jobs()


@app.get("/datasets/versions")
async def list_dataset_versions(
    dataset_name: Optional[str] = None,
    source: Optional[str] = None
):
    """List all dataset versions with optional filtering"""
    try:
        from dataset_versioning import DatasetVersionManager
        version_manager = DatasetVersionManager("datasets")
        
        versions = version_manager.list_dataset_versions(dataset_name, source)
        return {
            "count": len(versions),
            "versions": [v.to_dict() for v in versions]
        }
    except ImportError:
        raise HTTPException(status_code=500, detail="Dataset versioning system not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list dataset versions: {str(e)}")

@app.get("/datasets/versions/{job_id}")
async def get_dataset_version(job_id: str):
    """Get detailed information about a specific dataset version"""
    try:
        from dataset_versioning import DatasetVersionManager
        version_manager = DatasetVersionManager("datasets")
        
        version = version_manager.get_dataset_version(job_id)
        if not version:
            raise HTTPException(status_code=404, detail=f"Dataset version {job_id} not found")
        
        return version.to_dict()
    except ImportError:
        raise HTTPException(status_code=500, detail="Dataset versioning system not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset version: {str(e)}")

@app.delete("/datasets/{job_id}")
async def delete_dataset(job_id: str):
    """Delete a dataset and its version information"""
    try:
        from dataset_versioning import DatasetVersionManager
        version_manager = DatasetVersionManager("datasets")
        
        success = version_manager.delete_dataset(job_id)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete dataset {job_id}")
        
        return {"message": f"Dataset {job_id} deleted successfully"}
    except ImportError:
        # Fall back to basic deletion if versioning is not available
        dataset_dir = Path(f"datasets/{job_id}")
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"Dataset {job_id} not found")
        
        try:
            import shutil
            shutil.rmtree(dataset_dir)
            return {"message": f"Dataset {job_id} deleted successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

# ==================== Dataset Upload Endpoints ====================

@app.post("/upload-dataset/{job_id}/{class_name}")
async def create_dataset_class(job_id: str, class_name: str):
    """Create a class directory in the dataset folder"""
    try:
        # Create dataset class directory
        class_dir = Path(f"datasets/{job_id}/{class_name}")
        class_dir.mkdir(parents=True, exist_ok=True)
        return {"message": f"Created class directory {class_name} for job {job_id}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-dataset/{job_id}")
async def upload_dataset_file(
    job_id: str,
    file: UploadFile = File(...),
    class_name: str = Form(None),
    file_type: str = Form("image")  # 'image', 'annotation', or 'zip'
):
    """Upload a file to the dataset folder. Supports images, annotation files, and zipped COCO datasets."""
    try:
        dataset_dir = Path(f"datasets/{job_id}")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if file_type == "zip":
            # Save and extract zip file using smart extraction
            zip_path = dataset_dir / file.filename
            with open(zip_path, "wb") as f:
                content = await file.read()
                f.write(content)
            smart_extract_zip(zip_path, dataset_dir)
            zip_path.unlink()  # Remove zip after extraction
            return {"message": f"Extracted {file.filename} to {dataset_dir}"}

        elif file_type == "annotation" and file.filename.lower().endswith(".json"):
            # Save annotation file to annotations/ or root
            annotations_dir = dataset_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            file_path = annotations_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            return {"message": f"Uploaded annotation {file.filename} to {annotations_dir}"}

        else:
            # Default: treat as image, requires class_name
            if not class_name:
                raise HTTPException(status_code=400, detail="class_name is required for image upload.")
            class_dir = dataset_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            file_path = class_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            return {"message": f"Uploaded {file.filename} to {class_name} directory"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/{job_id}")
async def predict(job_id: str, file: UploadFile = File(...)):
    """Make predictions using a trained model"""
    import time
    start_time = time.time()
    
    try:
        print(f"Prediction request for job_id: {job_id}")
        
        # Check if job exists
        job = job_manager.get_job(job_id)
        if not job:
            print(f"Job {job_id} not found")
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        print(f"Job status: {job.status}")
        print(f"Model path: {job.model_path}")
        
        if job.status != TrainingStatus.COMPLETED:
            print(f"Model not ready. Status: {job.status}")
            raise HTTPException(status_code=400, detail=f"Model not ready for prediction. Status: {job.status}")
        
        if not job.model_path:
            print("No model path found")
            raise HTTPException(status_code=400, detail="No model path found for this job")
        
        # Check if model file exists
        from pathlib import Path
        if not Path(job.model_path).exists():
            print(f"Model file not found: {job.model_path}")
            raise HTTPException(status_code=400, detail=f"Model file not found: {job.model_path}")
        
        # Load image
        print("Loading image...")
        image_data = await file.read()
        print(f"Image data size: {len(image_data)} bytes")
        
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"Image loaded: {image.size}, mode: {image.mode}")
        except Exception as img_error:
            print(f"Error loading image: {img_error}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(img_error)}")
        
        # Load model and make prediction
        print("Making prediction...")
        try:
            prediction_result = await job_manager.predict(job_id, image)
            print("Prediction successful")
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(pred_error)}")
        
        # Add task type to the response for client-side handling
        prediction_result["task_type"] = job.pipeline_config.task_type
        if job.pipeline_config.task_type == TaskType.IMAGE_SEGMENTATION:
            prediction_result["segmentation_type"] = job.pipeline_config.segmentation_type
        
        # Calculate processing time
        processing_time = time.time() - start_time
        prediction_result["processing_time"] = processing_time
        
        # Import visualization utilities
        from visualization_utils import (
            draw_bounding_boxes, draw_segmentation_mask, 
            pil_to_base64, log_prediction_results
        )
        
        # Log prediction results
        log_prediction_results(
            job_id, 
            job.pipeline_config.task_type, 
            prediction_result, 
            processing_time
        )
        
        # Create annotated images for visualization
        if job.pipeline_config.task_type == TaskType.OBJECT_DETECTION:
            # Draw bounding boxes on the image
            detections = prediction_result.get("detections", [])
            if detections:
                annotated_image = draw_bounding_boxes(image, detections)
                # Convert to base64 for frontend display
                prediction_result["annotated_image"] = pil_to_base64(annotated_image)
                print(f"Created annotated image with {len(detections)} detections")
            else:
                # No detections, return original image
                prediction_result["annotated_image"] = pil_to_base64(image)
                print("No detections found, returning original image")
                
        elif job.pipeline_config.task_type == TaskType.IMAGE_SEGMENTATION:
            # Draw segmentation mask overlay
            if job.pipeline_config.segmentation_type == SegmentationType.SEMANTIC:
                segmentation_mask = prediction_result.get("segmentation_mask")
                class_mapping = prediction_result.get("class_mapping", {})
                if segmentation_mask is not None and class_mapping:
                    annotated_image = draw_segmentation_mask(image, segmentation_mask, class_mapping)
                    prediction_result["annotated_image"] = pil_to_base64(annotated_image)
                    print("Created segmentation overlay")
                else:
                    prediction_result["annotated_image"] = pil_to_base64(image)
            else:
                # For instance segmentation, just return original for now
                # TODO: Implement instance segmentation visualization
                prediction_result["annotated_image"] = pil_to_base64(image)
                
        elif job.pipeline_config.task_type == TaskType.IMAGE_CLASSIFICATION:
            # For classification, just return the original image
            prediction_result["annotated_image"] = pil_to_base64(image)
            
        print("Returning prediction result with annotations")
        return prediction_result
    
    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected error in prediction endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
# ==================== Pipeline Management Endpoints ====================
@app.delete("/pipelines/{job_id}")
async def delete_pipeline(job_id: str):
    """Delete a training job and its associated resources"""
    success = job_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_jobs": len(job_manager.running_jobs)
    }

@app.get("/datasets/available")
async def list_available_datasets():
    """List all available datasets"""
    try:
        dataset_dir = Path("datasets")
        if not dataset_dir.exists():
            return []
        
        datasets = []
        for d in dataset_dir.iterdir():
            if d.is_dir():
                # Skip special directories
                if d.name.startswith('.') or d.name == '__pycache__':
                    continue
                
                # Try to determine the task type based on directory structure or content
                task_type = "image_classification"  # Default
                is_coco_dataset = False
                
                # Check for COCO format (annotations directory or JSON files)
                annotations_dir = d / "annotations"
                if annotations_dir.exists() and annotations_dir.is_dir():
                    json_files = list(annotations_dir.glob("*.json"))
                    if json_files:
                        is_coco_dataset = True
                        task_type = "object_detection"
                
                # Also check for JSON files in the root directory that might be COCO annotations
                if not is_coco_dataset:
                    json_files = list(d.glob("*.json"))
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                content = json.load(f)
                                # Simple check for COCO format
                                if all(key in content for key in ["images", "annotations", "categories"]):
                                    is_coco_dataset = True
                                    task_type = "object_detection"
                                    break
                        except:
                            continue
                
                # Look for directory structure indicators
                if not is_coco_dataset and "detection" in d.name.lower():
                    task_type = "object_detection"
                elif "segmentation" in d.name.lower():
                    task_type = "image_segmentation"
                
                # Check for class directories (for classification datasets)
                classes = [c.name for c in d.iterdir() if c.is_dir() and not c.name == "annotations"]
                
                # Count items (images) in the dataset
                item_count = 0
                if is_coco_dataset:
                    # For COCO datasets, count all images in the dataset recursively
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                    for ext in image_extensions:
                        found_images = list(d.glob(f"**/*{ext}"))
                        item_count += len(found_images)
                    
                    # If still 0, try a more comprehensive search (might be slower but more reliable)
                    if item_count == 0:
                        import glob
                        for ext in image_extensions:
                            found_images = glob.glob(str(d) + f"/**/*{ext}", recursive=True)
                            item_count += len(found_images)
                    
                    # Print debug info
                    print(f"Found {item_count} images in COCO dataset: {d.name}")
                    
                    # Try to get class names from the COCO annotations
                    if not classes:
                        # Use either json_files from annotations dir or the ones found in root
                        coco_jsons = json_files if json_files else list(d.glob("**/*.json"))
                        for json_file in coco_jsons:
                            try:
                                with open(json_file, 'r') as f:
                                    content = json.load(f)
                                    if "categories" in content:
                                        classes = [cat["name"] for cat in content["categories"]]
                                        print(f"Found {len(classes)} classes in COCO dataset: {d.name}")
                                        break
                            except Exception as e:
                                print(f"Error parsing JSON {json_file}: {str(e)}")
                                continue
                else:
                    # For classification datasets, count by class
                    if classes:
                        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                        for cls in classes:
                            class_path = d / cls
                            image_files = [f for f in class_path.iterdir() 
                                        if f.is_file() and any(f.name.lower().endswith(ext) for ext in image_extensions)]
                            item_count += len(image_files)
                
                # Skip empty datasets - for COCO datasets, we don't require classes to be detected
                if not is_coco_dataset and not classes:
                    continue
                
                # For COCO datasets with no detected images, skip as well
                if is_coco_dataset and item_count == 0:
                    print(f"Skipping COCO dataset with no images: {d.name}")
                    continue
                
                datasets.append({
                    "id": d.name,
                    "name": d.name.replace('_', ' ').title(),
                    "classes": classes if classes else ["(COCO format dataset)"],
                    "task_type": task_type,
                    "item_count": item_count,
                    "is_coco_format": is_coco_dataset
                })
        
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATASETS_DIR = BASE_DIR / "datasets"

@app.post("/pipelines/{job_id}/dataset/{dataset_id}")
async def link_dataset_to_job(job_id: str, dataset_id: str):
    try:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        dataset_path = DATASETS_DIR / dataset_id
        print(f"Linking dataset: {dataset_id}")
        print(f"Dataset path: {dataset_path}")
        print(f"Dataset path exists: {dataset_path.exists()}")
        print(f"Dataset path is directory: {dataset_path.is_dir() if dataset_path.exists() else False}")
        
        all_datasets = [d.name for d in DATASETS_DIR.iterdir() if d.is_dir()]
        print(f"Available datasets: {all_datasets}")
        
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        job.linked_dataset_id = dataset_id
        return {"message": f"Successfully linked dataset {dataset_id} to job {job_id}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to link dataset: {str(e)}")

@app.post("/upload-detection-dataset/{job_id}")
async def upload_detection_dataset(job_id: str, file: UploadFile = File(...)):
    """
    Upload a COCO format object detection dataset as a zip file.
    The zip file should contain the images and annotations in the COCO format structure.
    The entire structure is preserved when extracting.
    """
    try:
        dataset_dir = Path(f"datasets/{job_id}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify that the uploaded file is a zip file
        if not file.filename.lower().endswith(('.zip')):
            raise HTTPException(status_code=400, detail="Only ZIP files are supported for object detection datasets")
        
        # Save the zip file temporarily
        zip_path = dataset_dir / file.filename
        with open(zip_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract the zip file using smart extraction to avoid nested folders
        smart_extract_zip(zip_path, dataset_dir)
        
        # Delete the temporary zip file
        zip_path.unlink()
        
        return {
            "message": f"Object detection dataset uploaded and extracted successfully to {dataset_dir}",
            "dataset_id": job_id,
            "task_type": "object_detection"
        }
    except Exception as e:
        # If something goes wrong, delete any partial files
        if dataset_dir.exists():
            import shutil
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except:
                pass
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")

@app.post("/upload-speech-dataset/{job_id}")
async def upload_speech_dataset(job_id: str, file: UploadFile = File(...)):
    """
    Upload a speech recognition dataset as a zip file.
    Supports both CSV metadata format and AudioFolder format.
    
    Expected structures:
    1. CSV Format:
       dataset.zip/
         train/metadata.csv (columns: file_name, sentence)
         train/audio1.wav
         validation/metadata.csv
         validation/audio2.wav
         test/metadata.csv
         test/audio3.wav
    
    2. AudioFolder Format:
       dataset.zip/
         train/audio1.wav
         train/audio2.wav
         metadata.jsonl (or other HF format files)
    """
    try:
        dataset_dir = Path(f"datasets/{job_id}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify that the uploaded file is a zip file
        if not file.filename.lower().endswith(('.zip')):
            raise HTTPException(status_code=400, detail="Only ZIP files are supported for speech datasets")
        
        # Save the zip file temporarily
        zip_path = dataset_dir / file.filename
        with open(zip_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract the zip file using smart extraction to avoid nested folders
        smart_extract_zip(zip_path, dataset_dir)
        
        # Delete the temporary zip file
        zip_path.unlink()
        
        # Detect dataset format
        from pipelines.speech_utils import detect_dataset_format
        dataset_format = detect_dataset_format(str(dataset_dir))
        
        # Validate the dataset structure
        validation_result = await validate_speech_dataset_structure(dataset_dir, dataset_format)
        
        return {
            "message": f"Speech recognition dataset uploaded and extracted successfully to {dataset_dir}",
            "dataset_id": job_id,
            "task_type": "speech_recognition",
            "dataset_format": dataset_format,
            "validation": validation_result
        }
    except Exception as e:
        # If something goes wrong, delete any partial files
        if dataset_dir.exists():
            import shutil
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except:
                pass
        raise HTTPException(status_code=400, detail=f"Failed to process speech dataset: {str(e)}")


async def validate_speech_dataset_structure(dataset_dir: Path, dataset_format: str) -> Dict[str, Any]:
    """Validate the structure of an uploaded speech dataset"""
    validation_result = {
        "format": dataset_format,
        "splits_found": [],
        "total_samples": 0,
        "issues": [],
        "recommendations": []
    }
    
    try:
        if dataset_format == "csv":
            # Validate CSV format
            for potential_split in ["train", "validation", "test"]:
                split_dir = dataset_dir / potential_split
                csv_file = split_dir / "metadata.csv"
                
                if split_dir.exists() and csv_file.exists():
                    import pandas as pd
                    
                    # Read and validate CSV
                    try:
                        df = pd.read_csv(csv_file)
                        required_columns = ["file_name", "sentence"]
                        missing_columns = [col for col in required_columns if col not in df.columns]
                        
                        if missing_columns:
                            validation_result["issues"].append(
                                f"Missing required columns in {potential_split}/metadata.csv: {missing_columns}"
                            )
                        else:
                            # Count valid audio files
                            valid_count = 0
                            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
                            
                            for _, row in df.iterrows():
                                audio_file = split_dir / row["file_name"]
                                if audio_file.exists() and audio_file.suffix.lower() in audio_extensions:
                                    valid_count += 1
                                else:
                                    validation_result["issues"].append(
                                        f"Missing audio file: {potential_split}/{row['file_name']}"
                                    )
                            
                            validation_result["splits_found"].append({
                                "split": potential_split,
                                "samples": valid_count,
                                "csv_rows": len(df)
                            })
                            validation_result["total_samples"] += valid_count
                            
                    except Exception as e:
                        validation_result["issues"].append(f"Error reading {potential_split}/metadata.csv: {str(e)}")
                        
        elif dataset_format == "audiofolder":
            # Validate AudioFolder format
            for potential_split in dataset_dir.iterdir():
                if potential_split.is_dir():
                    audio_files = [
                        f for f in potential_split.iterdir() 
                        if f.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
                    ]
                    
                    if audio_files:
                        validation_result["splits_found"].append({
                            "split": potential_split.name,
                            "samples": len(audio_files)
                        })
                        validation_result["total_samples"] += len(audio_files)
        
        # Add recommendations
        if validation_result["total_samples"] < 10:
            validation_result["recommendations"].append(
                "Dataset is very small (<10 samples). Consider adding more data for better training results."
            )
        
        if len(validation_result["splits_found"]) < 2:
            validation_result["recommendations"].append(
                "Consider adding validation and test splits for better evaluation."
            )
        
        if dataset_format == "unknown":
            validation_result["issues"].append(
                "Unknown dataset format. Please ensure your dataset follows CSV or AudioFolder structure."
            )
            validation_result["recommendations"].append(
                "For CSV format: create train/validation/test directories with metadata.csv files. "
                "For AudioFolder format: place audio files directly in split directories."
            )
            
    except Exception as e:
        validation_result["issues"].append(f"Validation error: {str(e)}")
    
    return validation_result


@app.get("/validate-speech-dataset/{job_id}")
async def validate_speech_dataset_endpoint(job_id: str):
    """Validate an existing speech dataset structure"""
    try:
        dataset_dir = Path(f"datasets/{job_id}")
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        from pipelines.speech_utils import detect_dataset_format
        dataset_format = detect_dataset_format(str(dataset_dir))
        
        validation_result = await validate_speech_dataset_structure(dataset_dir, dataset_format)
        
        return {
            "dataset_id": job_id,
            "validation": validation_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.get("/speech-dataset-info/{job_id}")
async def get_speech_dataset_info(job_id: str):
    """Get detailed information about a speech dataset"""
    try:
        dataset_dir = Path(f"datasets/{job_id}")
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        from pipelines.speech_utils import detect_dataset_format, load_csv_speech_dataset
        
        dataset_format = detect_dataset_format(str(dataset_dir))
        info = {
            "dataset_id": job_id,
            "format": dataset_format,
            "directory": str(dataset_dir),
            "splits": {}
        }
        
        if dataset_format == "csv":
            try:
                # Load dataset to get detailed info
                dataset = load_csv_speech_dataset(
                    dataset_path=str(dataset_dir),
                    target_sampling_rate=None  # Don't load audio for info
                )
                
                for split_name, split_data in dataset.items():
                    info["splits"][split_name] = {
                        "samples": len(split_data),
                        "columns": split_data.column_names
                    }
                    
                    # Sample a few examples for preview
                    if len(split_data) > 0:
                        sample_size = min(3, len(split_data))
                        samples = []
                        for i in range(sample_size):
                            sample = split_data[i]
                            samples.append({
                                "sentence": sample.get("sentence", ""),
                                "audio_path": sample.get("audio", {}).get("path", ""),
                                "speaker_id": sample.get("speaker_id", ""),
                                "duration": sample.get("duration", "")
                            })
                        info["splits"][split_name]["samples_preview"] = samples
                        
            except Exception as e:
                info["error"] = f"Failed to load CSV dataset: {str(e)}"
                
        elif dataset_format == "audiofolder":
            # Get basic info for AudioFolder format
            for item in dataset_dir.iterdir():
                if item.is_dir():
                    audio_files = [
                        f.name for f in item.iterdir() 
                        if f.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
                    ]
                    info["splits"][item.name] = {
                        "samples": len(audio_files),
                        "sample_files": audio_files[:5]  # Show first 5 files
                    }
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")

@app.post("/create-example-speech-dataset/{job_id}")
async def create_example_speech_dataset(
    job_id: str,
    train_samples: int = 10,
    val_samples: int = 3,
    test_samples: int = 3,
    language: str = "en"
):
    """
    Create an example speech dataset for testing purposes.
    Generates silent audio files with sample transcriptions.
    """
    try:
        import tempfile
        import shutil
        import numpy as np
        import pandas as pd
        
        dataset_dir = Path(f"datasets/{job_id}")
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample sentences for different languages
        sample_sentences = {
            "en": [
                "Hello world, this is a test recording",
                "The quick brown fox jumps over the lazy dog",
                "Speech recognition technology is advancing rapidly",
                "Machine learning enables automatic transcription",
                "Deep learning models can process audio signals",
                "Artificial intelligence transforms how we interact with computers",
                "Natural language processing helps understand human speech",
                "Voice assistants are becoming increasingly popular",
                "Audio processing requires sophisticated algorithms",
                "Digital signal processing enables real-time speech analysis"
            ],
            "es": [
                "Hola mundo, esta es una grabación de prueba",
                "El reconocimiento de voz está avanzando rápidamente",
                "La inteligencia artificial transforma la tecnología",
                "Los asistentes de voz son cada vez más populares",
                "El procesamiento de audio requiere algoritmos sofisticados"
            ],
            "fr": [
                "Bonjour le monde, ceci est un enregistrement de test",
                "La reconnaissance vocale progresse rapidement",
                "L'intelligence artificielle transforme la technologie",
                "Les assistants vocaux deviennent de plus en plus populaires",
                "Le traitement audio nécessite des algorithmes sophistiqués"
            ]
        }
        
        sentences = sample_sentences.get(language, sample_sentences["en"])
        
        def create_silent_audio(filepath, duration=2.0, sample_rate=16000):
            """Create a silent audio file"""
            try:
                import soundfile as sf
                # Create silent audio
                samples = int(duration * sample_rate)
                audio_data = np.zeros(samples, dtype=np.float32)
                sf.write(filepath, audio_data, sample_rate)
            except ImportError:
                # Fallback: create empty WAV header
                import wave
                with wave.open(str(filepath), 'w') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(b'\x00' * int(duration * sample_rate * 2))
        
        # Create splits and metadata
        splits_data = {
            "train": train_samples,
            "validation": val_samples,
            "test": test_samples
        }
        
        audio_counter = 1
        total_created = 0
        
        for split_name, num_samples in splits_data.items():
            if num_samples <= 0:
                continue
                
            split_dir = dataset_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Generate metadata
            metadata_rows = []
            for i in range(num_samples):
                filename = f"audio_{audio_counter:04d}.wav"
                sentence = sentences[i % len(sentences)]
                duration = np.random.uniform(1.0, 3.0)  # Random duration 1-3 seconds
                speaker_id = f"speaker_{(i % 5) + 1:03d}"  # 5 different speakers
                
                metadata_rows.append({
                    "file_name": filename,
                    "sentence": sentence,
                    "speaker_id": speaker_id,
                    "duration": duration,
                    "language": language
                })
                
                # Create silent audio file
                audio_path = split_dir / filename
                create_silent_audio(audio_path, duration)
                
                audio_counter += 1
                total_created += 1
            
            # Save metadata CSV
            df = pd.DataFrame(metadata_rows)
            df.to_csv(split_dir / "metadata.csv", index=False)
        
        # Validate the created dataset
        from pipelines.speech_utils import detect_dataset_format
        dataset_format = detect_dataset_format(str(dataset_dir))
        validation_result = await validate_speech_dataset_structure(dataset_dir, dataset_format)
        
        return {
            "message": f"Example speech dataset created successfully",
            "dataset_id": job_id,
            "task_type": "speech_recognition",
            "total_samples": total_created,
            "splits": {split: num for split, num in splits_data.items() if num > 0},
            "language": language,
            "dataset_format": dataset_format,
            "validation": validation_result
        }
        
    except Exception as e:
        # Clean up on error
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        raise HTTPException(status_code=500, detail=f"Failed to create example dataset: {str(e)}")


@app.get("/supported-speech-formats")
async def get_supported_speech_formats():
    """Get information about supported speech dataset formats and requirements"""
    return {
        "formats": {
            "csv": {
                "description": "CSV metadata format with separate audio files",
                "structure": {
                    "train/": "Training split directory",
                    "train/metadata.csv": "CSV with file_name and sentence columns",
                    "train/audio_*.wav": "Audio files referenced in CSV",
                    "validation/": "Validation split directory (optional)",
                    "test/": "Test split directory (optional)"
                },
                "required_columns": ["file_name", "sentence"],
                "optional_columns": ["speaker_id", "duration", "language"],
                "example_csv": "file_name,sentence,speaker_id\naudio_001.wav,\"Hello world\",spk_001"
            },
            "audiofolder": {
                "description": "Hugging Face AudioFolder format",
                "structure": {
                    "train/": "Directory with audio files",
                    "train/audio_*.wav": "Audio files with automatic transcription discovery",
                    "metadata.jsonl": "Optional JSONL metadata file"
                },
                "note": "Less commonly used for ASR, CSV format recommended"
            }
        },
        "supported_audio_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"],
        "recommended_settings": {
            "sampling_rate": "16000 Hz",
            "duration_range": "1-30 seconds per clip",
            "quality": "High quality, minimal background noise",
            "format": "WAV format preferred for best compatibility"
        },
        "dataset_size_recommendations": {
            "minimum": "100+ samples for basic training",
            "good": "1000+ samples for decent performance", 
            "excellent": "10000+ samples for production quality"
        }
    }


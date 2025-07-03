"""
Gradio UI for No-Code AI Platform Backend
This UI connects to the FastAPI backend and provides a user interface to interact with the ML pipelines.
"""

import os
import gradio as gr
import requests
import json
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Union

# Configuration
API_URL = "http://ec2-67-202-20-17.compute-1.amazonaws.com:8000"  # FastAPI backend URL

# Helper functions
def get_api_status():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            return f"API Status: Healthy. Active jobs: {data.get('active_jobs', 0)}"
        else:
            return f"API Status: Error ({response.status_code})"
    except Exception as e:
        return f"API Status: Error - {str(e)}"

def list_all_jobs():
    """List all training jobs from the API"""
    try:
        response = requests.get(f"{API_URL}/pipelines")
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                return "No jobs found"
            
            # Format jobs into a nice table
            result = "| ID | Name | Status | Created | Architecture |\n"
            result += "|-----|------|--------|---------|-------------|\n"
            
            for job in jobs:
                job_id = job.get('id', 'N/A')
                name = job.get('pipeline_config', {}).get('name', 'N/A')
                status = job.get('status', 'N/A')
                created = job.get('created_at', 'N/A').split('T')[0]  # Just the date part
                arch = job.get('pipeline_config', {}).get('architecture', 'N/A')
                
                result += f"| {job_id} | {name} | {status} | {created} | {arch} |\n"
            
            return result
        else:
            return f"Error fetching jobs: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def start_mlflow():
    """Start MLflow server through the API"""
    try:
        response = requests.post(f"{API_URL}/mlflow/start-server")
        if response.status_code == 200:
            data = response.json()
            
            # Get the MLflow UI URL
            mlflow_url_response = requests.get(f"{API_URL}/mlflow/ui-url")
            if mlflow_url_response.status_code == 200:
                mlflow_url = mlflow_url_response.json().get('url', 'Unknown')
                return f"MLflow started successfully. UI available at: {mlflow_url}"
            
            return f"MLflow started: {data.get('message', 'Success')}"
        else:
            return f"Error starting MLflow: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def create_pipeline(name, task_type, architecture, num_classes, batch_size, epochs, learning_rate, 
                   image_size, augmentation_enabled, early_stopping):
    """Create a new computer vision training pipeline"""
    try:
        # Base configuration for CV tasks only
        pipeline_config = {
            "name": name,
            "task_type": task_type,
            "architecture": architecture,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "early_stopping": early_stopping,
            "feature_extraction_only": False,
            "patience": 3,
            "num_classes": int(num_classes),
            "image_size": [int(x.strip()) for x in image_size.split(',')],
            "augmentation_enabled": augmentation_enabled
        }
        
        response = requests.post(
            f"{API_URL}/pipelines",
            json=pipeline_config
        )
        
        if response.status_code == 200:
            job = response.json()
            job_id = job.get('id', 'Unknown')
            return f"Pipeline created successfully! Job ID: {job_id}"
        else:
            return f"Error creating pipeline: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_job_status(job_id):
    """Get status of a specific job"""
    if not job_id:
        return "Error: Please provide a Job ID"
    
    try:
        response = requests.get(f"{API_URL}/pipelines/{job_id}")
        if response.status_code == 200:
            job = response.json()
            
            # Format the job details
            config = job.get('pipeline_config', {})
            result = f"## Job: {job_id}\n\n"
            result += f"**Name**: {config.get('name', 'N/A')}\n"
            result += f"**Status**: {job.get('status', 'N/A')}\n"
            result += f"**Task**: {config.get('task_type', 'N/A')}\n" 
            result += f"**Model**: {config.get('architecture', 'N/A')}\n"
            result += f"**Created**: {job.get('created_at', 'N/A')}\n"
            
            if job.get('started_at'):
                result += f"**Started**: {job.get('started_at')}\n"
                
            if job.get('completed_at'):
                result += f"**Completed**: {job.get('completed_at')}\n"
                
            if job.get('metrics'):
                result += f"\n### Metrics\n"
                for metric, value in job.get('metrics', {}).items():
                    result += f"- **{metric}**: {value}\n"
            
            if job.get('model_path'):
                result += f"\n**Model Path**: {job.get('model_path')}\n"
            
            # Get MLflow info if available
            try:
                mlflow_response = requests.get(f"{API_URL}/jobs/{job_id}/mlflow")
                if mlflow_response.status_code == 200:
                    mlflow_data = mlflow_response.json()
                    if 'mlflow_ui_url' in mlflow_data:
                        result += f"\n**MLflow UI URL**: {mlflow_data['mlflow_ui_url']}\n"
            except Exception:
                pass
                
            return result
        else:
            return f"Error fetching job: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def start_training(job_id):
    """Start training for a specific job ID"""
    if not job_id:
        return "Error: Please provide a Job ID"
    
    try:
        response = requests.post(f"{API_URL}/pipelines/{job_id}/train")
        if response.status_code == 200:
            data = response.json()
            return f"Training started for job: {data.get('job_id', job_id)}"
        else:
            return f"Error starting training: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def upload_dataset(job_id, class_name, file_list):
    """Upload dataset files for a job"""
    if not job_id or not class_name:
        return "Error: Please provide Job ID and Class Name"
    
    if not file_list:
        return "Error: No files selected"
    
    try:
        # First create the class directory
        create_class_response = requests.post(f"{API_URL}/upload-dataset/{job_id}/{class_name}")
        if create_class_response.status_code != 200:
            return f"Error creating class directory: {create_class_response.status_code}"
        
        # Upload each file
        success_count = 0
        for file_path in file_list:
            with open(file_path, "rb") as f:
                # Create form data
                files = {"file": (os.path.basename(file_path), f, "image/jpeg")}
                data = {"class_name": class_name}
                
                # Upload the file
                response = requests.post(
                    f"{API_URL}/upload-dataset/{job_id}",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    success_count += 1
        
        return f"Successfully uploaded {success_count} of {len(file_list)} files to class '{class_name}' for job {job_id}"
    except Exception as e:
        return f"Error: {str(e)}"

def make_prediction(job_id, image):
    """Make prediction using a trained model"""
    if not job_id:
        return "Error: Please provide a Job ID", None
    
    if image is None:
        return "Error: No image provided", None
    
    try:
        # Convert image to bytes for upload
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create form data
        files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
        
        # Make prediction request
        response = requests.post(
            f"{API_URL}/predict/{job_id}",
            files=files
        )
        
        if response.status_code == 200:
            prediction = response.json()
            
            # Handle different task types
            task_type = prediction.get("task_type", "")
            processing_time = prediction.get("processing_time", 0)
            
            # Extract annotated image if available
            annotated_image = None
            if "annotated_image" in prediction and prediction["annotated_image"]:
                try:
                    img_bytes = base64.b64decode(prediction["annotated_image"])
                    annotated_image = Image.open(io.BytesIO(img_bytes))
                except Exception as e:
                    annotated_image = image  # fallback to original if decoding fails
            else:
                annotated_image = image  # fallback to original
            
            if "image_classification" in task_type:
                # For classification
                if "top_prediction" in prediction and prediction["top_prediction"]:
                    top_pred = prediction["top_prediction"]
                    class_name = top_pred.get("class_name", "Unknown")
                    confidence = top_pred.get("confidence", 0)
                    result_text = f"## Prediction Results\n\n"
                    result_text += f"**Class**: {class_name}\n"
                    result_text += f"**Confidence**: {confidence:.2f}%\n"
                    result_text += f"**Processing Time**: {processing_time:.3f}s\n"
                    
                    # Show all predictions if available
                    if "predictions" in prediction and len(prediction["predictions"]) > 1:
                        result_text += "\n\n### All Predictions:\n"
                        for pred in prediction["predictions"][:5]:  # Show top 5
                            result_text += f"- {pred['class_name']}: {pred['confidence']:.2f}%\n"
                else:
                    result_text = "Prediction completed but no results returned"
                return result_text, annotated_image
            
            elif "object_detection" in task_type:
                # For object detection - use backend-generated annotated image
                result = "## Object Detection Results\n\n"
                detections = prediction.get("detections", [])
                result += f"**Objects Found**: {len(detections)}\n"
                result += f"**Processing Time**: {processing_time:.3f}s\n\n"
                
                if detections:
                    result += "### Detected Objects:\n"
                    for i, detection in enumerate(detections, 1):
                        class_name = detection.get("class_name", "Unknown")
                        conf = detection.get("confidence", 0)
                        box = detection.get("box", [0, 0, 0, 0])
                        result += f"{i}. **{class_name}** (Confidence: {conf:.1f}%)\n"
                        result += f"   Location: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]\n\n"
                else:
                    result += "No objects detected.\n"

                # Return the backend-generated annotated image
                return result, annotated_image
            
            else:
                # Generic case
                result = f"## Prediction Results\n\n"
                result += f"**Task Type**: {task_type}\n"
                result += f"**Processing Time**: {processing_time:.3f}s\n\n"
                result += f"Prediction completed successfully."
                return result, annotated_image
        else:
            return f"Error making prediction: {response.status_code}, {response.text}", None
    except Exception as e:
        return f"Error: {str(e)}", None

def delete_job(job_id):
    """Delete a job"""
    if not job_id:
        return "Error: Please provide a Job ID"
    
    try:
        response = requests.delete(f"{API_URL}/pipelines/{job_id}")
        if response.status_code == 200:
            return f"Job {job_id} deleted successfully"
        else:
            return f"Error deleting job: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def list_available_datasets():
    """List all available datasets from the datasets directory"""
    try:
        response = requests.get(f"{API_URL}/datasets/available")
        if response.status_code == 200:
            datasets = response.json()
            if not datasets:
                return "No datasets found", []
            
            # Format datasets into a nice table
            result = "| ID | Name | Classes | Task Type | Items | Format |\n"
            result += "|-----|------|---------|-----------|-------|--------|\n"
            
            dataset_list = []
            
            for dataset in datasets:
                dataset_id = dataset.get('id', 'N/A')
                name = dataset.get('name', dataset_id)
                classes = len(dataset.get('classes', []))
                task = dataset.get('task_type', 'Unknown')
                item_count = dataset.get('item_count', 0)
                is_coco = "COCO" if dataset.get('is_coco_format', False) else "Standard"
                
                result += f"| {dataset_id} | {name} | {classes} | {task} | {item_count} | {is_coco} |\n"
                dataset_list.append(dataset_id)
            
            return result, dataset_list
        else:
            return f"Error fetching datasets: {response.status_code}", []
    except Exception as e:
        return f"Error: {str(e)}", []

def upload_dataset_folder(job_id, dataset_folder):
    """Upload dataset folder for a job"""
    if not job_id or not dataset_folder:
        return "Error: Please provide Job ID and Dataset Folder"
    
    try:
        # Check if it's a valid folder
        if not os.path.isdir(dataset_folder):
            return f"Error: {dataset_folder} is not a valid directory"
        
        # Get list of classes (subfolders)
        classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
        if not classes:
            return "Error: No class folders found in the dataset directory"
        
        success_count = 0
        total_files = 0
        
        # Process each class folder
        for class_name in classes:
            class_path = os.path.join(dataset_folder, class_name)
            
            # Create the class directory on the server
            create_class_response = requests.post(f"{API_URL}/upload-dataset/{job_id}/{class_name}")
            if create_class_response.status_code != 200:
                return f"Error creating class directory for {class_name}: {create_class_response.status_code}"
            
            # Get image files in the class folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_files = [f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) and 
                          any(f.lower().endswith(ext) for ext in image_extensions)]
            
            total_files += len(image_files)
            
            # Upload each file
            for img_file in image_files:
                file_path = os.path.join(class_path, img_file)
                with open(file_path, "rb") as f:
                    # Create form data
                    files = {"file": (img_file, f, "image/jpeg")}
                    data = {"class_name": class_name}
                    
                    # Upload the file
                    response = requests.post(
                        f"{API_URL}/upload-dataset/{job_id}",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
        
        return f"Successfully uploaded {success_count} of {total_files} files from {len(classes)} classes for job {job_id}"
    except Exception as e:
        return f"Error: {str(e)}"

def upload_detection_dataset(job_id, zip_file):
    """Upload object detection dataset as a zip file (COCO format)"""
    if not job_id:
        return "Error: Please provide Job ID"
    
    if not zip_file:
        return "Error: No zip file selected"
    
    try:
        # Verify file is a zip file
        file_path = zip_file
        if not file_path.lower().endswith('.zip'):
            return "Error: File must be a ZIP archive (.zip)"
        
        # Upload the zip file using the dedicated endpoint for object detection datasets
        with open(file_path, "rb") as f:
            # Create form data - use the specific endpoint for object detection datasets
            files = {"file": (os.path.basename(file_path), f, "application/zip")}
            
            # Upload the file
            response = requests.post(
                f"{API_URL}/upload-detection-dataset/{job_id}",
                files=files
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Automatically link the dataset to the job
                link_response = requests.post(f"{API_URL}/pipelines/{job_id}/dataset/{job_id}")
                
                result = f"Successfully uploaded object detection dataset: {data.get('message', '')}"
                
                if link_response.status_code == 200:
                    result += "\nDataset has been automatically linked to the job."
                    result += "\nYou can now proceed with training."
                else:
                    result += f"\nWarning: Failed to automatically link dataset to job: {link_response.status_code}"
                    result += "\nYou may need to manually link the dataset before training."
                
                # Suggest refreshing the datasets list
                result += "\n\nPlease refresh the datasets list to see your uploaded dataset."
                
                return result
            else:
                return f"Error uploading dataset: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def list_jobs_for_dropdown():
    """Get jobs formatted for dropdown selection"""
    try:
        response = requests.get(f"{API_URL}/pipelines")
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                return []
            
            # Format as "Name - ID (Status)" for easy selection
            choices = []
            for job in jobs:
                job_id = job.get('id', 'N/A')
                name = job.get('pipeline_config', {}).get('name', 'N/A')
                status = job.get('status', 'N/A')
                task = job.get('pipeline_config', {}).get('task_type', 'N/A')
                choices.append((f"{name} - {job_id} ({task}, {status})", job_id))
            
            return choices
        else:
            return []
    except Exception as e:
        return []

def link_dataset(job_id, dataset_id):
    """Link an existing dataset to a job"""
    if not job_id or not dataset_id:
        return "Error: Please provide both Job ID and Dataset ID"
    
    try:                
        # Verify the dataset exists in the API (optional, can be removed)
        datasets_response = requests.get(f"{API_URL}/datasets/available")
        if datasets_response.status_code != 200:
            return f"Error verifying dataset: {datasets_response.status_code}"
        
        datasets = datasets_response.json()
        dataset_exists = any(d.get('id') == dataset_id for d in datasets)
        
        # Even if dataset is not in the API response, try to link it
        if not dataset_exists:
            print(f"Warning: Dataset {dataset_id} not found in API response, attempting to link anyway.")
        
        # Print available datasets for debugging
        available_ids = [d.get('id') for d in datasets]
        print(f"Available dataset IDs: {available_ids}")
        
        # Link the dataset to job
        response = requests.post(
            f"{API_URL}/pipelines/{job_id}/dataset/{dataset_id}"
        )
        
        if response.status_code == 200:
            return f"Successfully linked dataset {dataset_id} to job {job_id}"
        else:
            error_text = response.text
            if "404" in str(response.status_code):
                error_text += "\n\nThe API could not find the dataset. Try refreshing the datasets list before linking."
            return f"Error linking dataset: {response.status_code}, {error_text}"
    except Exception as e:
        return f"Error: {str(e)}"
    

# Build Gradio Interface
with gr.Blocks(title="No-Code AI Platform") as app:
    gr.Markdown("# No-Code AI Platform")
    gr.Markdown("Connect to the FastAPI backend for machine learning model training and inference")
    
    with gr.Tab("Dashboard"):
        with gr.Row():
            with gr.Column():
                status_btn = gr.Button("Check API Status")
                status_output = gr.Markdown()
                
                mlflow_btn = gr.Button("Start MLflow Server")
                mlflow_output = gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Jobs")
                refresh_jobs_btn = gr.Button("Refresh Jobs List")
                jobs_output = gr.Markdown()
    
    with gr.Tab("Create Pipeline"):
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(label="Pipeline Name", placeholder="My Image Classifier")
                
                # Simplified Computer Vision only
                task_type = gr.Dropdown(
                    label="Task Type",
                    choices=["image_classification", "object_detection"],
                    value="image_classification"
                )
                
                # Architecture selection based on task
                architecture = gr.Dropdown(
                    label="Model Architecture",
                    choices=["resnet18", "resnet50", "vgg16", "mobilenet", "efficientnet"],
                    value="resnet18"
                )
                
                # Function to update architecture choices based on task
                def update_architecture(task):
                    if task == "image_classification":
                        return gr.Dropdown(
                            choices=["resnet18", "resnet50", "vgg16", "mobilenet", "efficientnet"],
                            value="resnet18"
                        )
                    elif task == "object_detection":
                        return gr.Dropdown(
                            choices=["faster_rcnn"],
                            value="faster_rcnn"
                        )
                
                task_type.change(update_architecture, inputs=task_type, outputs=architecture)
                
                num_classes = gr.Number(label="Number of Classes", value=2, precision=0)
                batch_size = gr.Number(label="Batch Size", value=8, precision=0)
                epochs = gr.Number(label="Epochs", value=5, precision=0)
                learning_rate = gr.Number(label="Learning Rate", value=0.001)
                image_size = gr.Textbox(label="Image Size (width, height)", value="224, 224")
                
                with gr.Row():
                    augmentation = gr.Checkbox(label="Enable Data Augmentation", value=True)
                    early_stopping = gr.Checkbox(label="Enable Early Stopping", value=True)
                
                create_btn = gr.Button("Create Pipeline")
                create_output = gr.Markdown()
    
    with gr.Tab("Train Model"):
        # Current Job Status Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Current Job Status")
                gr.Markdown("*Shows the most recently created job (ready for training)*")
                
                refresh_current_job_btn = gr.Button("Refresh Current Job")
                current_job_display = gr.Markdown("No job found. Create a pipeline first.")
                
                def get_current_job():
                    """Get the most recent job as the current working job (for training)"""
                    try:
                        response = requests.get(f"{API_URL}/pipelines")
                        if response.status_code == 200:
                            jobs = response.json()
                            if not jobs:
                                return "No jobs found. Create a pipeline first.", None
                            
                            # Find the most recent job that's ready for training (not completed)
                            current_job = None
                            for job in jobs:
                                status = job.get('status', '').lower()
                                # Look for jobs that are ready for training
                                if status in ['pending', 'created', 'ready', 'initialized']:
                                    current_job = job
                                    break
                            
                            # If no pending job found, show the most recent one but with a warning
                            if not current_job:
                                current_job = jobs[0]
                                return "⚠️ **No new jobs found ready for training.**\n\nMost recent job shown below, but it may already be trained.\n\n**Please create a new pipeline to start fresh training.**", None
                            
                            job_id = current_job.get('id', 'N/A')
                            name = current_job.get('pipeline_config', {}).get('name', 'N/A')
                            status = current_job.get('status', 'N/A')
                            task = current_job.get('pipeline_config', {}).get('task_type', 'N/A')
                            arch = current_job.get('pipeline_config', {}).get('architecture', 'N/A')
                            
                            # Add status indicator for new jobs
                            if status.lower() in ['pending', 'created', 'ready', 'initialized']:
                                status_icon = "🆕"
                                status_text = "Ready for training"
                                help_text = "\n💡 *This job is ready for dataset upload and training*"
                            elif status.lower() == "training":
                                status_icon = "⏳"
                                status_text = "Currently training"
                                help_text = "\n⏳ *Training in progress...*"
                            elif status.lower() == "completed":
                                status_icon = "✅"
                                status_text = "Training completed"
                                help_text = "\n✅ *Training completed - available for inference*\n\n⚠️ **This job is already trained. Create a new pipeline for fresh training.**"
                            else:
                                status_icon = "❓"
                                status_text = status
                                help_text = f"\n❓ *Status: {status}*"
                            
                            result = f"## {status_icon} Current Job: {name}\n\n"
                            result += f"**ID**: {job_id}\n"
                            result += f"**Status**: {status_text}\n"
                            result += f"**Task**: {task}\n"
                            result += f"**Architecture**: {arch}\n"
                            result += help_text
                            
                            return result, job_id
                        else:
                            return f"Error fetching jobs: {response.status_code}", None
                    except Exception as e:
                        return f"Error: {str(e)}", None
        
        # Upload Options for Classification Tasks
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 Classification Upload Options")
                gr.Markdown("*For image classification tasks - upload files or folders to the newly created job*")
                
                with gr.Tabs():
                    with gr.Tab("Upload Files"):
                        class_name = gr.Textbox(label="Class Name", placeholder="class1")
                        file_upload = gr.File(label="Upload Images", file_count="multiple")
                        upload_btn = gr.Button("Upload Files to New Job")
                    
                    with gr.Tab("Upload Folder"):
                        dataset_folder = gr.Textbox(label="Dataset Folder Path", placeholder="/path/to/dataset")
                        folder_upload_btn = gr.Button("Upload Folder to New Job")
                
                classification_output = gr.Markdown()
        
        # Upload Options for Object Detection Tasks
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎯 Object Detection Upload Options")
                gr.Markdown("*For object detection tasks - upload COCO format zip files to the newly created job*")
                
                detection_zip_upload = gr.File(label="Upload COCO Zip File", file_count="single", file_types=[".zip"])
                detection_upload_btn = gr.Button("Upload Detection Dataset to New Job")
                detection_output = gr.Markdown()
        
        # Link Existing Dataset Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔗 Link Existing Dataset")
                gr.Markdown("*Connect a pending job to an existing dataset*")
                
                # Dropdown for pending jobs
                pending_jobs_dropdown = gr.Dropdown(
                    label="Select Pending Job",
                    choices=[],
                    value=None
                )
                
                refresh_pending_jobs_btn = gr.Button("Refresh Pending Jobs")
                
                # Dropdown for datasets
                refresh_datasets_btn = gr.Button("Refresh Available Datasets")
                datasets_list_output = gr.Markdown()
                
                dataset_dropdown = gr.Dropdown(
                    label="Select Dataset",
                    choices=[],
                    value=None
                )
                
                def update_pending_jobs_dropdown():
                    """Update pending jobs dropdown with jobs that need dataset linking"""
                    try:
                        response = requests.get(f"{API_URL}/pipelines")
                        if response.status_code == 200:
                            jobs = response.json()
                            if not jobs:
                                return gr.Dropdown(choices=[], value=None)
                            
                            # Filter for pending jobs only
                            pending_jobs = []
                            for job in jobs:
                                status = job.get('status', '').lower()
                                if status == 'pending':
                                    job_id = job.get('id', 'N/A')
                                    name = job.get('pipeline_config', {}).get('name', f'Job {job_id}')
                                    task = job.get('pipeline_config', {}).get('task_type', 'Unknown')
                                    arch = job.get('pipeline_config', {}).get('architecture', 'Unknown')
                                    pending_jobs.append((f"⏳ {name} - {job_id} ({task}, {arch})", job_id))
                            
                            return gr.Dropdown(choices=pending_jobs, value=None)
                        else:
                            return gr.Dropdown(choices=[], value=None)
                    except Exception as e:
                        return gr.Dropdown(choices=[], value=None)
                
                def update_dataset_dropdown():
                    """Update dataset dropdown with available datasets"""
                    try:
                        response = requests.get(f"{API_URL}/datasets/available")
                        if response.status_code == 200:
                            datasets = response.json()
                            if not datasets:
                                return gr.Dropdown(choices=[], value=None)
                            
                            # Format datasets for dropdown: "Name - ID (Task, Format)"
                            choices = []
                            for dataset in datasets:
                                dataset_id = dataset.get('id', 'N/A')
                                name = dataset.get('name', dataset_id)
                                task = dataset.get('task_type', 'Unknown')
                                is_coco = "COCO" if dataset.get('is_coco_format', False) else "Standard"
                                choices.append((f"{name} - {dataset_id} ({task}, {is_coco})", dataset_id))
                            
                            return gr.Dropdown(choices=choices, value=None)
                        else:
                            return gr.Dropdown(choices=[], value=None)
                    except Exception as e:
                        return gr.Dropdown(choices=[], value=None)
                
                def display_datasets():
                    """Display available datasets in a table format"""
                    result, _ = list_available_datasets()
                    return result
                
                link_dataset_btn = gr.Button("Link Dataset to Selected Job")
                link_dataset_output = gr.Markdown()
        
        # Start Training Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🚀 Start Training")
                gr.Markdown("*Start training jobs that have datasets linked*")
                
                # Dropdown for jobs ready for training
                trainable_jobs_dropdown = gr.Dropdown(
                    label="Select Job Ready for Training",
                    choices=[],
                    value=None
                )
                
                refresh_trainable_jobs_btn = gr.Button("Refresh Trainable Jobs")
                
                def update_trainable_jobs_dropdown():
                    """Update trainable jobs dropdown with jobs that have datasets linked"""
                    try:
                        response = requests.get(f"{API_URL}/pipelines")
                        if response.status_code == 200:
                            jobs = response.json()
                            if not jobs:
                                return gr.Dropdown(choices=[], value=None)
                            
                            # Filter for jobs that are ready for training (not completed, not currently training)
                            trainable_jobs = []
                            for job in jobs:
                                status = job.get('status', '').lower()
                                # Include jobs that are not completed or training (broader criteria)
                                # This includes: pending, created, ready, initialized, with_dataset, etc.
                                if status not in ['completed', 'training', 'failed']:
                                    job_id = job.get('id', 'N/A')
                                    name = job.get('pipeline_config', {}).get('name', f'Job {job_id}')
                                    task = job.get('pipeline_config', {}).get('task_type', 'Unknown')
                                    arch = job.get('pipeline_config', {}).get('architecture', 'Unknown')
                                    
                                    # Add more specific status indicators
                                    if status in ['ready', 'initialized', 'with_dataset']:
                                        status_icon = "✅"
                                    elif status == 'pending':
                                        status_icon = "⏳"
                                    else:
                                        status_icon = "🆕"
                                    
                                    trainable_jobs.append((f"{status_icon} {name} - {job_id} ({task}, {arch}, {status})", job_id))
                            
                            return gr.Dropdown(choices=trainable_jobs, value=None)
                        else:
                            return gr.Dropdown(choices=[], value=None)
                    except Exception as e:
                        return gr.Dropdown(choices=[], value=None)
                
                train_btn = gr.Button("Start Training Selected Job")
                train_output = gr.Markdown()
    
    with gr.Tab("Make Predictions"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔮 Model Selection for Inference")
                predict_job_dropdown = gr.Dropdown(
                    label="Select Trained Model",
                    choices=[],
                    value=None
                )
                refresh_models_btn = gr.Button("Refresh Available Models")
                
                def update_prediction_models():
                    """Update dropdown with available trained models"""
                    try:
                        response = requests.get(f"{API_URL}/pipelines")
                        if response.status_code == 200:
                            jobs = response.json()
                            if not jobs:
                                return gr.Dropdown(choices=[], value=None)
                            
                            # Filter for completed jobs only
                            choices = []
                            for job in jobs:
                                status = job.get('status', '').lower()
                                if status == 'completed':
                                    job_id = job.get('id', 'N/A')
                                    name = job.get('pipeline_config', {}).get('name', 'N/A')
                                    task = job.get('pipeline_config', {}).get('task_type', 'N/A')
                                    arch = job.get('pipeline_config', {}).get('architecture', 'N/A')
                                    choices.append((f"✅ {name} - {job_id} ({task}, {arch})", job_id))
                            
                            return gr.Dropdown(choices=choices, value=None)
                        else:
                            return gr.Dropdown(choices=[], value=None)
                    except Exception as e:
                        return gr.Dropdown(choices=[], value=None)
                
                input_image = gr.Image(label="Upload Image for Prediction", type="pil")
                predict_btn = gr.Button("Make Prediction")
            
            with gr.Column():
                predict_output = gr.Markdown()
                annotated_image_output = gr.Image(label="Prediction Result", visible=True)
    
    with gr.Tab("Delete Job"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🗑️ Delete Job")
                delete_job_dropdown = gr.Dropdown(
                    label="Select Job to Delete",
                    choices=[],
                    value=None
                )
                refresh_delete_dropdown_btn = gr.Button("Refresh Jobs List")
                
                def update_delete_dropdown():
                    """Update dropdown with all available jobs for deletion"""
                    try:
                        response = requests.get(f"{API_URL}/pipelines")
                        if response.status_code == 200:
                            jobs = response.json()
                            if not jobs:
                                return gr.Dropdown(choices=[], value=None)
                            
                            choices = []
                            for job in jobs:
                                job_id = job.get('id', 'N/A')
                                name = job.get('pipeline_config', {}).get('name', 'N/A')
                                status = job.get('status', 'N/A').lower()
                                task = job.get('pipeline_config', {}).get('task_type', 'N/A')
                                
                                # Add status indicators
                                if status in ['pending', 'created', 'ready', 'initialized']:
                                    status_mark = "🆕"
                                elif status == "training":
                                    status_mark = "⏳"
                                elif status == "completed":
                                    status_mark = "✅"
                                else:
                                    status_mark = "❓"
                                
                                choices.append((f"{status_mark} {name} - {job_id} ({task}, {status})", job_id))
                            
                            return gr.Dropdown(choices=choices, value=None)
                        else:
                            return gr.Dropdown(choices=[], value=None)
                    except Exception as e:
                        return gr.Dropdown(choices=[], value=None)
                
                delete_btn = gr.Button("Delete Job")
                delete_output = gr.Markdown()
    
    # Helper function to get current job ID (newly created job)
    def get_current_job_id():
        """Get the ID of the most recently created job that's ready for training (not already trained)"""
        try:
            response = requests.get(f"{API_URL}/pipelines")
            if response.status_code == 200:
                jobs = response.json()
                if jobs:
                    # Find the most recent job that's NOT completed (i.e., newly created and ready for training)
                    for job in jobs:
                        status = job.get('status', '').lower()
                        # Look for jobs that are pending, created, or ready - NOT completed or training
                        if status in ['pending', 'created', 'ready', 'initialized']:
                            return job.get('id')
                    
                    # If no pending jobs found, get the most recent one as fallback
                    return jobs[0].get('id')
            return None
        except:
            return None
    
    # Helper functions for new job operations
    def upload_to_new_job(class_name, file_list):
        current_job_id = get_current_job_id()
        if not current_job_id:
            return "❌ **No new job found ready for training.**\n\nPlease create a new pipeline first, then upload your dataset."
        return upload_dataset(current_job_id, class_name, file_list)
    
    def upload_folder_to_new_job(dataset_folder):
        current_job_id = get_current_job_id()
        if not current_job_id:
            return "❌ **No new job found ready for training.**\n\nPlease create a new pipeline first, then upload your dataset."
        return upload_dataset_folder(current_job_id, dataset_folder)
    
    def upload_detection_to_new_job(zip_file):
        current_job_id = get_current_job_id()
        if not current_job_id:
            return "❌ **No new job found ready for training.**\n\nPlease create a new pipeline first, then upload your dataset."
        return upload_detection_dataset(current_job_id, zip_file)
    
    def link_dataset_to_selected_job(pending_job_id, dataset_id):
        """Link dataset to the selected pending job"""
        if not pending_job_id:
            return "❌ **No pending job selected.**\n\nPlease select a pending job from the dropdown first."
        if not dataset_id:
            return "❌ **No dataset selected.**\n\nPlease select a dataset from the dropdown first."
        return link_dataset(pending_job_id, dataset_id)
    
    def train_selected_job(trainable_job_id):
        """Train the selected job"""
        if not trainable_job_id:
            return "❌ **No trainable job selected.**\n\nPlease select a job that's ready for training from the dropdown first."
        return start_training(trainable_job_id)
    
    # Connect buttons to functions
    status_btn.click(get_api_status, outputs=status_output)
    mlflow_btn.click(start_mlflow, outputs=mlflow_output)
    refresh_jobs_btn.click(list_all_jobs, outputs=jobs_output)
    
    create_btn.click(
        create_pipeline,
        inputs=[name_input, task_type, architecture, num_classes, batch_size, epochs, learning_rate, image_size, 
                augmentation, early_stopping],
        outputs=create_output
    )
    
    # Connect current job management
    refresh_current_job_btn.click(
        lambda: get_current_job()[0],  # Just return the display text
        outputs=current_job_display
    )
    
    # Connect dataset management buttons
    refresh_datasets_btn.click(
        fn=lambda: [display_datasets(), update_dataset_dropdown()],
        outputs=[datasets_list_output, dataset_dropdown]
    )
    
    # Connect upload buttons for new job
    upload_btn.click(
        upload_to_new_job, 
        inputs=[class_name, file_upload],
        outputs=classification_output
    )
    
    folder_upload_btn.click(
        upload_folder_to_new_job, 
        inputs=[dataset_folder],
        outputs=classification_output
    )
    
    # Connect object detection upload button for new job
    detection_upload_btn.click(
        upload_detection_to_new_job, 
        inputs=[detection_zip_upload],
        outputs=detection_output
    )
    
    # Connect link dataset button for selected job
    link_dataset_btn.click(
        link_dataset_to_selected_job,
        inputs=[pending_jobs_dropdown, dataset_dropdown],
        outputs=link_dataset_output
    )
    
    # Connect training button for selected job
    train_btn.click(
        train_selected_job, 
        inputs=[trainable_jobs_dropdown],
        outputs=train_output
    )
    
    # Connect pending jobs dropdown refresh
    refresh_pending_jobs_btn.click(update_pending_jobs_dropdown, outputs=pending_jobs_dropdown)
    
    # Connect trainable jobs dropdown refresh
    refresh_trainable_jobs_btn.click(update_trainable_jobs_dropdown, outputs=trainable_jobs_dropdown)
    
    # Connect prediction model refresh
    refresh_models_btn.click(update_prediction_models, outputs=predict_job_dropdown)
    
    # Connect delete dropdown refresh
    refresh_delete_dropdown_btn.click(update_delete_dropdown, outputs=delete_job_dropdown)
    
    # Connect prediction button
    predict_btn.click(
        make_prediction,
        inputs=[predict_job_dropdown, input_image],
        outputs=[predict_output, annotated_image_output]
    )
    
    # Connect delete button
    delete_btn.click(delete_job, inputs=delete_job_dropdown, outputs=delete_output)

# Launch the app
if __name__ == "__main__":
    print("Starting Simplified No-Code AI Platform - Computer Vision Only")
    app.launch(server_name="0.0.0.0")
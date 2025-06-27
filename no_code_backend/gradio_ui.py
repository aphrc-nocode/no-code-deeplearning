"""
Gradio UI for No-Code AI Platform Backend
This UI connects to the FastAPI backend and provides a user interface to interact with the ML pipelines.
"""

import os
import gradio as gr
import requests
import json
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional, Union

# Configuration
API_URL = "http://localhost:8000"  # FastAPI backend URL

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

def create_pipeline(name, task_type, 
                   arch_classification, arch_detection, arch_segmentation,
                   num_classes, batch_size, epochs,
                   learning_rate, image_size, augmentation_enabled, early_stopping):
    """Create a new training pipeline"""
    try:
        # Select the appropriate architecture based on task type
        if task_type == "image_classification":
            architecture = arch_classification
        elif task_type == "object_detection":
            architecture = arch_detection
        elif task_type == "image_segmentation":
            architecture = arch_segmentation
        else:
            architecture = arch_classification  # Default
        
        pipeline_config = {
            "name": name,
            "task_type": task_type,
            "architecture": architecture,
            "num_classes": int(num_classes),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "image_size": [int(x.strip()) for x in image_size.split(',')],
            "augmentation_enabled": augmentation_enabled,
            "early_stopping": early_stopping,
            "feature_extraction_only": False,
            "patience": 3
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
        return "Error: Please provide a Job ID"
    
    if image is None:
        return "Error: No image provided"
    
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
            
            if "image_classification" in task_type:
                # For classification
                class_name = prediction.get("class_name", "Unknown")
                confidence = prediction.get("confidence", 0)
                return f"Prediction: {class_name} (Confidence: {confidence:.2f})"
            
            elif "object_detection" in task_type:
                # For object detection
                result = "## Object Detection Results\n\n"
                for detection in prediction.get("detections", []):
                    label = detection.get("label", "Unknown")
                    conf = detection.get("confidence", 0)
                    box = detection.get("box", [0, 0, 0, 0])
                    result += f"- **{label}** (Confidence: {conf:.2f}), Box: {box}\n"
                return result
            
            elif "image_segmentation" in task_type:
                # For segmentation tasks
                if "mask_base64" in prediction:
                    try:
                        # Display both original image and segmentation mask
                        mask_data = base64.b64decode(prediction.get("mask_base64"))
                        mask_image = Image.open(io.BytesIO(mask_data))
                        
                        # Make mask_output visible
                        gr.update(visible=True)
                        
                        return [
                            f"Segmentation completed. {len(prediction.get('class_ids', []))} segments found.",
                            mask_image
                        ]
                    except Exception as e:
                        return [f"Error processing segmentation mask: {str(e)}", None]
                else:
                    return ["Segmentation completed, but no mask returned.", None]
            
            else:
                # Generic case
                return f"Prediction successful: {prediction}"
        else:
            return f"Error making prediction: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

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
                
                # High-level category selection
                category_options = ["Computer Vision", "NLP"]
                high_level_category = gr.Dropdown(
                    label="Category",
                    choices=category_options,
                    value="Computer Vision"
                )
                
                # Define tasks per high-level category
                TASK_OPTIONS = {
                    "Computer Vision": ["image_classification", "object_detection", "image_segmentation"],
                    "NLP": ["text_classification", "named_entity_recognition", "question_answering"]
                }
                
                # Define architecture options per task type
                ARCHITECTURE_OPTIONS = {
                    "image_classification": ["resnet18", "resnet50", "vgg16", "mobilenet", "efficientnet"],
                    "object_detection": [
                        "faster_rcnn", "ssd", "yolo", "retinanet",
                        "detr_resnet50", "detr_resnet101", 
                        "yolos_small", "yolos_base", "yolos_tiny",
                        "owlv2_base"
                    ],
                    "image_segmentation": ["fcn", "deeplabv3", "mask_rcnn", "unet"],
                    "text_classification": ["bert", "roberta", "distilbert"],
                    "named_entity_recognition": ["bert", "roberta", "distilbert"],
                    "question_answering": ["bert", "roberta", "distilbert"]
                }
                
                # Initial task type dropdown with only computer vision tasks
                task_type = gr.Dropdown(
                    label="Task Type",
                    choices=TASK_OPTIONS["Computer Vision"],
                    value="image_classification"
                )
                
                # Function to update tasks based on high-level category
                def on_category_change(category):
                    tasks = TASK_OPTIONS.get(category, [])
                    default_task = tasks[0] if tasks else None
                    return gr.Dropdown.update(choices=tasks, value=default_task)
                
                # Connect high-level category to task type dropdown
                high_level_category.change(
                    fn=on_category_change,
                    inputs=high_level_category,
                    outputs=task_type
                )
                
                # Initial architecture dropdowns for each task type (only one will be visible)
                architecture_classification = gr.Dropdown(
                    label="Model Architecture",
                    choices=ARCHITECTURE_OPTIONS["image_classification"],
                    value="resnet18",
                    interactive=True,
                    visible=True
                )
                
                architecture_detection = gr.Dropdown(
                    label="Model Architecture",
                    choices=ARCHITECTURE_OPTIONS["object_detection"],
                    value="faster_rcnn",
                    interactive=True,
                    visible=False
                )
                
                architecture_segmentation = gr.Dropdown(
                    label="Model Architecture",
                    choices=ARCHITECTURE_OPTIONS["image_segmentation"],
                    value="fcn",
                    interactive=True,
                    visible=False
                )
                
                # Map task types to their corresponding dropdowns
                architecture_dropdowns = {
                    "image_classification": architecture_classification,
                    "object_detection": architecture_detection,
                    "image_segmentation": architecture_segmentation
                }
                
                # Function to show/hide architecture dropdowns based on task type
                def on_task_type_change(task):
                    result = {}
                    for task_name, dropdown in architecture_dropdowns.items():
                        result[dropdown] = gr.update(visible=(task == task_name))
                    return result
                
                # Connect task type to architecture dropdown visibility using change event
                task_type.change(
                    fn=on_task_type_change,
                    inputs=task_type,
                    outputs=list(architecture_dropdowns.values())
                )
                
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
        with gr.Row():
            with gr.Column():
                job_id_input = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                job_status_btn = gr.Button("Get Job Status")
                job_status_output = gr.Markdown()
        
        # First row for file upload
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Individual Files")
                upload_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                class_name = gr.Textbox(label="Class Name", placeholder="class1")
                file_upload = gr.File(label="Upload Images", file_count="multiple")
                upload_btn = gr.Button("Upload Files")
                upload_output = gr.Markdown()
        
        # Second row for folder upload
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Dataset Folder")
                folder_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                dataset_folder = gr.Textbox(label="Dataset Folder Path", placeholder="/path/to/dataset")
                folder_upload_btn = gr.Button("Upload Folder")
                folder_upload_output = gr.Markdown()
        
        # Add a new section for object detection dataset upload
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload Object Detection Dataset (COCO format)")
                gr.Markdown("Upload a zip file containing images and annotations in COCO format")
                detection_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                detection_zip_upload = gr.File(label="Upload Zip File", file_count="single", file_types=[".zip"])
                detection_upload_btn = gr.Button("Upload Object Detection Dataset")
                detection_upload_output = gr.Markdown()
        
        # Third row for existing datasets
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Select Existing Dataset")
                refresh_datasets_btn = gr.Button("Refresh Datasets List")
                datasets_list_output = gr.Markdown()
                existing_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                dataset_id = gr.Textbox(label="Dataset ID", placeholder="Enter Dataset ID from list")
                link_dataset_btn = gr.Button("Link Dataset to Job")
                link_dataset_output = gr.Markdown()
        
        with gr.Row():
            with gr.Column():
                train_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                train_btn = gr.Button("Start Training")
                train_output = gr.Markdown()
    
    with gr.Tab("Make Predictions"):
        with gr.Row():
            with gr.Column():
                predict_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                input_image = gr.Image(label="Upload Image for Prediction", type="pil")
                predict_btn = gr.Button("Make Prediction")
            
            with gr.Column():
                predict_output = gr.Markdown()
                mask_output = gr.Image(label="Segmentation Mask", visible=False)
    
    with gr.Tab("Delete Job"):
        with gr.Row():
            with gr.Column():
                delete_job_id = gr.Textbox(label="Job ID", placeholder="Enter Job ID")
                delete_btn = gr.Button("Delete Job")
                delete_output = gr.Markdown()
    
    # Connect buttons to functions
    status_btn.click(get_api_status, outputs=status_output)
    mlflow_btn.click(start_mlflow, outputs=mlflow_output)
    refresh_jobs_btn.click(list_all_jobs, outputs=jobs_output)
    
    # Helper function to get the selected architecture based on task type
    def get_selected_architecture(task_type, arch_classification, arch_detection, arch_segmentation):
        if task_type == "image_classification":
            return arch_classification
        elif task_type == "object_detection":
            return arch_detection
        elif task_type == "image_segmentation":
            return arch_segmentation
        return arch_classification  # Default to classification
        
    create_btn.click(
        create_pipeline,
        inputs=[name_input, task_type, 
                architecture_classification, architecture_detection, architecture_segmentation,  # All architecture dropdowns
                num_classes, batch_size, epochs, learning_rate, image_size, 
                augmentation, early_stopping],
        outputs=create_output
    )
    
    job_status_btn.click(get_job_status, inputs=job_id_input, outputs=job_status_output)
    
    upload_btn.click(
        upload_dataset, 
        inputs=[upload_job_id, class_name, file_upload],
        outputs=upload_output
    )
    
    # Connect dataset folder upload button
    folder_upload_btn.click(
        upload_dataset_folder, 
        inputs=[folder_job_id, dataset_folder],
        outputs=folder_upload_output
    )
    
    # Connect object detection dataset upload button
    detection_upload_btn.click(
        upload_detection_dataset, 
        inputs=[detection_job_id, detection_zip_upload],
        outputs=detection_upload_output
    )
    
    # Connect refresh datasets button
    def display_datasets():
        result, _ = list_available_datasets()
        return result
        
    refresh_datasets_btn.click(
        display_datasets,
        outputs=datasets_list_output
    )
    
    # Function to link an existing dataset to a job
    def link_dataset(job_id, dataset_id):
        if not job_id or not dataset_id:
            return "Error: Please provide both Job ID and Dataset ID"
        
        try:
            # First, check if the dataset actually exists on disk
            import os
            dataset_path = os.path.join("datasets", dataset_id)
            if not os.path.exists(dataset_path):
                return f"Error: Dataset directory {dataset_id} not found on disk"
                
            # Verify the dataset exists in the API
            datasets_response = requests.get(f"{API_URL}/datasets/available")
            if datasets_response.status_code != 200:
                return f"Error verifying dataset: {datasets_response.status_code}"
            
            datasets = datasets_response.json()
            dataset_exists = any(d.get('id') == dataset_id for d in datasets)
            
            # Even if dataset is not in the API response but exists on disk, try to link it
            if not dataset_exists:
                print(f"Warning: Dataset {dataset_id} exists on disk but not found in API response")
            
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
                
                # Provide more detailed error information
                if "404" in str(response.status_code):
                    error_text += f"\n\nThe dataset exists at {dataset_path}, but the API could not find it."
                    error_text += "\nTry refreshing the datasets list before linking."
                
                return f"Error linking dataset: {response.status_code}, {error_text}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Connect link dataset button
    link_dataset_btn.click(
        link_dataset,
        inputs=[existing_job_id, dataset_id],
        outputs=link_dataset_output
    )
    
    train_btn.click(start_training, inputs=train_job_id, outputs=train_output)
    
    predict_btn.click(
        make_prediction,
        inputs=[predict_job_id, input_image],
        outputs=[predict_output, mask_output]
    )
    
    delete_btn.click(delete_job, inputs=delete_job_id, outputs=delete_output)

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio UI with task-specific model architecture selection")
    app.launch(server_name="0.0.0.0")
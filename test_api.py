"""
Example script to interact with the NoCode AI Platform API
This demonstrates how to use the API endpoints for testing purposes
Works with both local and dockerized deployments
"""

import requests
import json
import uuid
import time
import os
import shutil
import argparse
from urllib.parse import urlparse
from pathlib import Path
from PIL import Image
import io

# Configure API endpoints based on environment
# Default URLs for different environments
URLS = {
    'dev': 'http://localhost:8000',            # Local development
    'docker': 'http://localhost:8000',         # Docker development
    'prod': 'https://localhost/api'            # Production with Nginx
}

# Default API URL
API_URL = URLS['dev']  # Will be overridden by command-line arguments

# Define constants to match main.py
class TrainingStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

def print_server_logs():
    """Helper function to retrieve and print server logs for troubleshooting"""
    try:
        import subprocess
        print("\nChecking server logs:")
        server_logs = subprocess.check_output("tail -n 20 /tmp/uvicorn_output.log 2>/dev/null || echo 'No server logs found'", shell=True)
        print(server_logs.decode('utf-8'))
    except Exception as e:
        print(f"Could not retrieve server logs: {str(e)}")

def prepare_test_dataset(job_id):
    """
    Creates a properly structured dataset for the image classification task
    This ensures that the datasets/{job_id}/ directory has the right structure
    with class subdirectories containing images.
    """
    # Create test data directory if needed
    if not Path("test_data").exists():
        print("Creating test data first...")
        from create_test_data import create_test_images
        create_test_images()
    
    # Define source and destination paths
    source_dir = Path("test_data")
    dest_dir = Path(f"datasets/{job_id}")
    
    # Ensure destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy class directories to destination
    for class_dir in ["class1", "class2"]:
        # Create class directory in destination
        (dest_dir / class_dir).mkdir(exist_ok=True)
        
        # Copy all images from source to destination
        for img_file in (source_dir / class_dir).glob("*.png"):
            shutil.copy2(img_file, dest_dir / class_dir / img_file.name)
            print(f"Copied {img_file.name} to {class_dir} directory")
    
    print(f"Dataset prepared at: {dest_dir}")
    return dest_dir

def upload_dataset_files(job_id):
    """
    Upload dataset files for the job using the API endpoint
    This ensures that the uploaded files are processed correctly
    """
   
    print("\nUploading test images...")
    source_dir = Path("test_data")
    
    # Upload files for each class separately
    for class_dir in ["class1", "class2"]:
        class_path = source_dir / class_dir
        
        # Ensure the class directory exists on the server
        class_endpoint = f"{API_URL}/upload-dataset/{job_id}/{class_dir}"
        requests.post(class_endpoint)
        
        # Upload all images from this class folder
        for img_file in class_path.glob("*.png"):
            with open(img_file, "rb") as f:
                # Fix: This is the key change - put both file and class_name in files parameter
                files = {"file": (img_file.name, f, "image/png")}
                
                # Use proper multipart form data format
                data = {"class_name": class_dir}
                
                response = requests.post(
                    f"{API_URL}/upload-dataset/{job_id}",
                    files=files,
                    data=data
                )
                print(f"Uploaded {img_file.name} to {class_dir}, status: {response.status_code}")
                if response.status_code != 200:
                    print(f"  Error: {response.text}")
    
    print(f"Dataset uploaded for job: {job_id}")

def print_job_logs(job_status):
    """Helper function to print job logs in a readable format"""
    if job_status and job_status.get('logs'):
        print("\nLatest logs:")
        for log in job_status['logs'][-5:]:  # Show the last 5 logs
            print(f"  - {log}")
    
    # Print any error message
    if job_status and job_status.get('status') == TrainingStatus.FAILED:
        print("\nTraining failed! Check logs above for details.")

def check_mlflow_status(verify_ssl=None):
    """Check if MLflow server is running"""
    if verify_ssl is None:
        verify_ssl = handle_ssl_verification(API_URL)
    
    try:
        response = requests.get(f"{API_URL}/mlflow/status", verify=verify_ssl)
        status = response.json()
        print(f"MLflow server running: {status['running']}")
        if status['running']:
            # For production, MLflow UI is under /mlflow
            if '/api' in API_URL:
                mlflow_url = API_URL.replace('/api', '/mlflow')
                print(f"MLflow UI URL: {mlflow_url}")
            else:
                print(f"MLflow UI URL: {status['url']}")
        return status['running']
    except Exception as e:
        print(f"Error checking MLflow status: {str(e)}")
        return False

def test_api_endpoints():
    """Test various API endpoints in a practical example"""
    
    print("-" * 50)
    print("TESTING NO-CODE AI PLATFORM API - IMAGE CLASSIFICATION")
    print("-" * 50)
    print(f"API URL: {API_URL}")
    
    job_status = None  # Initialize job_status variable for later use
    
    # Determine SSL verification setting
    verify_ssl = handle_ssl_verification(API_URL)
    
    # 1. Check health endpoint
    print("\n1. Checking API health...")
    try:
        health_response = requests.get(f"{API_URL}/health", verify=verify_ssl)
        print(f"Status code: {health_response.status_code}")
        print(f"Response: {health_response.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server. Please check:")
        print("- Local mode: Is the server running? (python main.py)")
        print("- Docker mode: Are the containers running? (docker-compose ps)")
        print("- Production mode: Is Nginx running? (docker-compose -f docker-compose.prod.yaml ps)")
        return
    
    # Check MLflow status
    check_mlflow_status(verify_ssl)
    
    # 2. Create a new pipeline
    print("\n2. Creating a new pipeline...")
    pipeline_config = {
        "name": "test_classifier",
        "task_type": "image_classification",
        "architecture": "resnet18",
        "num_classes": 2,  # We have two classes: class1 and class2
        "batch_size": 4,
        "learning_rate": 0.001,
        "epochs": 2,      # Small number of epochs for testing
        "image_size": [128, 128],
        "augmentation_enabled": True,
        "early_stopping": True,
        "patience": 2
    }
    
    create_response = requests.post(
        f"{API_URL}/pipelines",
        json=pipeline_config,
        verify=verify_ssl
    )
    
    print(f"Status code: {create_response.status_code}")
    job_data = create_response.json()
    job_id = job_data["id"]
    print(f"Created job ID: {job_id}")
    
    # 3. Upload the test dataset - USE THIS ALTERNATIVE APPROACH
    print("\n3. Preparing test dataset...")
    prepare_test_dataset(job_id)
    
    # Verify dataset structure by listing folders
    dataset_path = Path(f"datasets/{job_id}")
    print(f"Dataset structure at {dataset_path}:")
    if not dataset_path.exists():
        print(f"  Warning: Dataset directory does not exist!")
    else:
        for class_dir in dataset_path.glob("*"):
            if class_dir.is_dir():
                image_count = len(list(class_dir.glob("*.png")))
                print(f" - {class_dir.name}/: {image_count} images")
                if image_count == 0:
                    print(f"  Warning: No images found in {class_dir.name}/")
    
    # 4. Start training
    print("\n4. Starting training...")
    train_response = requests.post(f"{API_URL}/pipelines/{job_id}/train", verify=verify_ssl)
    print(f"Status code: {train_response.status_code}")
    print(f"Response: {train_response.json()}")
    
    # 5. Monitor training progress
    print("\n5. Monitoring training progress...")
    for i in range(15):  # Check more times with longer timeout
        status_response = requests.get(f"{API_URL}/pipelines/{job_id}", verify=verify_ssl)
        job_status = status_response.json()
        print(f"Status: {job_status['status']}")
        print(f"Progress: {job_status['metrics'].get('progress', 0)}%")
        
        # Print logs to help with debugging
        if job_status['logs']:
            print("\nLatest logs:")
            for log in job_status['logs']:
                print(f"  - {log}")
        
        if job_status['status'] == TrainingStatus.COMPLETED:
            print("\nTraining completed successfully!")
            break
        elif job_status['status'] == TrainingStatus.FAILED:
            print("\nTraining failed! Check logs above for details.")
            break
        
        print(f"Waiting for 5 seconds... (Check {i+1}/15)")
        time.sleep(5)
    
    # 6. Make a prediction (this would work after training completes)
    if job_status and job_status['status'] == TrainingStatus.COMPLETED:
        print("\n6. Making prediction with trained model...")
        try:
            # Get one of our test images
            test_image_path = list(Path("test_data/class1").glob("*.png"))[0]
            with open(test_image_path, "rb") as img_file:
                files = {"file": (test_image_path.name, img_file, "image/png")}
                predict_response = requests.post(
                    f"{API_URL}/predict/{job_id}",
                    files=files,
                    verify=verify_ssl
                )
                print(f"Status code: {predict_response.status_code}")
                if predict_response.status_code == 200:
                    result = predict_response.json()
                    print(f"Prediction: {result}")
                    if 'top_prediction' in result:
                        print(f"Top prediction: {result['top_prediction']['class_name']} with {result['top_prediction']['confidence']:.2f}% confidence")
                else:
                    print(f"Prediction failed: {predict_response.json()}")
        except Exception as e:
            print(f"Prediction error: {str(e)}")
    else:
        print("\n6. Skipping prediction as training did not complete successfully")
    
    # Check for any potential server logs
    print_server_logs()
    
    # 7. Delete pipeline when done
    print("\n7. Cleaning up - deleting pipeline...")
    delete_response = requests.delete(f"{API_URL}/pipelines/{job_id}", verify=verify_ssl)
    print(f"Status code: {delete_response.status_code}")
    print(f"Response: {delete_response.json()}")
    
    print("\nAPI testing complete!")

def setup_required_directories():
    """Ensure all required directories exist before testing"""
    # Create all directories needed for the API to function
    for directory in ["models", "datasets", "logs"]:
        Path(directory).mkdir(exist_ok=True)
        print(f"Ensured directory exists: {directory}/")

def handle_ssl_verification(url):
    """Determine whether to use SSL verification based on URL"""
    # For localhost, disable verification (self-signed certs)
    # For production, enable verification
    parsed_url = urlparse(url)
    is_localhost = parsed_url.hostname in ('localhost', '127.0.0.1')
    return not is_localhost

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test the No-Code AI Platform API"
    )
    parser.add_argument(
        '--env', 
        choices=['dev', 'docker', 'prod'], 
        default='dev',
        help='Environment to test against (dev, docker, or prod)'
    )
    parser.add_argument(
        '--url', 
        help='Custom API URL (overrides --env)'
    )
    parser.add_argument(
        '--task',
        choices=['classification', 'detection', 'semantic_segmentation', 'instance_segmentation'],
        default='classification',
        help='Type of ML task to test'
    )
    parser.add_argument(
        '--verify-ssl',
        action='store_true',
        help='Enable SSL certificate verification (disabled by default for localhost)'
    )
    args = parser.parse_args()
    
    # Set API URL based on environment or custom URL
    global API_URL
    if args.url:
        API_URL = args.url
    else:
        API_URL = URLS[args.env]
    
    return args

def test_object_detection():
    """Test the object detection pipeline"""
    print("Testing object detection pipeline...")
    # Implementation similar to test_api_endpoints but for object detection
    # This would be similar to run_object_detection_example in examples.py

def test_semantic_segmentation():
    """Test the semantic segmentation pipeline"""
    print("Testing semantic segmentation pipeline...")
    # Implementation similar to test_api_endpoints but for semantic segmentation
    # This would be similar to run_semantic_segmentation_example in examples.py

def test_instance_segmentation():
    """Test the instance segmentation pipeline"""
    print("Testing instance segmentation pipeline...")
    # Implementation similar to test_api_endpoints but for instance segmentation
    # This would be similar to run_instance_segmentation_example in examples.py

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories first
    setup_required_directories()
    
    print(f"Testing against API at: {API_URL}")
    print(f"Selected task: {args.task}")
    
    # Determine SSL verification setting
    verify_ssl = args.verify_ssl if args.verify_ssl else handle_ssl_verification(API_URL)
    print(f"SSL verification: {'Enabled' if verify_ssl else 'Disabled'}")
    
    # Select the appropriate test function based on the task
    if args.task == 'classification':
        # The original test function is for classification
        test_api_endpoints()
    elif args.task == 'detection':
        test_object_detection()
    elif args.task == 'semantic_segmentation':
        test_semantic_segmentation()
    elif args.task == 'instance_segmentation':
        test_instance_segmentation()
    else:
        print(f"Unknown task: {args.task}")
        print("Please select from: classification, detection, semantic_segmentation, instance_segmentation")

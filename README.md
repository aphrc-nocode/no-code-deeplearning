# No-Code Computer Vision Backend

This project is a modular backend for computer vision tasks, supporting image classification, object detection, and image segmentation (semantic and instance) with appropriate metrics for each task.

## Features

- **Image Classification**: Train models to classify images into predefined categories
  - Metrics: Accuracy, Precision, Recall, F1 Score
  - Supported architectures: ResNet18/50, VGG16, EfficientNet, MobileNet
  - Persistent train/val/test splits for consistent evaluation

- **Object Detection**: Train models to detect and localize objects with bounding boxes
  - Metrics: Average Precision (AP), Mean Average Precision (mAP), AP at different IoU thresholds
  - Supported architectures: Faster R-CNN, SSD, RetinaNet, YOLO

- **Image Segmentation**: 
  - Semantic Segmentation: Pixel-level classification
  - Instance Segmentation: Detect, segment, and classify individual objects
  - Metrics: Mean Intersection over Union (mIoU), Pixel Accuracy, AP metrics for instance segmentation
  - Supported architectures: FCN, DeepLabV3, Mask R-CNN, UNet

## Project Structure

```
no_code_backend/
├── main.py                      # FastAPI app and API endpoints
├── pipelines/                   # Pipeline implementations for different CV tasks
│   ├── base_pipeline.py         # Abstract base pipeline interface
│   ├── image_classification_pipeline.py
│   ├── object_detection_pipeline.py
│   └── image_segmentation_pipeline.py
├── models/                      # Model definitions and factories
│   ├── classification/
│   ├── detection/
│   └── segmentation/
├── metrics/                     # Metrics calculation for each task
│   ├── classification/
│   ├── detection/
│   └── segmentation/
├── datasets_module/             # Dataset handling for each task
│   ├── classification/
│   ├── detection/
│   └── segmentation/
├── models/                      # Saved models
│   └── [job_id]/                # Job-specific model data
│       └── model.pth            # Saved model weights
└── dataset_splits/             # Persistent dataset splits (gitignored)
    └── [job_id]/               # Job-specific splits
        └── dataset_splits.json # Train/val/test splits for reproducibility
```

## Dataset Split Persistence

For image classification tasks, the system now generates and saves train/validation/test splits during training, ensuring that:

1. The same test data is used consistently during model evaluation
2. No leakage occurs between training and test data
3. Evaluation metrics are reliable and reproducible

Split information is stored in JSON format with the following structure:
```json
{
  "train": [0, 3, 7, ...],  // Indices for training set
  "val": [1, 4, 9, ...],    // Indices for validation set
  "test": [2, 5, 8, ...],   // Indices for testing set
  "random_seed": 42,        // Random seed used for reproducibility
  "dataset_path": "/path/to/dataset"  // Original dataset path
}
```

These splits are automatically loaded during evaluation when the same job_id is provided, ensuring that the model is always evaluated on the same test data that was held out during training.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- FastAPI

### Installation

#### Option 1: Standard Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Option 2: Docker Installation

1. Clone the repository
2. Make sure you have Docker and Docker Compose installed
3. Run the setup script:
   ```bash
   ./docker-setup.sh
   ```
   
   Or manually:
   ```bash
   # Create necessary directories
   mkdir -p models datasets logs mlruns
   
   # Build and start the containers
   docker-compose build
   docker-compose up -d
   ```

### Running the Server

#### Standard Mode:

```bash
python main.py
```

The server will be available at http://localhost:8000

#### Docker Mode:

The server will be automatically started when running Docker Compose and will be available at http://localhost:8000. MLflow UI will be available at http://localhost:5000.

### Quick Start Guide

#### Development Setup:

```bash
# Clone this repository
git clone <repository-url>
cd no_code_backend

# Start the development environment
./docker-setup.sh

# Open the API documentation in your browser
xdg-open http://localhost:8000/docs  # Linux
open http://localhost:8000/docs      # macOS
```

#### Production Setup:

```bash
# Clone this repository
git clone <repository-url>
cd no_code_backend

# Create or update .env.prod with your settings
cp .env.example .env.prod
nano .env.prod

# Deploy the application
sudo ./deploy-prod.sh

# Access the application
xdg-open https://localhost/api/docs  # Linux
open https://localhost/api/docs      # macOS
```

### API Endpoints

- **POST /pipelines**: Create a new training pipeline
- **POST /pipelines/{job_id}/train**: Start training for a pipeline
- **GET /pipelines/{job_id}**: Get status of a training job
- **POST /predict/{job_id}**: Make predictions with a trained model
- **GET /mlflow/ui-url**: Get MLflow UI URL

## Usage Examples

Check the `examples.py` file for examples of using different pipelines:

```bash
python examples.py
```

## Docker Management

### Development Environment

Use the development Docker setup for local testing and development:

```bash
# Start the development environment
./docker-setup.sh

# View logs from all services
docker-compose logs -f

# View logs from specific service
docker-compose logs -f app

# Stop all services but retain volumes
docker-compose down

# Stop all services and remove volumes
docker-compose down -v
```

### Production Deployment

For production deployment with Nginx and SSL:

```bash
# Deploy in production mode (requires root for ports 80/443)
sudo ./deploy-prod.sh

# View logs from all services
docker-compose -f docker-compose.prod.yaml logs -f

# Stop production deployment
docker-compose -f docker-compose.prod.yaml down
```

### Rebuilding Images

```bash
# Rebuild development images
docker-compose build

# Rebuild and restart development services 
docker-compose up -d --build

# Rebuild production images
docker-compose -f docker-compose.prod.yaml build
```

### Supported Dataset Tasks

The integration supports the following dataset tasks:

- **Image Classification**: Datasets with image and label fields
- **Object Detection**: Datasets with image and object annotations (boxes, labels)
- **Semantic Segmentation**: Datasets with image and segmentation mask
- **Instance Segmentation**: Datasets with image and instance masks

## License

[MIT](LICENSE)

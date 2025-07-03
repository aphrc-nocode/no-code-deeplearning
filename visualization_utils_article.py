"""
Visualization utilities for object detection following the article's approach.
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional


def visualize_batch_from_dataloader(dataloader, num_samples=2, figsize=(16, 12)):
    """
    Visualize a batch of samples from dataloader exactly as shown in the article
    
    Args:
        dataloader: DataLoader containing images and targets
        num_samples: Number of samples to visualize
        figsize: Figure size for matplotlib
    """
    # Get one batch from the DataLoader
    images, targets = next(iter(dataloader))

    # Convert PIL Image and draw annotations
    for i in range(min(len(images), num_samples)):
        image = images[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        image = (image * 255).astype(np.uint8)  # Rescale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        boxes = targets[i]['boxes']
        labels = targets[i]['labels']

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Class {label.item()}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show image with boxes using matplotlib
        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Sample {i + 1}")
        plt.show()


def visualize_predictions(image_path, model, device, label_list, threshold=0.8, figsize=(16, 12)):
    """
    Visualize model predictions on a single image following the article's approach
    
    Args:
        image_path: Path to the test image
        model: Trained model
        device: Device to run inference on
        label_list: List of class names
        threshold: Confidence threshold for displaying predictions
        figsize: Figure size for matplotlib
    """
    from torchvision import transforms
    
    # Load image with OpenCV and convert to RGB
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Transform image
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Parse predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Draw predictions above threshold
    for i in range(len(boxes)):
        if scores[i] > threshold:
            box = boxes[i].cpu().numpy().astype(int)
            label = label_list[labels[i]] if labels[i] < len(label_list) else f"Class {labels[i]}"
            score = scores[i].item()

            # Draw label and score
            text = f"{label}: {score:.2f}"
            cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw rectangle
            cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

    # Convert BGR to RGB for correct display with matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Show image with larger figure size
    plt.figure(figsize=figsize)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"Predictions (threshold: {threshold})")
    plt.show()


def visualize_training_batch(images, targets, max_images=2, figsize=(12, 12)):
    """
    Visualize a batch of training images with their bounding boxes
    
    Args:
        images: List of tensor images
        targets: List of target dictionaries with boxes and labels
        max_images: Maximum number of images to display
        figsize: Figure size for the plot
    """
    plt.figure(figsize=figsize)
    for i, (img, target) in enumerate(zip(images, targets)):
        if i >= max_images:
            break
            
        # Convert tensor to numpy array
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f"Class {label}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Convert back to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, max_images, i+1)
        plt.imshow(img_rgb)
        plt.axis("off")
        plt.title(f"Training Sample {i+1}")
    
    plt.tight_layout()
    plt.show()

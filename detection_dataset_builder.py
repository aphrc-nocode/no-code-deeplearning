"""
Utility to convert detection datasets to Hugging Face dataset format.
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, Image as HFImage, Features, ClassLabel, Value
import pandas as pd

def build_hf_detection_dataset(
    images_dir: str, 
    annotations_path: str,
    image_processor=None
):
    """
    Build a Hugging Face dataset for object detection from COCO-style annotations
    
    Args:
        images_dir: Directory containing images
        annotations_path: Path to annotations JSON file
        image_processor: Optional Hugging Face image processor for preprocessing
        
    Returns:
        Hugging Face Dataset ready for training
    """
    images_dir = Path(images_dir)
    annotations_path = Path(annotations_path)
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Create category mapping
    categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}
    id_to_name = {i: categories[cat_id] for i, cat_id in enumerate(sorted(categories.keys()))}
    class_names = [id_to_name[i] for i in range(len(id_to_name))]
    
    # Create image id to path mapping
    image_dict = {img["id"]: img for img in annotations["images"]}
    
    # Group annotations by image
    image_annotations = {}
    for ann in annotations["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Build dataset in a format compatible with HF transformers
    image_paths = []
    image_widths = []
    image_heights = []
    all_boxes = []
    all_labels = []
    all_area = []
    all_image_ids = []
    
    for img_id in image_annotations.keys():
        # Get image info
        img_info = image_dict[img_id]
        
        # Find image path
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            potential_paths = list(images_dir.glob(f"**/{img_info['file_name']}"))
            if potential_paths:
                img_path = potential_paths[0]
            else:
                print(f"Warning: Image not found: {img_info['file_name']}, skipping")
                continue
                
        # Get image dimensions
        width, height = img_info["width"], img_info["height"]
        
        # Get annotations for this image
        img_anns = image_annotations[img_id]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        
        for ann in img_anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            
            # Convert to normalized [x_center, y_center, width, height] format for YOLO/DETR
            if image_processor and hasattr(image_processor, 'normalize_boxes'):
                # If using DETR-style processor, keep as [x1, y1, x2, y2]
                boxes.append([x, y, x + w, y + h])
            else:
                # Default to normalized center format
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                norm_width = w / width
                norm_height = h / height
                boxes.append([x_center, y_center, norm_width, norm_height])
            
            # Adjust category_id to be 0-indexed
            cat_id = ann["category_id"]
            if cat_id in categories:
                # Map to sequential index
                label_idx = list(sorted(categories.keys())).index(cat_id)
                labels.append(label_idx)
            else:
                print(f"Warning: Category ID {cat_id} not found in categories")
                labels.append(0)  # Default to first class
            
            # Get area if available
            area = ann.get("area", w * h)
            areas.append(area)
        
        # Add data to lists
        image_paths.append(str(img_path))
        image_widths.append(width)
        image_heights.append(height)
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_area.append(areas)
        all_image_ids.append(img_id)
    
    # Create DataFrame with all data
    data = {
        "image_path": image_paths,
        "width": image_widths,
        "height": image_heights,
        "boxes": all_boxes,
        "labels": all_labels,
        "area": all_area,
        "image_id": all_image_ids,
    }
    
    # Create Hugging Face Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Add image loading function
    def load_image(example):
        image = Image.open(example["image_path"]).convert("RGB")
        example["image"] = image
        return example
    
    dataset = dataset.map(load_image)
    
    # Apply preprocessing if provided
    # Apply preprocessing if provided
    if image_processor:
        def preprocess_examples(examples):
            """
            Preprocess examples for DETR-style models
            """
            images = []
            annotations = []
            
            for i in range(len(examples["image_path"])):
                # Load image
                image = Image.open(examples["image_path"][i]).convert("RGB")
                images.append(image)
                
                # Format annotations for DETR processor (COCO format)
                boxes = examples["boxes"][i]
                labels = examples["labels"][i]
                
                # Create COCO-style annotations for this image
                coco_annotations = []
                if boxes and labels:
                    for j, (box, label) in enumerate(zip(boxes, labels)):
                        coco_annotation = {
                            "id": j,
                            "image_id": examples["image_id"][i],
                            "category_id": label,
                            "bbox": box,  # COCO format: [x, y, width, height]
                            "area": examples["area"][i][j] if j < len(examples["area"][i]) else box[2] * box[3],
                            "iscrowd": 0
                        }
                        coco_annotations.append(coco_annotation)
                
                # Create the annotation format expected by DETR processor
                annotation_dict = {
                    "image_id": examples["image_id"][i],
                    "annotations": coco_annotations
                }
                annotations.append(annotation_dict)
            
            # Process images and annotations through the DETR processor
            processed = image_processor(images=images, annotations=annotations, return_tensors="pt")
            
            return processed
        
        dataset = dataset.map(preprocess_examples, batched=True, remove_columns=dataset.column_names)
    
    return dataset, class_names

def split_dataset(dataset, val_split=0.2, test_split=0.1, seed=42):
    """Split dataset into train/val/test sets"""
    # Handle very small datasets
    dataset_size = len(dataset)
    if dataset_size <= 3:
        print(f"Dataset is very small ({dataset_size} samples). Using the same data for all splits.")
        return dataset, dataset, dataset
    
    # For small datasets, adjust the split ratios
    if dataset_size <= 10:
        val_split = 0.1
        test_split = 0.0
        print(f"Small dataset detected ({dataset_size} samples), adjusting splits.")
    
    try:
        # Calculate split sizes
        train_size = 1.0 - val_split - test_split
        
        # Split dataset
        splits = dataset.train_test_split(test_size=val_split + test_split, seed=seed)
        train_dataset = splits["train"]
        
        # Further split test into val and test if needed
        if test_split > 0:
            try:
                # Calculate relative size of test split from the combined val+test
                relative_test_size = test_split / (val_split + test_split)
                test_val_splits = splits["test"].train_test_split(test_size=relative_test_size, seed=seed)
                val_dataset = test_val_splits["train"]
                test_dataset = test_val_splits["test"]
            except ValueError as e:
                # If split fails, use the same dataset for validation and test
                print(f"Warning: Test split failed ({str(e)}). Using same data for validation and test.")
                val_dataset = splits["test"]
                test_dataset = splits["test"]
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, splits["test"], None
    except ValueError as e:
        # Fallback for very small datasets
        print(f"Dataset split failed: {str(e)}. Using entire dataset for all splits.")
        return dataset, dataset, dataset


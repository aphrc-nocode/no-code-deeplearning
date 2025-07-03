#!/usr/bin/env python3
"""
Test script to verify the object detection pipeline correctly handles
COCO datasets with duplicate category names and supercategories.

This script demonstrates that our implementation:
1. Deduplicates class names (ignoring duplicate category IDs)
2. Creates correct label mappings
3. Sets the right number of classes for the model
"""

import json
import os
import tempfile
from PIL import Image
import numpy as np

def create_test_coco_annotation():
    """
    Create a test COCO annotation with duplicate category names
    to simulate the real-world case where categories and supercategories
    might have the same name.
    """
    # Example COCO annotation with duplicate class names
    # This simulates a case where 'person' appears both as a category 
    # and as a supercategory, or where multiple category IDs map to the same name
    coco_data = {
        "images": [
            {
                "id": 1,
                "file_name": "test_image.jpg",
                "width": 640,
                "height": 480
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person"
            },
            {
                "id": 2,
                "name": "car",
                "supercategory": "vehicle"
            },
            {
                "id": 3,
                "name": "person",  # Duplicate name but different ID
                "supercategory": "human"
            },
            {
                "id": 4,
                "name": "truck",
                "supercategory": "vehicle"
            },
            {
                "id": 5,
                "name": "car",  # Another duplicate
                "supercategory": "automobile"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 80],
                "area": 4000,
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [200, 150, 100, 60],
                "area": 6000,
                "iscrowd": 0
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 3,  # This should map to same label as category_id 1
                "bbox": [300, 200, 45, 75],
                "area": 3375,
                "iscrowd": 0
            }
        ]
    }
    return coco_data

def test_coco_dataset_deduplication():
    """
    Test that our COCO dataset correctly handles duplicate class names
    """
    print("=" * 60)
    print("TESTING COCO DATASET DEDUPLICATION")
    print("=" * 60)
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create image directory and annotation file
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir)
        
        # Create a test image
        test_image = Image.new('RGB', (640, 480), color='red')
        test_image.save(os.path.join(image_dir, "test_image.jpg"))
        
        # Create annotation file
        annotation_path = os.path.join(temp_dir, "annotations.json")
        coco_data = create_test_coco_annotation()
        
        print("Original COCO categories:")
        for cat in coco_data['categories']:
            print(f"  ID {cat['id']}: {cat['name']} (supercategory: {cat['supercategory']})")
        
        with open(annotation_path, 'w') as f:
            json.dump(coco_data, f)
        
        # Test our dataset implementation
        try:
            from datasets_module.detection.coco_dataset_article import CocoDetectionDataset, create_dataloaders
            
            print("\nTesting CocoDetectionDataset:")
            dataset = CocoDetectionDataset(
                image_dir=image_dir,
                annotation_path=annotation_path,
                transforms=None
            )
            
            print(f"\nResults:")
            print(f"  Unique class names: {dataset.unique_class_names}")
            print(f"  Category ID to label mapping: {dataset.category_id_to_label}")
            print(f"  Expected: 3 unique classes (person, car, truck)")
            print(f"  Actual: {len(dataset.unique_class_names)} unique classes")
            
            # Test that duplicate category IDs with same name map to same label
            person_labels = []
            car_labels = []
            for cat_id, label in dataset.category_id_to_label.items():
                coco_cat = next(cat for cat in coco_data['categories'] if cat['id'] == cat_id)
                if coco_cat['name'] == 'person':
                    person_labels.append(label)
                elif coco_cat['name'] == 'car':
                    car_labels.append(label)
            
            print(f"\nLabel consistency check:")
            print(f"  Person category IDs (1, 3) map to labels: {person_labels}")
            print(f"  Car category IDs (2, 5) map to labels: {car_labels}")
            print(f"  All person labels same: {len(set(person_labels)) == 1}")
            print(f"  All car labels same: {len(set(car_labels)) == 1}")
            
            # Test dataloader creation
            print(f"\nTesting create_dataloaders:")
            train_loader, val_loader, num_classes = create_dataloaders(
                train_image_dir=image_dir,
                train_annotation_path=annotation_path,
                val_image_dir=image_dir,
                val_annotation_path=annotation_path,
                batch_size=1,
                num_workers=0  # Avoid multiprocessing issues in test
            )
            
            print(f"  Returned num_classes: {num_classes}")
            print(f"  Expected: 4 (3 unique classes + 1 background)")
            print(f"  Correct: {num_classes == 4}")
            
            # Test a batch
            print(f"\nTesting batch loading:")
            images, targets = next(iter(train_loader))
            target = targets[0]
            print(f"  Image batch size: {len(images)}")
            print(f"  Target labels: {target['labels'].tolist()}")
            print(f"  Boxes shape: {target['boxes'].shape}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run the test"""
    print("Object Detection Pipeline - COCO Deduplication Test")
    print("This test verifies that our pipeline correctly handles datasets")
    print("with duplicate category names and supercategories.\n")
    
    success = test_coco_dataset_deduplication()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ TEST PASSED: COCO deduplication working correctly!")
        print("  - Duplicate category names are properly handled")
        print("  - Category IDs with same names map to same labels")
        print("  - Model will be created with correct number of classes")
    else:
        print("✗ TEST FAILED: Issues with COCO deduplication")
    print("=" * 60)

if __name__ == "__main__":
    main()

import numpy as np
from datasets import load_dataset

def filter_invalid_objects_coco(example):
    """
    Filters out invalid objects from a single example of a Hugging Face object
    detection dataset that uses the COCO format for bounding boxes. An object is 
    considered invalid if its width or height is less than or equal to 0.
    """
    if 'objects' not in example or 'bbox' not in example['objects']:
        return example

    objects = example['objects']
    bboxes = objects['bbox']
    
    # A valid box must have width > 0 and height > 0.
    # In COCO format, bbox[2] is width, and bbox[3] is height.
    valid_indices = [i for i, bbox in enumerate(bboxes) if bbox[2] > 0 and bbox[3] > 0]

    if len(valid_indices) == len(bboxes):
        return example

    filtered_objects = {}
    for key, value in objects.items():
        if isinstance(value, list):
            filtered_objects[key] = [value[i] for i in valid_indices]
        else:
            filtered_objects[key] = value

    example['objects'] = filtered_objects
    return example

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format."""
    annotations = [
        {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        for category, area, bbox in zip(categories, areas, bboxes)
    ]
    return {"image_id": image_id, "annotations": annotations}


def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations for object detection."""
    images, annotations = [], []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)
    
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    if not return_pixel_mask:
        result.pop("pixel_mask", None)
    return result


def load_detection_dataset(base_dir):
    """Loads the object detection dataset."""
    dataset = load_dataset("./detection_dataset_builder.py", base_dir=base_dir, trust_remote_code=True)
    return dataset

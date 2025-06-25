import numpy as np
from datasets import load_dataset

def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format.

    Args:
        image_id (str): Image ID.
        categories (List[int]): List of class labels.
        areas (List[float]): List of bounding box areas.
        bboxes (List[Tuple[float]]): List of bounding boxes in COCO format.

    Returns:
        dict: A dictionary containing the image ID and formatted annotations.
    """
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
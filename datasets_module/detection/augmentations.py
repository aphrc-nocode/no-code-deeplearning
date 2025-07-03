"""
Data augmentation pipeline for object detection using albumentations.
"""
import albumentations as A
from albumentations.core.composition import Compose

def get_train_transform(max_image_size=800):
    """Returns the training augmentation pipeline."""
    return A.Compose(
        [
            A.OneOf([
                A.HorizontalFlip(p=0.9),
                A.VerticalFlip(p=0.6)
            ], p=0.8),
            A.OneOf([
                A.ColorJitter(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.8),
            A.BBoxSafeRandomCrop(erosion_rate=0.5, p=0.5),
            A.LongestMaxSize(max_size=max_image_size),
            A.PadIfNeeded(max_image_size, max_image_size, border_mode=0, position="center"),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_visibility=0.5),
    )

def get_validation_transform(max_image_size=800):
    """Returns the validation transformation pipeline."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_image_size),
            A.PadIfNeeded(max_image_size, max_image_size, border_mode=0, position="center"),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations for object detection."""
    import numpy as np
    
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

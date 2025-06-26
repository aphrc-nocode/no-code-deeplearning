import torch
from transformers import AutoConfig, AutoModelForObjectDetection, AutoImageProcessor

def load_model(model_checkpoint, id2label, label2id):
    """Loads the object detection model using a custom config."""
    config = AutoConfig.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model

def load_image_processor(model_checkpoint, max_image_size):
    """Loads the image processor with specific resizing and padding."""
    return AutoImageProcessor.from_pretrained(
        model_checkpoint,
        do_resize=True,
        size={"max_height": max_image_size, "max_width": max_image_size},
        do_pad=True,
        pad_size={"height": max_image_size, "width": max_image_size},
        use_fast=True,
    )


def collate_fn(batch):
    """Custom collate function for object detection."""
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data

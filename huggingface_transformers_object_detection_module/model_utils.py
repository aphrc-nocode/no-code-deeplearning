import torch
from transformers import AutoModelForObjectDetection

def load_model(model_checkpoint, id2label, label2id):
    """Loads the object detection model."""
    model = AutoModelForObjectDetection.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model

def collate_fn(batch):
    """Custom collate function for object detection."""
    data = {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": [x["labels"] for x in batch],
    }
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data
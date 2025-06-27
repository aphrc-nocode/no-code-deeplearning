import warnings
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import Wav2Vec2BertProcessor


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        valid_features = [
            f
            for f in features
            if f.get("input_features") is not None and f.get("labels") is not None and len(f["input_features"]) > 0
        ]

        if not valid_features:
            warnings.warn("Data collator received an empty list of valid features or features with empty input_features.")
            return {
                "input_features": torch.zeros((0, 0)),
                "labels": torch.full((0, 0), -100, dtype=torch.long),
            }

        input_features_list = [{"input_features": feature["input_features"]} for feature in valid_features]
        label_features_list = [{"input_ids": feature["labels"]} for feature in valid_features]

        batch = self.processor.pad(
            input_features=input_features_list,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.pad(
            labels=label_features_list,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def compute_metrics_fn(processor, wer_metric_obj, cer_metric_obj):
    """Returns a compute_metrics function for the Trainer."""

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric_obj.compute(predictions=pred_str, references=label_str)
        cer = cer_metric_obj.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics

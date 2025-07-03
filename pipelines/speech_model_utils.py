"""
Model utilities for speech recognition pipeline.
Contains the data collator and compute metrics function for CTC training.
"""
import warnings
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import torch
from transformers import Wav2Vec2BertProcessor
import jiwer


def compute_metrics_fn(processor, wer_metric_obj=None, cer_metric_obj=None):
    """Returns a compute_metrics function for the Trainer."""

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # Use jiwer directly instead of evaluate library to avoid version conflicts
        try:
            wer = jiwer.wer(label_str, pred_str)
            cer = jiwer.cer(label_str, pred_str)
        except Exception as e:
            print(f"Warning: Error computing WER/CER: {e}")
            # Return dummy values if calculation fails
            wer = 1.0
            cer = 1.0

        return {"wer": wer, "cer": cer}

    return compute_metrics

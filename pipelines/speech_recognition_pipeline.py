"""
Pipeline for automatic speech recognition (ASR) tasks.
This module implements the BasePipeline interface for speech recognition using Wav2Vec2-BERT.
"""
import os
import json
import math
import random
import warnings
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from functools import partial

import torch
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict, Audio, load_dataset
from transformers import (
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from pipelines.base_pipeline import BasePipeline


class SpeechRecognitionPipeline(BasePipeline):
    """Pipeline for automatic speech recognition tasks"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # ASR-specific configuration with defaults
        self.language_code = getattr(config, 'language_code', 'en')
        self.language = getattr(config, 'language', 'english')
        self.target_sampling_rate = getattr(config, 'target_sampling_rate', 16000)
        self.min_duration_s = getattr(config, 'min_duration_s', 1.0)
        self.max_duration_s = getattr(config, 'max_duration_s', 30.0)
        self.min_transcript_len = getattr(config, 'min_transcript_len', 10)
        self.max_transcript_len = getattr(config, 'max_transcript_len', 300)
        self.outlier_std_devs = getattr(config, 'outlier_std_devs', 2.0)
        self.apply_outlier_filtering = getattr(config, 'apply_outlier_filtering', True)
        self.alphabet = getattr(config, 'alphabet', " 'abcdefghijklmnopqrstuvwxyz")
        self.model_checkpoint = getattr(config, 'model_checkpoint', 'facebook/w2v-bert-2.0')
        
        # Training-specific params
        self.warmup_ratio = getattr(config, 'warmup_ratio', 0.1)
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 5)
        self.early_stopping_threshold = getattr(config, 'early_stopping_threshold', 1e-3)
        
        # Column names
        self.audio_column = getattr(config, 'audio_column', 'audio')
        self.text_column = getattr(config, 'text_column', 'sentence')
        self.normalized_text_column = getattr(config, 'normalized_text_column', 'normalized_text')
        self.duration_column = getattr(config, 'duration_column', 'duration')
        
        # Constants
        self.UNK_TOKEN = "[UNK]"
        self.PAD_TOKEN = "[PAD]"
        self.WORD_DELIMITER_TOKEN = "|"
        
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train the ASR model with the given dataset"""
        try:
            # Start MLflow run
            from mlflow_utils import start_run, log_metrics, log_model, end_run
            self.run_id = start_run(job_id, self.config.dict())
            
            await self._update_job_log(job_id, "Loading and preprocessing speech dataset...")
            
            # Detect dataset format and load accordingly
            from pipelines.speech_utils import detect_dataset_format, load_csv_speech_dataset
            
            dataset_format = detect_dataset_format(dataset_path)
            await self._update_job_log(job_id, f"Detected dataset format: {dataset_format}")
            
            if dataset_format == "csv":
                dataset = load_csv_speech_dataset(
                    dataset_path=dataset_path,
                    audio_column=self.audio_column,
                    text_column=self.text_column,
                    target_sampling_rate=self.target_sampling_rate
                )
            elif dataset_format == "audiofolder":
                dataset = load_dataset("audiofolder", data_dir=dataset_path)
                dataset = dataset.cast_column(self.audio_column, Audio(sampling_rate=self.target_sampling_rate))
            else:
                raise ValueError(f"Unsupported or unknown dataset format. Expected 'csv' or 'audiofolder', got '{dataset_format}'")
            
            # Check for problematic files
            for split in dataset.keys():
                problematic_files = self._find_problematic_audio_files(
                    dataset[split], audio_column=self.audio_column
                )
                
                if problematic_files:
                    indices_to_remove = {item["index"] for item in problematic_files}
                    indices_to_keep = sorted(list(set(range(len(dataset[split]))) - indices_to_remove))
                    dataset[split] = dataset[split].select(indices_to_keep)
                    await self._update_job_log(job_id, f"Removed {len(problematic_files)} problematic audio files from {split}")
            
            if len(dataset) == 0:
                raise ValueError("Dataset is empty after removing problematic files")
            
            # Add duration column
            dataset = self._add_duration_column(dataset)
            
            # Split dataset if not pre-split
            if 'validation' not in dataset and 'test' not in dataset:
                await self._update_job_log(job_id, "Splitting dataset into train/validation/test...")
                dataset = self._split_dataset(dataset)
            
            # Process dataset for speech recognition
            dataset = self._process_speech_dataset(dataset)
            
            await self._update_job_log(job_id, f"Dataset loaded: Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")
            
            # Create vocabulary and processor
            await self._update_job_log(job_id, "Creating vocabulary and processor...")
            processor = self._create_processor(job_id)
            
            # Prepare datasets for CTC
            await self._update_job_log(job_id, "Preparing datasets for CTC training...")
            dataset = self._prepare_datasets_for_ctc(dataset, processor)
            
            # Filter incompatible samples
            train_dataset = dataset["train"].filter(lambda x: self._check_ctc_compatibility(x))
            val_dataset = dataset["validation"].filter(lambda x: self._check_ctc_compatibility(x))
            test_dataset = dataset["test"].filter(lambda x: self._check_ctc_compatibility(x))
            
            await self._update_job_log(job_id, f"After CTC filtering: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            if len(train_dataset) == 0:
                raise ValueError("Training dataset is empty after CTC compatibility filtering")
            
            # Create model
            await self._update_job_log(job_id, "Creating ASR model...")
            model = self._create_model(processor)
            
            # Setup training
            data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
            
            # Metrics
            wer_metric = evaluate.load("wer")
            cer_metric = evaluate.load("cer")
            compute_metrics = self._create_compute_metrics_fn(processor, wer_metric, cer_metric)
            
            # Training arguments
            output_dir = f"models/{job_id}"
            training_args = TrainingArguments(
                output_dir=output_dir,
                group_by_length=True,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                eval_strategy="epoch",
                logging_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=self.config.epochs,
                gradient_checkpointing=True,
                fp16=torch.cuda.is_available(),
                learning_rate=self.config.learning_rate,
                lr_scheduler_type="linear",
                warmup_ratio=self.warmup_ratio,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="wer",
                greater_is_better=False,
                remove_unused_columns=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=processor.feature_extractor,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.early_stopping_patience,
                        early_stopping_threshold=self.early_stopping_threshold,
                    )
                ],
            )
            
            # Train
            await self._update_job_log(job_id, "Starting ASR model training...")
            train_result = trainer.train()
            
            # Evaluate on test set
            await self._update_job_log(job_id, "Evaluating on test set...")
            test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            
            # Save model and processor
            model_path = Path(output_dir) / "final_model"
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)
            
            # Log to MLflow
            log_model(model, "model", {
                'model_type': 'speech_recognition',
                'architecture': self.config.architecture,
                'language': self.language,
                'language_code': self.language_code,
                'target_sampling_rate': self.target_sampling_rate
            })
            
            # Log metrics
            final_metrics = {
                "train_loss": train_result.training_loss,
                "test_wer": test_results.get("test_wer", 0.0),
                "test_cer": test_results.get("test_cer", 0.0),
            }
            log_metrics(final_metrics)
            
            end_run()
            
            await self._update_job_log(job_id, f"Training completed! Test WER: {test_results.get('test_wer', 0.0):.4f}")
            
            return {
                "status": "completed",
                "model_path": str(model_path),
                "test_wer": test_results.get("test_wer", 0.0),
                "test_cer": test_results.get("test_cer", 0.0),
                "train_loss": train_result.training_loss,
                "epochs_completed": int(train_result.epoch)
            }
            
        except Exception as e:
            await self._update_job_log(job_id, f"Error during training: {str(e)}")
            
            try:
                from mlflow_utils import end_run
                end_run()
            except:
                pass
                
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def create_model(self):
        """Create a Wav2Vec2-BERT model for CTC"""
        # This is a simplified version for interface compliance
        # The actual model creation happens in _create_model with processor
        from transformers import Wav2Vec2BertForCTC
        model = Wav2Vec2BertForCTC.from_pretrained(self.model_checkpoint)
        return model
    
    async def predict(self, audio_path, model=None) -> Dict[str, Any]:
        """Make a prediction on a single audio file"""
        try:
            import librosa
            from transformers import Wav2Vec2BertProcessor
            
            # Load audio
            audio_array, sampling_rate = librosa.load(audio_path, sr=self.target_sampling_rate)
            
            # Create processor if not provided
            if model is None:
                # For prediction, we'd need a trained model path
                raise ValueError("Model required for prediction")
            
            # Create processor (simplified - in real use we'd load saved processor)
            processor = Wav2Vec2BertProcessor.from_pretrained(self.model_checkpoint)
            
            # Process audio
            inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
            
            # Predict
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            return {
                "transcription": transcription,
                "confidence": float(torch.softmax(logits, dim=-1).max())
            }
            
        except Exception as e:
            return {"error": str(e), "transcription": ""}
    
    async def evaluate(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate the model on a test dataset"""
        try:
            # This would load a saved model and evaluate it
            # For now, return a placeholder implementation
            return {
                "wer": 0.0,
                "cer": 0.0,
                "dataset_size": 0,
                "note": "Evaluation requires a trained model"
            }
        except Exception as e:
            return {"error": str(e)}

    def get_transforms(self):
        """Get audio transforms - not applicable for speech recognition in the same way as vision"""
        # Speech preprocessing is handled by the processor, not traditional transforms
        return None
    
    def _create_processor(self, job_id: str):
        """Create vocabulary, tokenizer, and processor"""
        # Create vocabulary
        alphabet_list = sorted(list(set(self.alphabet)))
        vocab_dict = {v: k for k, v in enumerate(alphabet_list)}
        vocab_dict[self.WORD_DELIMITER_TOKEN] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict[self.UNK_TOKEN] = len(vocab_dict)
        vocab_dict[self.PAD_TOKEN] = len(vocab_dict)
        
        # Save vocabulary
        vocab_path = f"models/{job_id}/vocab.json"
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False)
        
        # Create tokenizer
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            os.path.dirname(vocab_path),
            unk_token=self.UNK_TOKEN,
            pad_token=self.PAD_TOKEN,
            word_delimiter_token=self.WORD_DELIMITER_TOKEN,
        )
        
        # Create feature extractor
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(self.model_checkpoint)
        
        # Create processor
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, 
            tokenizer=tokenizer
        )
        
        return processor
    
    def _create_model(self, processor):
        """Create the ASR model"""
        model = Wav2Vec2BertForCTC.from_pretrained(
            self.model_checkpoint,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.0,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            add_adapter=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
            ignore_mismatched_sizes=True,
        )
        return model.to(self.device)
    
    def _split_dataset(self, dataset):
        """Split dataset into train/validation/test"""
        from pipelines.speech_utils import random_split
        
        train, dev, test = random_split(
            dataset,
            train_ratio=0.8,
            dev_ratio=0.1,
            random_seed=42
        )
        
        return DatasetDict({"train": train, "validation": dev, "test": test})
    
    def _process_speech_dataset(self, dataset):
        """Process speech dataset with filtering and normalization"""
        from pipelines.speech_data_processing import process_prepared_speech_dataset
        
        return process_prepared_speech_dataset(
            input_dataset=dataset,
            text_column=self.text_column,
            normalized_text_column=self.normalized_text_column,
            duration_column=self.duration_column,
            min_duration_s=self.min_duration_s,
            max_duration_s=self.max_duration_s,
            min_transcript_len=self.min_transcript_len,
            max_transcript_len=self.max_transcript_len,
            outlier_std_devs=self.outlier_std_devs,
            apply_outlier_filtering=self.apply_outlier_filtering,
            num_proc=1
        )
    
    def _prepare_datasets_for_ctc(self, dataset, processor):
        """Prepare datasets for CTC training"""
        from pipelines.speech_data_processing import prepare_dataset_for_ctc
        
        prepare_fn = partial(
            prepare_dataset_for_ctc,
            processor=processor,
            audio_col=self.audio_column,
            normalized_text_col=self.normalized_text_column,
        )
        
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                prepare_fn,
                remove_columns=dataset[split].column_names,
                desc=f"Preparing {split} data for CTC"
            )
        
        return dataset
    
    def _check_ctc_compatibility(self, batch):
        """Check if batch is compatible with CTC training"""
        from pipelines.speech_data_processing import check_ctc_compatibility
        return check_ctc_compatibility(batch)
    
    def _add_duration_column(self, dataset):
        """Add duration column to dataset"""
        from pipelines.speech_utils import add_duration_column
        return add_duration_column(dataset, self.audio_column, self.duration_column)
    
    def _find_problematic_audio_files(self, dataset, audio_column):
        """Find problematic audio files in dataset"""
        from pipelines.speech_utils import find_problematic_audio_files
        return find_problematic_audio_files(dataset, audio_column, verbose=False)
    
    def _create_compute_metrics_fn(self, processor, wer_metric, cer_metric):
        """Create compute metrics function"""
        from pipelines.speech_model_utils import compute_metrics_fn
        return compute_metrics_fn(processor, wer_metric, cer_metric)
    
    @staticmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        return ["wer", "cer", "train_loss"]
    
    async def _update_job_log(self, job_id: str, message: str):
        """Update the job log with a message"""
        print(f"Job {job_id}: {message}")


# Data Collator class needed for CTC training
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding
    
    def __call__(self, features):
        valid_features = [
            f for f in features
            if f.get("input_features") is not None and 
               f.get("labels") is not None and 
               len(f["input_features"]) > 0
        ]
        
        if not valid_features:
            warnings.warn("Data collator received empty features")
            return {
                "input_features": torch.zeros((0, 0)),
                "labels": torch.full((0, 0), -100, dtype=torch.long),
            }
        
        input_features = [{"input_features": f["input_features"]} for f in valid_features]
        label_features = [{"input_ids": f["labels"]} for f in valid_features]
        
        batch = self.processor.pad(
            input_features=input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        
        return batch

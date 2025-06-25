import os
import argparse
from functools import partial
from pprint import pprint
import wandb

from transformers import AutoImageProcessor, TrainingArguments, Trainer
from data_utils import load_detection_dataset, augment_and_transform_batch
from augmentations import get_train_transform, get_validation_transform
from model_utils import load_model, collate_fn
from metrics import compute_metrics

def main(args):
    """Main function to run the training and evaluation."""
    
    # --- Disable W&B if not needed ---
    if not args.use_wandb:
        wandb.init(mode="disabled")

    # --- Data Loading and Setup ---
    dataset = load_detection_dataset(args.base_dir)
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = {i: name for i, name in enumerate(categories)}
    label2id = {name: i for i, name in id2label.items()}
    
    # --- Image Processor ---
    image_processor = AutoImageProcessor.from_pretrained(args.model_checkpoint, do_pad=False, do_resize=False)

    # --- Augmentation and Transformation ---
    train_transform = get_train_transform(args.max_image_size)
    val_transform = get_validation_transform(args.max_image_size)
    
    train_transform_batch = partial(augment_and_transform_batch, transform=train_transform, image_processor=image_processor)
    val_transform_batch = partial(augment_and_transform_batch, transform=val_transform, image_processor=image_processor)

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(val_transform_batch)
    test_dataset = dataset["test"].with_transform(val_transform_batch)

    # --- Metrics and Model Initialization ---
    eval_compute_metrics_fn = partial(compute_metrics, image_processor=image_processor, id2label=id2label)
    model = load_model(args.model_checkpoint, id2label, label2id)
    
    # --- Training Arguments ---
    output_dir = f"{args.model_checkpoint.split('/')[-1]}-{args.run_name}-{args.version}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.lr,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        optim='adamw_torch',
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower eval_loss is better
        load_best_model_at_end=True,
        remove_unused_columns=False,
        dataloader_num_workers=max(os.cpu_count() // 2, 1),
        push_to_hub=args.push_to_hub,
    )
    
    # --- Trainer Setup ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        # compute_metrics=eval_compute_metrics_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    # --- Training and Evaluation ---
    print("--- Starting Training ---")
    trainer.train()

    if args.push_to_hub:
        print("\n--- Pushing Best Model to Hugging Face Hub ---")
        trainer.push_to_hub()

#    print("\n--- Evaluating on Test Set ---")
#    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
#    pprint(test_metrics)

#    print("\n--- Training and Evaluation Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an object detection model.")
    parser.add_argument("--base_dir", type=str, default="/home/alvin/aquarium_dataset", help="Base directory for the dataset.")
    parser.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50", help="Model checkpoint from Hugging Face Hub.")
    parser.add_argument("--run_name", type=str, default="dswb_test", help="A name for the training run.")
    parser.add_argument("--version", type=str, default="0.0", help="Version of the training run.")
    parser.add_argument("--max_image_size", type=int, default=800, help="Maximum size for image resizing.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed-precision training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the final model to the Hugging Face Hub.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging with Weights & Biases.")

    args = parser.parse_args()
    main(args)

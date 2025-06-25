import gradio as gr
import subprocess
import sys
import os

def run_training_and_stream_output(
    base_dir, model_checkpoint, run_name, version,
    epochs, lr, batch_size, eval_batch_size,
    gradient_accumulation_steps, max_image_size,
    fp16, use_wandb, push_to_hub
):
    """
    Constructs the training command from the UI inputs and executes it as a subprocess.
    This function yields the standard output and error in real-time to display
    the training progress in the Gradio UI.
    """
    # Use sys.executable to ensure the script is run with the same Python interpreter
    # that is running the Gradio app. This helps avoid environment issues.
    cmd = [sys.executable, "train.py"]

    # --- Argument Construction ---
    # Add string/path arguments to the command
    if base_dir:
        cmd.extend(["--base_dir", base_dir])
    if model_checkpoint:
        cmd.extend(["--model_checkpoint", model_checkpoint])
    if run_name:
        cmd.extend(["--run_name", run_name])
    if version:
        cmd.extend(["--version", version])

    # Add numerical arguments
    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--lr", str(lr)])
    cmd.extend(["--batch_size", str(batch_size)])
    cmd.extend(["--eval_batch_size", str(eval_batch_size)])
    cmd.extend(["--gradient_accumulation_steps", str(gradient_accumulation_steps)])
    cmd.extend(["--max_image_size", str(max_image_size)])

    # Add boolean flags only if they are checked
    if fp16:
        cmd.append("--fp16")
    if use_wandb:
        cmd.append("--use_wandb")
    if push_to_hub:
        cmd.append("--push_to_hub")

    # --- Subprocess Execution ---
    # Start the training script as a subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Redirect stderr to stdout
        text=True,
        encoding='utf-8',
        bufsize=1  # Line-buffered output
    )

    # --- Stream Output to UI ---
    output = ""
    # The `iter` function reads from the process's stdout line by line
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output # Yield the cumulative output to the Gradio textbox
    
    process.stdout.close()
    return_code = process.wait()
    
    # --- Error Handling ---
    if return_code != 0:
        # If the subprocess returns a non-zero exit code, it indicates an error.
        # We construct a detailed error message to help with debugging.
        full_command = " ".join(cmd)
        error_message = (
            f"--- ❌ TRAINING PROCESS FAILED (Exit Code: {return_code}) ---\n\n"
            f"This error usually means there's a problem with the command-line arguments passed to the training script.\n\n"
            f"Full Command Executed:\n"
            f"------------------------\n"
            f"{full_command}\n\n"
            f"Full Output Log:\n"
            f"----------------\n"
            f"{output}"
        )
        raise gr.Error(error_message)
    
    yield "✅ Training complete!"


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🚀 Object Detection Training UI
        This interface allows you to run the modular object detection training pipeline.
        Fill in the parameters below and click "Start Training" to begin.
        The training progress and logs will be displayed in real-time in the output box.
        """
    )
    
    with gr.Row():
        # The main output log will take up the left side of the UI
        output_log = gr.Textbox(
            label="Training Log", 
            lines=30, 
            interactive=False, 
            placeholder="Training progress will appear here..."
        )

        # The controls will be on the right side
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Core Parameters")
            base_dir_input = gr.Textbox(label="Dataset Base Directory", value="./aquarium_dataset", placeholder="/path/to/your/dataset")
            model_checkpoint_input = gr.Textbox(label="Model Checkpoint", value="facebook/detr-resnet-50", placeholder="e.g., facebook/detr-resnet-50")
            run_name_input = gr.Textbox(label="Run Name", value="gradio-ui-test")
            version_input = gr.Textbox(label="Version", value="0.1")
            
            gr.Markdown("## 🧠 Hyperparameters")
            epochs_input = gr.Slider(minimum=1, maximum=100, value=10, step=1, label="Epochs")
            # Removed the deprecated `precision` argument from gr.Number
            lr_input = gr.Number(label="Learning Rate", value=1e-4)
            
            with gr.Accordion("Advanced Hyperparameters", open=False):
                batch_size_input = gr.Slider(minimum=1, maximum=64, value=4, step=1, label="Training Batch Size")
                eval_batch_size_input = gr.Slider(minimum=1, maximum=64, value=4, step=1, label="Evaluation Batch Size")
                grad_accum_input = gr.Slider(minimum=1, maximum=16, value=1, step=1, label="Gradient Accumulation Steps")
                max_image_size_input = gr.Slider(minimum=224, maximum=1333, value=800, step=1, label="Max Image Size")
            
            gr.Markdown("## 🚩 Flags & Options")
            with gr.Row():
                fp16_input = gr.Checkbox(label="Enable FP16", value=True)
                use_wandb_input = gr.Checkbox(label="Use W&B", value=False)
                push_to_hub_input = gr.Checkbox(label="Push to Hub", value=False)

            start_button = gr.Button("Start Training", variant="primary")
    
    # List of all input components for the click handler
    inputs_list = [
        base_dir_input, model_checkpoint_input, run_name_input, version_input,
        epochs_input, lr_input, batch_size_input, eval_batch_size_input,
        grad_accum_input, max_image_size_input,
        fp16_input, use_wandb_input, push_to_hub_input
    ]

    # Connect the button to the training function
    start_button.click(
        fn=run_training_and_stream_output,
        inputs=inputs_list,
        outputs=output_log
    )

if __name__ == "__main__":
    # Launch the Gradio app
    app.launch()

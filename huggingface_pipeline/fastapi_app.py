import sys
import subprocess
import uuid
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multi-Task Training API",
    description="An API to start and monitor Object Detection and ASR training jobs.",
    version="2.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dictionary to store job statuses
jobs: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models for Training Parameters ---

class ObjectDetectionJob(BaseModel):
    base_dir: str
    model_checkpoint: str = "facebook/detr-resnet-50"
    run_name: str = "obj-detection-run"
    version: str = "0.0"
    max_image_size: int = 512
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    epochs: int = 5
    lr: float = 5e-5
    weight_decay: float = 1e-4
    seed: int = 42
    dataloader_num_workers: int = 8
    fp16: bool = False
    use_wandb: bool = False
    push_to_hub: bool = False

class AsrJob(BaseModel):
    data_dir: str
    output_dir: str = "w2v-bert-2.0-en-finetuned-asr"
    model_checkpoint: str = "facebook/w2v-bert-2.0"
    num_train_epochs: float = 5.0
    learning_rate: float = 3e-4
    train_batch_size: int = 16
    eval_batch_size: int = 16
    random_seed: int = 42
    push_to_hub: bool = False
    log_to_wandb: bool = False
    wandb_project: str = ""


# --- Background Training Functions ---

async def run_training_script(job_id: str, script_name: str, args: list):
    """
    Generic function to run a training script in the background.
    """
    jobs[job_id]["status"] = "running"
    
    cmd = [sys.executable, script_name] + args

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode == 0:
        jobs[job_id]["status"] = "completed"
    else:
        jobs[job_id]["status"] = "failed"
    
    jobs[job_id]["log"] = stdout.decode() + stderr.decode()


# --- API Endpoints ---

@app.post("/train/object-detection", status_code=202)
async def start_object_detection_training(job_params: ObjectDetectionJob, background_tasks: BackgroundTasks):
    """
    Starts a new Object Detection training job.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "log": None, "task": "Object Detection"}
    
    args = [
        "--base_dir", job_params.base_dir,
        "--model_checkpoint", job_params.model_checkpoint,
        "--run_name", job_params.run_name,
        "--version", job_params.version,
        "--epochs", str(job_params.epochs),
        "--lr", str(job_params.lr),
        "--batch_size", str(job_params.batch_size),
        "--eval_batch_size", str(job_params.eval_batch_size),
        "--max_image_size", str(job_params.max_image_size),
        "--weight_decay", str(job_params.weight_decay),
        "--seed", str(job_params.seed),
    ]
    if job_params.fp16: args.append("--fp16")
    if job_params.use_wandb: args.append("--use_wandb")
    if job_params.push_to_hub: args.append("--push_to_hub")

    background_tasks.add_task(run_training_script, job_id, "object_detection_train.py", args)
    
    return {"job_id": job_id, "status": "queued"}


@app.post("/train/asr", status_code=202)
async def start_asr_training(job_params: AsrJob, background_tasks: BackgroundTasks):
    """
    Starts a new ASR training job.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "log": None, "task": "ASR"}
    
    args = [
        "--data_dir", job_params.data_dir,
        "--output_dir", job_params.output_dir,
        "--model_checkpoint", job_params.model_checkpoint,
        "--num_train_epochs", str(job_params.num_train_epochs),
        "--learning_rate", str(job_params.learning_rate),
        "--train_batch_size", str(job_params.train_batch_size),
        "--eval_batch_size", str(job_params.eval_batch_size),
        "--random_seed", str(job_params.random_seed),
    ]
    if job_params.push_to_hub: args.append("--push_to_hub")
    if job_params.log_to_wandb:
        args.append("--log_to_wandb")
        if job_params.wandb_project:
            args.extend(["--wandb_project", job_params.wandb_project])

    background_tasks.add_task(run_training_script, job_id, "asr_train.py", args)
    
    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Retrieves the status and log for a given job ID.
    """
    job = jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}, 404
    
    return job

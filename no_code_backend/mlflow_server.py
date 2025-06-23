"""
MLflow server management and access utilities

In Docker mode, this connects to an external MLflow server.
In standalone mode, it starts a local MLflow server.
"""
import os
import subprocess
import time
import mlflow
import atexit
from pathlib import Path

# Default paths - use environment variable if running in Docker
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./logs/mlflow")
MLFLOW_PORT = int(os.environ.get("MLFLOW_PORT", 5000))
DOCKER_MODE = os.environ.get("DOCKER_MODE", "false").lower() == "true"

# MLflow server hostname - in Docker this should be the service name
MLFLOW_HOST = os.environ.get("MLFLOW_HOST", "mlflow" if DOCKER_MODE else "localhost")

# Store the server process globally (only used in standalone mode)
_mlflow_server_process = None

def start_mlflow_server():
    """Start the MLflow UI server if not already running, or connect to existing server in Docker mode"""
    global _mlflow_server_process
    
    # In Docker mode, we don't start a server, just connect to the existing one
    if DOCKER_MODE:
        return f"MLflow Docker mode: connecting to {MLFLOW_TRACKING_URI}"
    
    # Standalone mode - start a local server
    if _mlflow_server_process is not None:
        return f"MLflow server already running at http://{MLFLOW_HOST}:{MLFLOW_PORT}"
    
    # Ensure directory exists
    Path("logs/mlflow").mkdir(parents=True, exist_ok=True)
    
    # Start MLflow server as a subprocess
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", MLFLOW_TRACKING_URI.replace("file:", ""),
        "--default-artifact-root", MLFLOW_TRACKING_URI.replace("file:", ""),
        "--host", "0.0.0.0",
        "--port", str(MLFLOW_PORT)
    ]
    
    try:
        _mlflow_server_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Give the server a moment to start
        time.sleep(2)
        
        # Register shutdown function to close server on exit
        atexit.register(stop_mlflow_server)
        
        return f"MLflow server started at http://{MLFLOW_HOST}:{MLFLOW_PORT}"
    except Exception as e:
        return f"Failed to start MLflow server: {str(e)}"

def stop_mlflow_server():
    """Stop the MLflow server if it's running"""
    global _mlflow_server_process
    
    if DOCKER_MODE:
        return "MLflow server managed by Docker, not stopping"
    
    if _mlflow_server_process is not None:
        _mlflow_server_process.terminate()
        _mlflow_server_process = None
        return "MLflow server stopped"
    
    return "No MLflow server was running"

def get_mlflow_ui_url():
    """Get the URL for the MLflow UI"""
    return f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"

def get_experiment_details():
    """Get details about current experiments"""
    # Set the tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Get experiment by name
        experiment = mlflow.get_experiment_by_name("no-code-ml-experiments")
        if experiment:
            # Get runs for this experiment
            runs = mlflow.search_runs([experiment.experiment_id])
            
            return {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "artifact_location": experiment.artifact_location,
                "tracking_uri": MLFLOW_TRACKING_URI,
                "run_count": len(runs),
            }
        else:
            return {"error": "No experiment found"}
    except Exception as e:
        return {"error": f"Error getting experiment details: {str(e)}"}

def ensure_mlflow_running():
    """Ensure MLflow server is running, starting it if needed"""
    global _mlflow_server_process
    
    if _mlflow_server_process is None or _mlflow_server_process.poll() is not None:
        return start_mlflow_server()
    
    return f"MLflow server already running at http://localhost:{MLFLOW_PORT}"

# Auto-start the server when this module is imported (in standalone mode only)
if not DOCKER_MODE:
    start_mlflow_server()
else:
    print(f"Running in Docker mode, connecting to MLflow at {MLFLOW_TRACKING_URI}")

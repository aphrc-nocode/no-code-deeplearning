#!/bin/bash

# Script to run the Gradio UI for the No-Code AI Platform Backend

echo "Starting Gradio UI for No-Code AI Platform..."
echo "Make sure the FastAPI backend is running on http://localhost:8000"
echo "-------------------------------------------------------------"

# Run the Gradio UI
python3 gradio_ui.py

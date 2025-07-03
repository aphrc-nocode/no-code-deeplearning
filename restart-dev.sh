#!/bin/bash
# Script to restart development environment

# Make sure we're in the correct directory
cd "$(dirname "$0")"

echo "========================================"
echo "Restarting No-Code AI Platform"
echo "========================================"

echo "Stopping containers..."
docker-compose down

echo "Rebuilding containers..."
docker-compose build

echo "Starting containers..."
docker-compose up -d

echo ""
echo "========================================"
echo "Restart complete!"
echo "========================================"
echo "FastAPI app is running at: http://localhost:8000"
echo "MLflow UI is available at: http://localhost:5000"
echo ""
echo "To view logs: docker-compose logs -f"

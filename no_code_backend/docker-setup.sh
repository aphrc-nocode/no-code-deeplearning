#!/bin/bash
# Setup script for Docker deployment (development environment)

# Make sure we're in the correct directory
cd "$(dirname "$0")"

# Print header
echo "========================================"
echo "No-Code AI Platform - Dev Environment"
echo "========================================"

# Create necessary directories if they don't exist
mkdir -p models datasets logs mlruns

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "ERROR: Docker is not running"
  exit 1
fi

echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d
 
# Check if services started properly
if docker-compose ps | grep -q "Up"; then
  echo ""
  echo "========================================"
  echo "Services started successfully!"
  echo "========================================"
  echo "FastAPI app is running at: http://localhost:8000"
  echo "MLflow UI is available at: http://localhost:5000"
  echo ""
  echo "API Documentation: http://localhost:8000/docs"
  echo ""
  echo "To view logs: docker-compose logs -f"
  echo "To stop services: docker-compose down"
else
  echo ""
  echo "========================================"
  echo "ERROR: Some services failed to start"
  echo "========================================"
  echo "Check logs for details: docker-compose logs"
  exit 1
fi

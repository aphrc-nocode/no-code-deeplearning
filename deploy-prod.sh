#!/bin/bash
# Production deployment script

# Make sure we're in the correct directory
cd "$(dirname "$0")"

# Check if running as root (required for port 80/443)
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root to bind to ports 80/443"
  exit 1
fi

# Print header
echo "========================================"
echo "No-Code AI Platform - Production Deploy"
echo "========================================"

# Check for .env.prod file
if [ ! -f .env.prod ]; then
  echo "ERROR: .env.prod file not found"
  echo "Please create this file with your production settings"
  exit 1
fi

# Check for SSL certificates
if [ ! -f nginx/ssl/server.crt ] || [ ! -f nginx/ssl/server.key ]; then
  echo "SSL certificates not found. Generating self-signed certificates..."
  ./generate-certs.sh
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models datasets logs mlruns

# Pull latest images
echo "Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yaml pull

# Build images
echo "Building Docker images..."
docker-compose -f docker-compose.prod.yaml build

# Start services
echo "Starting services..."
docker-compose -f docker-compose.prod.yaml up -d

# Check service health
echo "Checking service health..."
sleep 5

# Check if services are running
if docker-compose -f docker-compose.prod.yaml ps | grep -q "Up"; then
  echo ""
  echo "========================================"
  echo "Deployment successful!"
  echo "========================================"
  echo "API available at: https://localhost/api"
  echo "MLflow UI available at: https://localhost/mlflow"
  echo ""
  echo "To view logs: docker-compose -f docker-compose.prod.yaml logs -f"
  echo "To stop: docker-compose -f docker-compose.prod.yaml down"
else
  echo ""
  echo "========================================"
  echo "ERROR: Some services failed to start"
  echo "========================================"
  echo "Check logs for details: docker-compose -f docker-compose.prod.yaml logs"
  exit 1
fi

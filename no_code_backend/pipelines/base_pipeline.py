"""
Base pipeline interface for all computer vision tasks.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import os
import mlflow

class BasePipeline(ABC):
    """Base class for all ML pipelines"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_id = None
    
    @abstractmethod
    async def train(self, dataset_path: str, job_id: str) -> Dict[str, Any]:
        """Train a model on the given dataset and return metrics"""
        pass
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create a model based on the configuration"""
        pass
    
    @abstractmethod
    def get_transforms(self):
        """Get transforms for the input data"""
        pass
    
    @abstractmethod
    async def predict(self, image, model=None) -> Dict[str, Any]:
        """Make a prediction using the trained model"""
        pass
    
    @abstractmethod
    async def evaluate(self, dataset_path: str) -> Dict[str, Any]:
        """Evaluate the model on a test dataset"""
        pass
    
    @staticmethod
    @abstractmethod
    def get_metrics() -> List[str]:
        """Get the list of metrics supported by this pipeline"""
        pass

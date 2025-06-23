"""
Dataset Versioning and Caching System for No-Code AI Platform

This module provides functionality to:
1. Version datasets by storing metadata and checksums
2. Cache datasets to prevent redundant downloads
3. Track dataset lineage for reproducibility
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetVersion:
    """Represents a single version of a dataset"""
    
    def __init__(
        self, 
        version_id: str,
        dataset_name: str,
        source: str,
        creation_timestamp: float,
        parameters: Dict[str, Any],
        sample_count: int,
        size_bytes: int,
        checksum: str
    ):
        self.version_id = version_id
        self.dataset_name = dataset_name
        self.source = source
        self.creation_timestamp = creation_timestamp
        self.parameters = parameters
        self.sample_count = sample_count
        self.size_bytes = size_bytes
        self.checksum = checksum
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetVersion':
        """Create a DatasetVersion instance from a dictionary"""
        return cls(
            version_id=data["version_id"],
            dataset_name=data["dataset_name"],
            source=data["source"],
            creation_timestamp=data["creation_timestamp"],
            parameters=data["parameters"],
            sample_count=data["sample_count"],
            size_bytes=data["size_bytes"],
            checksum=data["checksum"]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "version_id": self.version_id,
            "dataset_name": self.dataset_name,
            "source": self.source,
            "creation_timestamp": self.creation_timestamp,
            "parameters": self.parameters,
            "sample_count": self.sample_count,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum
        }
    
    @property
    def creation_date(self) -> str:
        """Return a human-readable creation date"""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.creation_timestamp))
    
    @property
    def size_readable(self) -> str:
        """Return a human-readable size"""
        for unit in ["B", "KB", "MB", "GB"]:
            if self.size_bytes < 1024:
                return f"{self.size_bytes:.2f} {unit}"
            self.size_bytes /= 1024
        return f"{self.size_bytes:.2f} TB"


class DatasetVersionManager:
    """Manages dataset versions and caching"""
    
    def __init__(self, base_dir: str = "datasets"):
        self.base_dir = Path(base_dir)
        self.versions_file = self.base_dir / "versions.json"
        self.cache_dir = self.base_dir / "cache"
        
        # Create required directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize versions registry
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Dict[str, Any]]:
        """Load versions registry from file or initialize if it doesn't exist"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading versions file: {str(e)}")
                return {}
        return {}
    
    def _save_versions(self) -> None:
        """Save versions registry to file"""
        with open(self.versions_file, "w") as f:
            json.dump(self.versions, f, indent=2)
    
    def _calculate_checksum(self, dataset_path: Union[str, Path]) -> str:
        """Calculate a checksum for a dataset directory"""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Calculate checksum based on dataset_config.json and class_mapping.json if they exist
        files_to_hash = []
        
        config_file = dataset_path / "dataset_config.json"
        if config_file.exists():
            files_to_hash.append(config_file)
        
        mapping_file = dataset_path / "class_mapping.json"
        if mapping_file.exists():
            files_to_hash.append(mapping_file)
        
        # If no config files found, use some samples
        if not files_to_hash:
            # Find image files for sampling
            image_files = []
            for ext in [".jpg", ".jpeg", ".png"]:
                image_files.extend(dataset_path.glob(f"**/*{ext}"))
            
            # Take a sample of images (max 10)
            if image_files:
                files_to_hash = sorted(image_files)[:10]
        
        # Calculate checksum
        hasher = hashlib.sha256()
        
        for file_path in files_to_hash:
            try:
                with open(file_path, "rb") as f:
                    # Read in chunks to handle large files
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
            except Exception as e:
                logger.warning(f"Error hashing file {file_path}: {str(e)}")
        
        return hasher.hexdigest()
    
    def _calculate_size(self, path: Union[str, Path]) -> int:
        """Calculate the total size of a directory in bytes"""
        path = Path(path)
        if not path.exists():
            return 0
        
        total_size = 0
        if path.is_file():
            return path.stat().st_size
        
        for item in path.glob("**/*"):
            if item.is_file():
                total_size += item.stat().st_size
        
        return total_size
    
    def _count_samples(self, dataset_path: Union[str, Path]) -> int:
        """Count the number of samples (images) in a dataset"""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return 0
        
        image_count = 0
        # Count all image files
        for ext in [".jpg", ".jpeg", ".png"]:
            image_count += len(list(dataset_path.glob(f"**/*{ext}")))
        
        return image_count
    
    def register_dataset(
        self,
        job_id: str,
        dataset_name: str,
        source: str,
        parameters: Dict[str, Any]
    ) -> DatasetVersion:
        """
        Register a dataset version and calculate its metadata
        
        Args:
            job_id: The job ID used as directory name
            dataset_name: Name of the dataset
            source: Source of the dataset (e.g., "huggingface", "upload")
            parameters: Parameters used to create the dataset
            
        Returns:
            DatasetVersion object
        """
        dataset_path = self.base_dir / job_id
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Calculate metadata
        checksum = self._calculate_checksum(dataset_path)
        size_bytes = self._calculate_size(dataset_path)
        sample_count = self._count_samples(dataset_path)
        
        # Create version object
        version = DatasetVersion(
            version_id=job_id,
            dataset_name=dataset_name,
            source=source,
            creation_timestamp=time.time(),
            parameters=parameters,
            sample_count=sample_count,
            size_bytes=size_bytes,
            checksum=checksum
        )
        
        # Store in registry
        self.versions[job_id] = version.to_dict()
        self._save_versions()
        
        return version
    
    def find_similar_dataset(
        self, 
        dataset_name: str,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find a similar dataset in the cache based on name and parameters
        
        Args:
            dataset_name: Name of the dataset to find
            parameters: Parameters used to create the dataset
            
        Returns:
            Job ID of a similar dataset if found, None otherwise
        """
        # Filter versions by dataset name
        candidates = [
            (job_id, data) for job_id, data in self.versions.items()
            if data["dataset_name"] == dataset_name
        ]
        
        if not candidates:
            return None
        
        # Check for parameter match
        for job_id, data in candidates:
            # Check for key parameter matches
            param_match = True
            for key in ["subset", "split", "task_type", "max_samples"]:
                if (key in parameters and key in data["parameters"] and 
                    parameters[key] != data["parameters"][key]):
                    param_match = False
                    break
            
            if param_match:
                # Check if the dataset directory still exists
                if (self.base_dir / job_id).exists():
                    return job_id
        
        return None
    
    def get_dataset_version(self, job_id: str) -> Optional[DatasetVersion]:
        """Get dataset version information by job ID"""
        if job_id in self.versions:
            return DatasetVersion.from_dict(self.versions[job_id])
        return None
    
    def list_dataset_versions(
        self, 
        dataset_name: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[DatasetVersion]:
        """
        List all dataset versions, optionally filtered
        
        Args:
            dataset_name: Filter by dataset name
            source: Filter by source
            
        Returns:
            List of DatasetVersion objects
        """
        versions = []
        
        for job_id, data in self.versions.items():
            if dataset_name and data["dataset_name"] != dataset_name:
                continue
            
            if source and data["source"] != source:
                continue
            
            # Check if directory exists
            if (self.base_dir / job_id).exists():
                versions.append(DatasetVersion.from_dict(data))
        
        # Sort by creation timestamp (newest first)
        return sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
    
    def copy_dataset(self, job_id: str, new_job_id: str) -> Optional[str]:
        """
        Create a copy of a dataset with a new job ID
        
        Args:
            job_id: Source job ID
            new_job_id: New job ID
            
        Returns:
            New job ID if successful, None otherwise
        """
        source_path = self.base_dir / job_id
        target_path = self.base_dir / new_job_id
        
        if not source_path.exists():
            logger.error(f"Source dataset does not exist: {source_path}")
            return None
        
        if target_path.exists():
            logger.error(f"Target path already exists: {target_path}")
            return None
        
        try:
            shutil.copytree(source_path, target_path)
            
            # If source is in registry, copy its entry with the new ID
            if job_id in self.versions:
                version_data = self.versions[job_id].copy()
                version_data["version_id"] = new_job_id
                version_data["creation_timestamp"] = time.time()
                self.versions[new_job_id] = version_data
                self._save_versions()
            
            return new_job_id
        except Exception as e:
            logger.error(f"Error copying dataset: {str(e)}")
            return None
    
    def delete_dataset(self, job_id: str) -> bool:
        """
        Delete a dataset and its version information
        
        Args:
            job_id: Job ID of the dataset to delete
            
        Returns:
            True if successful, False otherwise
        """
        dataset_path = self.base_dir / job_id
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            # Remove from registry if it exists
            if job_id in self.versions:
                del self.versions[job_id]
                self._save_versions()
            return True
        
        try:
            shutil.rmtree(dataset_path)
            
            # Remove from registry
            if job_id in self.versions:
                del self.versions[job_id]
                self._save_versions()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset: {str(e)}")
            return False

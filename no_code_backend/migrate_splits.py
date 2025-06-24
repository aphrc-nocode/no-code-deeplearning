#!/usr/bin/env python3
"""
Utility script to migrate dataset splits from the old location (models/{job_id}/splits)
to the new location (dataset_splits/{job_id}).
"""

import os
import json
import shutil
from pathlib import Path

def migrate_splits():
    """Migrate dataset splits from old location to new location"""
    # Old location: models/{job_id}/splits/
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found. Nothing to migrate.")
        return
    
    # New location: dataset_splits/{job_id}/
    new_base_dir = Path("dataset_splits")
    new_base_dir.mkdir(exist_ok=True)
    
    # Find all job_id directories
    count = 0
    for job_dir in models_dir.iterdir():
        if not job_dir.is_dir():
            continue
        
        # Check if it has a splits directory
        splits_dir = job_dir / "splits"
        if not splits_dir.exists() or not splits_dir.is_dir():
            continue
        
        # Check for dataset_splits.json
        splits_file = splits_dir / "dataset_splits.json"
        if not splits_file.exists():
            continue
        
        # Create new directory
        job_id = job_dir.name
        new_job_dir = new_base_dir / job_id
        new_job_dir.mkdir(exist_ok=True)
        
        # Copy the file
        new_splits_file = new_job_dir / "dataset_splits.json"
        print(f"Migrating {splits_file} -> {new_splits_file}")
        
        # Copy the file
        shutil.copy2(splits_file, new_splits_file)
        
        # Clean up old file (optional, uncomment to remove)
        # splits_file.unlink()
        # If the splits directory is now empty, remove it
        # if not list(splits_dir.iterdir()):
        #     splits_dir.rmdir()
        
        count += 1
    
    if count > 0:
        print(f"Successfully migrated {count} dataset splits.")
    else:
        print("No dataset splits found to migrate.")

if __name__ == "__main__":
    migrate_splits()

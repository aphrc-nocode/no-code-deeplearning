#!/usr/bin/env python3
"""
Test script for the new ASR dataset upload endpoints
"""
import requests
import json
from pathlib import Path


def test_asr_endpoints():
    """Test the new ASR dataset endpoints"""
    base_url = "http://localhost:8000"  # Adjust if running on different port
    
    print("Testing ASR Dataset Endpoints")
    print("=" * 40)
    
    # Test 1: Get supported formats
    print("\n1. Testing supported formats endpoint...")
    try:
        response = requests.get(f"{base_url}/supported-speech-formats")
        if response.status_code == 200:
            print("✅ Supported formats endpoint works")
            data = response.json()
            print(f"   Supported formats: {list(data['formats'].keys())}")
            print(f"   Audio formats: {data['supported_audio_formats']}")
        else:
            print(f"❌ Error: {response.status_code}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 2: Create example dataset
    print("\n2. Testing example dataset creation...")
    try:
        job_id = "test_asr_api"
        params = {
            "train_samples": 5,
            "val_samples": 2,
            "test_samples": 2,
            "language": "en"
        }
        response = requests.post(f"{base_url}/create-example-speech-dataset/{job_id}", params=params)
        if response.status_code == 200:
            print("✅ Example dataset creation works")
            data = response.json()
            print(f"   Created {data['total_samples']} samples")
            print(f"   Dataset format: {data['dataset_format']}")
            print(f"   Splits: {data['splits']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 3: Validate dataset
    print("\n3. Testing dataset validation...")
    try:
        response = requests.get(f"{base_url}/validate-speech-dataset/{job_id}")
        if response.status_code == 200:
            print("✅ Dataset validation works")
            data = response.json()
            validation = data["validation"]
            print(f"   Format: {validation['format']}")
            print(f"   Splits found: {len(validation['splits_found'])}")
            print(f"   Total samples: {validation['total_samples']}")
            if validation["issues"]:
                print(f"   Issues: {validation['issues']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 4: Get dataset info
    print("\n4. Testing dataset info endpoint...")
    try:
        response = requests.get(f"{base_url}/speech-dataset-info/{job_id}")
        if response.status_code == 200:
            print("✅ Dataset info endpoint works")
            data = response.json()
            print(f"   Format: {data['format']}")
            print(f"   Splits: {list(data['splits'].keys())}")
            for split_name, split_info in data['splits'].items():
                print(f"     {split_name}: {split_info['samples']} samples")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    print("\n🎉 All API endpoint tests completed!")
    return True


if __name__ == "__main__":
    print("Note: Make sure the FastAPI server is running (python main.py)")
    print("This test requires an active server connection.\n")
    
    success = test_asr_endpoints()
    if success:
        print("\n✅ ASR API endpoints are working correctly!")
    else:
        print("\n❌ Some tests failed. Check server status and logs.")

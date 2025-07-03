"""
Test script for Speech Recognition integration
"""
import asyncio
from pathlib import Path
import sys

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from main import PipelineConfig, TaskType, ModelArchitecture
from pipelines.speech_recognition_pipeline import SpeechRecognitionPipeline


async def test_speech_pipeline():
    """Test the speech recognition pipeline creation"""
    
    # Create a test configuration
    config = PipelineConfig(
        name="Test ASR Pipeline",
        task_type=TaskType.SPEECH_RECOGNITION,
        architecture=ModelArchitecture.WAV2VEC2_BERT,
        num_classes=10,  # Not used for ASR but required by schema
        batch_size=8,
        epochs=2, 
        learning_rate=3e-4,
        language_code="en",
        language="english",
        target_sampling_rate=16000,
        min_duration_s=1.0,
        max_duration_s=10.0,
        min_transcript_len=5,
        max_transcript_len=100
    )
    
    print("Creating Speech Recognition Pipeline...")
    try:
        pipeline = SpeechRecognitionPipeline(config)
        print("✓ Pipeline created successfully!")
        print(f"  - Language: {pipeline.language}")
        print(f"  - Model: {pipeline.model_checkpoint}")
        print(f"  - Sampling rate: {pipeline.target_sampling_rate}")
        print(f"  - Duration range: {pipeline.min_duration_s}s - {pipeline.max_duration_s}s")
        return True
    except Exception as e:
        print(f"✗ Error creating pipeline: {e}")
        return False


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from pipelines.speech_recognition_pipeline import SpeechRecognitionPipeline
        print("✓ SpeechRecognitionPipeline imported")
    except ImportError as e:
        print(f"✗ Failed to import SpeechRecognitionPipeline: {e}")
        return False
    
    try:
        from pipelines.speech_utils import clean_transcript, add_duration_column
        print("✓ Speech utilities imported")
    except ImportError as e:
        print(f"✗ Failed to import speech utilities: {e}")
        return False
    
    try:
        from pipelines.speech_data_processing import process_prepared_speech_dataset
        print("✓ Speech data processing imported")
    except ImportError as e:
        print(f"✗ Failed to import speech data processing: {e}")
        return False
    
    try:
        from pipelines.speech_model_utils import compute_metrics_fn
        print("✓ Speech model utilities imported")
    except ImportError as e:
        print(f"✗ Failed to import speech model utilities: {e}")
        return False
    
    return True


async def test_csv_dataset_integration():
    """Test CSV dataset integration with speech pipeline"""
    print("\n" + "-" * 30)
    print("Testing CSV Dataset Integration")
    print("-" * 30)
    
    try:
        # Test CSV dataset loading
        from pipelines.speech_utils import detect_dataset_format, load_csv_speech_dataset
        
        # Check if example dataset exists
        dataset_path = "test_csv_dataset"
        if not Path(dataset_path).exists():
            print("⚠️  Example CSV dataset not found. Create it with:")
            print("python prepare_csv_speech_dataset.py create-example test_csv_dataset")
            return False
        
        # Test format detection
        format_detected = detect_dataset_format(dataset_path)
        print(f"Dataset format detected: {format_detected}")
        
        if format_detected != "csv":
            print("❌ CSV format not detected correctly")
            return False
        
        # Test dataset loading
        dataset = load_csv_speech_dataset(
            dataset_path=dataset_path,
            target_sampling_rate=16000
        )
        
        print(f"✓ Loaded dataset with splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} examples")
        
        # Test basic speech pipeline configuration compatibility
        config = PipelineConfig(
            name="CSV Test Pipeline",
            task_type=TaskType.SPEECH_RECOGNITION,
            architecture=ModelArchitecture.WAV2VEC2_BERT,
            num_classes=10,  # Not used for ASR
            batch_size=2,
            epochs=1,
            learning_rate=3e-4,
            target_sampling_rate=16000,
            min_duration_s=0.5,
            max_duration_s=5.0
        )
        
        pipeline = SpeechRecognitionPipeline(config)
        print("✓ Pipeline created for CSV dataset")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV integration test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("=" * 50)
    print("Speech Recognition Integration Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed!")
        return
    
    print("\n✅ All imports successful!")
    
    # Test pipeline creation
    print("\n" + "-" * 30)
    pipeline_success = await test_speech_pipeline()
    
    # Test CSV dataset integration
    csv_success = await test_csv_dataset_integration()
    
    overall_success = pipeline_success and csv_success
    
    if overall_success:
        print("\n🎉 All tests passed! Speech recognition with CSV support is ready to use.")
    else:
        print("\n❌ Some tests failed!")
    
    print("\nNext steps:")
    print("1. Install speech dependencies: pip install librosa soundfile evaluate jiwer accelerate")
    print("2. Prepare your speech dataset in CSV metadata format:")
    print("   dataset/")
    print("     train/metadata.csv (columns: file_name, sentence)")
    print("     train/audio1.wav")
    print("     validation/metadata.csv")
    print("     validation/audio2.wav")
    print("     ...")
    print("3. Create a speech recognition pipeline via the Gradio UI")
    print("4. Upload your audio dataset and start training!")
    print("5. The pipeline auto-detects CSV vs audiofolder format")


if __name__ == "__main__":
    asyncio.run(main())

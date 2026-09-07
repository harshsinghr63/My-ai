#!/usr/bin/env python3
"""
Script to download the Kronos-base model from Hugging Face
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_kronos_base():
    """Download Kronos-base model and tokenizer"""
    
    model_name = "NeoQuasar/Kronos-base"
    models_dir = Path(__file__).parent.parent / "models" / "kronos-base"
    
    # Create models directory if it doesn't exist
    models_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    print(f"Saving to: {models_dir}")
    
    try:
        # Download model using huggingface_hub
        snapshot_download(
            repo_id=model_name,
            local_dir=str(models_dir),
            repo_type="model"
        )
        
        print(f"\n✓ Model downloaded successfully to {models_dir}")
        
        # Verify the download
        print("\nVerifying model files...")
        if (models_dir / "pytorch_model.bin").exists() or (models_dir / "model.safetensors").exists():
            print("✓ Model weights found")
        
        if (models_dir / "tokenizer.json").exists() or (models_dir / "tokenizer.model").exists():
            print("✓ Tokenizer found")
        
        print("\n✓ Download completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {str(e)}")
        print("Make sure you have internet connection and sufficient disk space")
        raise

if __name__ == "__main__":
    download_kronos_base()

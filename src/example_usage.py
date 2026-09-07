#!/usr/bin/env python3
"""
Example usage of Kronos-base model
"""

from model_loader import load_kronos_base

def main():
    """Example: Generate text with Kronos-base"""
    
    # Load the model
    print("Loading Kronos-base model...\n")
    model = load_kronos_base(use_cuda=True)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}\n")
    
    try:
        generated_text = model.generate(
            prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9
        )
        print(f"Generated text:\n{generated_text}\n")
    except Exception as e:
        print(f"Error during generation: {str(e)}")

if __name__ == "__main__":
    main()

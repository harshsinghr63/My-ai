"""
Model loader utility for Kronos-base
"""

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class KronosBaseLoader:
    """Utility class to load and manage Kronos-base model"""
    
    def __init__(self, model_path=None, use_cuda=True):
        """
        Initialize the model loader
        
        Args:
            model_path: Path to local model directory (optional)
            use_cuda: Whether to use CUDA if available
        """
        self.model_path = model_path or self._get_default_model_path()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"
        self.model = None
        self.tokenizer = None
    
    @staticmethod
    def _get_default_model_path():
        """Get default model path"""
        return Path(__file__).parent.parent / "models" / "kronos-base"
    
    def load(self):
        """Load the model and tokenizer"""
        try:
            print(f"Loading model from {self.model_path}...")
            print(f"Using device: {self.device}")
            
            # Try loading from local path first, fallback to Hugging Face
            if Path(self.model_path).exists():
                model_source = str(self.model_path)
            else:
                model_source = "NeoQuasar/Kronos-base"
                print(f"Local model not found, loading from Hugging Face: {model_source}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                torch_dtype=torch.float16 if self.use_cuda else torch.float32,
                device_map="auto" if self.use_cuda else None
            )
            
            if not self.use_cuda:
                self.model = self.model.to(self.device)
            
            print("✓ Model loaded successfully!")
            return self
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            raise
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """
        Generate text using the model
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text string
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return "Model not loaded"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_class": self.model.__class__.__name__,
            "total_parameters": total_params,
            "device": self.device,
            "tokenizer_vocab_size": len(self.tokenizer)
        }


def load_kronos_base(model_path=None, use_cuda=True):
    """
    Convenience function to load Kronos-base model
    
    Args:
        model_path: Optional path to local model
        use_cuda: Whether to use CUDA
    
    Returns:
        Loaded KronosBaseLoader instance
    """
    loader = KronosBaseLoader(model_path=model_path, use_cuda=use_cuda)
    return loader.load()

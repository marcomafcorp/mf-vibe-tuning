import os
import json
import torch

# Try to import LoRA-related dependencies
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import (
        prepare_model_for_kbit_training,
        LoraConfig, 
        get_peft_model,
        PeftModel,
        TaskType
    )
    LORA_DEPS_AVAILABLE = True
except ImportError:
    LORA_DEPS_AVAILABLE = False

class LoRAImplementation:
    def __init__(self, model_id=None, lora_config_path="lora_config.json", device="auto"):
        """
        Initialize LoRA/QLoRA implementation
        
        Args:
            model_id: Hugging Face model ID or local path
            lora_config_path: Path to LoRA config JSON
            device: Device to use (auto, cuda, cpu)
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.lora_config = self._load_config(lora_config_path)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
    
    def _load_config(self, config_path):
        """Load LoRA config from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default config
            return {
                "adapter_type": "lora",
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"]
            }
    
    def save_config(self, config_dict, config_path="lora_config.json"):
        """Save updated LoRA config to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.lora_config = config_dict
        print(f"LoRA config saved to {config_path}")
    
    def load_base_model(self, model_id=None, quantization=None):
        """
        Load the base model for vibe-tuning
        
        Args:
            model_id: Model ID to load (overrides the one set in init)
            quantization: None for full precision, '4bit' or '8bit' for QLoRA
        """
        if model_id:
            self.model_id = model_id
        
        if not self.model_id:
            raise ValueError("No model ID provided")
        
        print(f"Loading model: {self.model_id}")
        
        # Check for different model ID formats
        if self.model_id.startswith("hf_direct:"):
            # This is a direct HF model ID that was already mapped by the API
            # Just strip the prefix and use it directly
            actual_model_id = self.model_id.replace("hf_direct:", "")
            print(f"Using direct Hugging Face model: {actual_model_id}")
            
        elif self.model_id.startswith("ollama:"):
            # For Ollama models, we need to map to an equivalent Hugging Face model
            # Strip the 'ollama:' prefix
            ollama_model_name = self.model_id.replace("ollama:", "")
            print(f"Using Ollama model: {ollama_model_name}")
            
            # Extract base model name before any colon or version number
            # e.g., "llama3.2:1b" -> "llama3"
            parts = ollama_model_name.split(":")
            base_part = parts[0]  # Before any colon
            base_model_name = base_part.split(".")[0]  # Take only the main model name without version
            print(f"Extracted base model name: {base_model_name}")
            
            # Map to reliable open-source models only
            if base_model_name.startswith("llama3"):
                actual_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"Mapped {ollama_model_name} to {actual_model_id}")
            elif base_model_name.startswith("llama2"):
                actual_model_id = "openlm-research/open_llama_3b_v2"
                print(f"Mapped {ollama_model_name} to {actual_model_id}")
            elif base_model_name.startswith("mistral"):
                actual_model_id = "eachadea/legacy-mistral-7b-v0.1"
                print(f"Mapped {ollama_model_name} to {actual_model_id}")
            elif base_model_name.startswith("phi"):
                actual_model_id = "microsoft/phi-1_5"
                print(f"Mapped {ollama_model_name} to {actual_model_id}")
            else:
                # Default to TinyLlama for unknown models
                actual_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"No mapping for {ollama_model_name}, using default: {actual_model_id}")
        else:
            # For other model IDs (like direct Hugging Face IDs), use as is
            actual_model_id = self.model_id
            print(f"Using model ID as-is: {actual_model_id}")
        
        # Set up quantization for QLoRA if requested
        if quantization == '4bit':
            print("Using 4-bit quantization (QLoRA)")
            # Check if we're on Apple Silicon
            is_apple_silicon = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if is_apple_silicon:
                print("Detected Apple Silicon - using modified 4-bit config")
                # On Apple Silicon, we need to modify the quantization approach
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float32,  # Use float32 on Apple Silicon
                    bnb_4bit_use_double_quant=False,       # Disable double quantization for compatibility
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage=torch.float32   # Use float32 for storage
                )
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
        elif quantization == '8bit':
            print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            print("Using full precision (standard LoRA)")
            quantization_config = None
        
        try:
            # Load tokenizer
            print(f"Loading tokenizer for model: {actual_model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(actual_model_id)
            
            # Load model with quantization if specified
            print(f"Loading model from: {actual_model_id}")
            try:
                # Extra memory optimization settings
                print("Using maximum memory optimization settings")
                # Only call cuda.empty_cache() if CUDA is available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    print("CUDA not available, skipping VRAM cleanup")
                
                # For 4-bit quantization (most memory efficient)
                if quantization == '4bit':
                    print("Loading with 4-bit quantization for maximum memory efficiency")
                    # Check if we're on Apple Silicon
                    is_apple_silicon = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                    if is_apple_silicon:
                        print("Detected Apple Silicon - using modified 4-bit config")
                        # On Apple Silicon, we need to modify the quantization approach
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float32,  # Use float32 on Apple Silicon
                            bnb_4bit_use_double_quant=False,       # Disable double quantization for compatibility
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_quant_storage=torch.float32   # Use float32 for storage
                        )
                    else:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        actual_model_id,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,  # Allow remote code execution for some models
                        offload_folder="offload",  # Enable CPU offloading
                        low_cpu_mem_usage=True,   # Lower CPU memory usage
                    )
                    
                    # Prepare model for k-bit training
                    self.model = prepare_model_for_kbit_training(self.model)
                    
                # For 8-bit quantization (less memory efficient than 4-bit)
                elif quantization == '8bit':
                    print("Loading with 8-bit quantization")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        actual_model_id,
                        quantization_config=quantization_config,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        offload_folder="offload",
                        low_cpu_mem_usage=True,
                    )
                    
                    # Prepare model for k-bit training
                    self.model = prepare_model_for_kbit_training(self.model)
                    
                # Standard precision (most memory intensive)
                else:
                    print("WARNING: Using full precision will require more GPU memory")
                    # Still try to optimize memory usage
                    self.model = AutoModelForCausalLM.from_pretrained(
                        actual_model_id,
                        torch_dtype=torch.float16,  # Use half precision
                        device_map="auto",
                        trust_remote_code=True,
                        offload_folder="offload",  # Enable CPU offloading
                        low_cpu_mem_usage=True,    # Lower CPU memory usage
                    )
            except Exception as load_error:
                print(f"Error loading model with float16: {load_error}")
                print("Trying to load model with alternative approach...")
                
                # Try alternative loading approach
                # torch is already imported at the top of the file
                if quantization:
                    # For Apple Silicon, try a different approach
                    is_apple_silicon = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                    if is_apple_silicon and quantization == '4bit':
                        print("Apple Silicon detected - trying alternative approach without quantization")
                        # On Apple Silicon, fall back to non-quantized model if 4-bit fails
                        self.model = AutoModelForCausalLM.from_pretrained(
                            actual_model_id,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            actual_model_id,
                            quantization_config=quantization_config,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                    # Move to device after loading
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
                    
                    # Prepare model for k-bit training
                    self.model = prepare_model_for_kbit_training(self.model)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        actual_model_id,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    # Move to device after loading
                    if self.device == "cuda":
                        self.model = self.model.to(self.device)
            
            # Handle special tokens if not in tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model {self.model_id}: {str(e)}")
    
    def add_lora_adapter(self, custom_config=None):
        """
        Add LoRA adapter to the model
        
        Args:
            custom_config: Optional dictionary to override the loaded config
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_base_model first.")
        
        # Use custom config if provided, otherwise use loaded config
        config = custom_config if custom_config else self.lora_config
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=config.get("rank", 8),  # Rank
            lora_alpha=config.get("alpha", 16),  # Alpha parameter
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=config.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA adapter
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
        
        return self.model
    
    def print_trainable_parameters(self):
        """Print information about trainable parameters"""
        if self.model is None:
            print("Model not loaded.")
            return
        
        # Get total and trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Percentage of parameters being fine-tuned: {100 * trainable_params / total_params:.2f}%")
    
    def save_adapter(self, output_dir):
        """Save the trained LoRA adapter"""
        if self.model is None:
            raise ValueError("Model not loaded.")
            
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer as well
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
            
        print(f"LoRA adapter saved to {output_dir}")
    
    def load_adapter(self, adapter_path, model_id=None):
        """
        Load a saved adapter
        
        Args:
            adapter_path: Path to the saved adapter
            model_id: Base model ID (if different from the one used during training)
        """
        if model_id:
            self.model_id = model_id
            
        if not self.model_id:
            raise ValueError("No model ID provided")
            
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        print(f"Adapter loaded from {adapter_path}")
        
        return self.model, self.tokenizer
    
    def merge_and_export(self):
        """
        Merge the LoRA adapter with the base model and return the merged model
        
        Returns:
            The merged model that can be saved as a standard transformers model
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Check if the model has LoRA adapters
        if not hasattr(self.model, 'merge_and_unload'):
            raise ValueError("Model does not have LoRA adapters applied.")
        
        print("Merging LoRA adapter with base model...")
        
        # Merge the adapter weights into the base model
        # This creates a standard transformers model without any adapters
        merged_model = self.model.merge_and_unload()
        
        print("LoRA adapter successfully merged with base model")
        
        # Return the merged model
        return merged_model


# Example usage if run directly
if __name__ == "__main__":
    # Load a test model and apply LoRA
    lora_impl = LoRAImplementation()
    
    # Get LoRA config
    print(f"LoRA config: {lora_impl.lora_config}")
    
    # Example to update config
    custom_config = {
        "adapter_type": "lora",
        "rank": 16,  # Higher rank
        "alpha": 32,  # Higher alpha
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    # Save custom config
    lora_impl.save_config(custom_config)
    
    print("LoRA implementation ready")

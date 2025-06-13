from flask import Flask, request, jsonify
import os
import threading
import time
import datetime
import sys
import codecs
import json
from flask import Flask, request, jsonify

# Set UTF-8 as the default encoding for stdout/stderr to handle Unicode characters
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Helper function for generating timestamps
def get_formatted_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Flag to track if torch is available
TORCH_AVAILABLE = False
LORA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    from lora_implementation import LoRAImplementation
    LORA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies are not available: {e}")
    print("LoRA functionality will be limited. Please install the missing packages.")
    print("Try running: pip install torch transformers peft bitsandbytes")

# Initialize Flask app
app = Flask(__name__)

# Global variables to store training state and control training process
training_state = {
    'status': 'idle',  # 'idle', 'training', 'completed', 'error', 'aborted'
    'progress': 0,
    'logs': [],
    'adapter_id': None,
    'error': None,
    'loss': None
}

# Flag to signal training should be aborted
abort_training = False

# Path to store model-adapter mappings
MODEL_ADAPTER_MAPPING_FILE = 'model_adapter_mappings.json'

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize LoRA implementation
if LORA_AVAILABLE:
    lora_impl = LoRAImplementation()
else:
    lora_impl = None

@app.route('/api/lora/config', methods=['GET'])
def get_lora_config():
    """Get current LoRA configuration"""
    try:
        with open('lora_config.json', 'r') as f:
            config = json.load(f)
        return jsonify({"status": "success", "config": config})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/config', methods=['POST'])
def update_lora_config():
    """Update LoRA configuration"""
    try:
        config_data = request.json
        
        # Validate required fields
        required_fields = ["adapter_type", "rank", "alpha", "dropout", "target_modules"]
        for field in required_fields:
            if field not in config_data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400
        
        # Save config
        lora_impl.save_config(config_data)
        
        return jsonify({"status": "success", "message": "LoRA configuration updated", "config": config_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/available_targets', methods=['GET'])
def get_available_targets():
    """Get available target modules for LoRA"""
    # Common target modules for various model architectures
    available_targets = {
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        "default": ["query", "key", "value", "dense"]
    }
    
    return jsonify({"status": "success", "available_targets": available_targets})

@app.route('/api/lora/hardware', methods=['GET'])
def get_lora_hardware_info():
    """
    Endpoint to get hardware information for the LoRA training frontend
    """
    try:
        import torch
        import platform
        import subprocess
        import re
        import os
        
        # Initialize GPU info structure
        gpu_info = {
            'available': False,
            'devices': []
        }
        
        # Check if we're on macOS
        is_macos = platform.system() == 'Darwin'
        
        if is_macos:
            # First check for PyTorch MPS (Metal Performance Shaders) support
            # This is the most reliable way to detect GPU support on Apple Silicon
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if mps_available:
                print("MPS (Metal Performance Shaders) is available - Apple Silicon GPU detected")
                gpu_info['available'] = True
                
                # Try to get more detailed information about the GPU
                try:
                    # Check if we're on Apple Silicon
                    is_apple_silicon = platform.processor() == '' or 'arm' in platform.processor().lower()
                    
                    # Get model name from sysctl
                    model_name = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                    print(f"CPU Model: {model_name}")
                    
                    # Check for Metal support and get GPU details using system_profiler
                    gpu_output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], text=True)
                    
                    # Try to get GPU name
                    gpu_name_match = re.search(r'Chipset Model: (.+)', gpu_output)
                    gpu_name = gpu_name_match.group(1) if gpu_name_match else 'Apple Silicon GPU'
                    print(f"Detected GPU: {gpu_name}")
                    
                    # Try to get VRAM info if available
                    vram_match = re.search(r'VRAM \(Total\): (\d+) MB', gpu_output)
                    total_memory_gb = round(int(vram_match.group(1)) / 1024, 2) if vram_match else None
                    
                    # If we can't find VRAM directly, try to get shared memory info
                    if total_memory_gb is None:
                        shared_match = re.search(r'Shared Memory: (\d+) MB', gpu_output)
                        if shared_match:
                            total_memory_gb = round(int(shared_match.group(1)) / 1024, 2)
                    
                    # If we still can't find memory info, try nvram as a last resort
                    if total_memory_gb is None and is_apple_silicon:
                        try:
                            # Try to get memory info from nvram
                            nvram_output = subprocess.check_output(['nvram', '-p'], text=True, stderr=subprocess.DEVNULL)
                            memory_values = re.findall(r'gpu-memory-info.*?(\d+)', nvram_output)
                            if memory_values:
                                # Use the largest value as an approximation of GPU memory in MB
                                largest_value = max([int(val) for val in memory_values])
                                total_memory_gb = round(largest_value / 1024, 2)
                                print(f"Found GPU memory from nvram: {total_memory_gb} GB")
                        except Exception as e:
                            print(f"Error getting nvram info: {e}")
                    
                    # If we still don't have memory info, estimate based on total system memory
                    if total_memory_gb is None:
                        # Get total system memory and estimate GPU memory as a portion of it
                        import psutil
                        system_memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
                        # Apple Silicon typically allocates a portion of system memory to GPU
                        # Use a conservative estimate of 25% of system memory
                        total_memory_gb = round(system_memory_gb * 0.25, 2)
                        print(f"Estimated GPU memory: {total_memory_gb} GB (25% of {system_memory_gb} GB system memory)")
                    
                    # Add the device to the list
                    gpu_info['devices'].append({
                        'id': 0,
                        'name': gpu_name,
                        'total_memory_gb': total_memory_gb,
                        'used_memory_gb': 0.0,  # Not easily available on macOS
                        'compute_capability': 'Metal/MPS',
                        'is_apple_silicon': is_apple_silicon
                    })
                    
                except Exception as e:
                    print(f"Error getting detailed Apple GPU info: {e}")
                    # Fall back to basic info
                    gpu_info['devices'].append({
                        'id': 0,
                        'name': 'Apple MPS Device',
                        'total_memory_gb': 8.0,  # Default reasonable estimate for modern Macs
                        'used_memory_gb': 0.0,
                        'compute_capability': 'MPS'
                    })
            else:
                # Try fallback methods for older Macs with Metal support
                try:
                    gpu_output = subprocess.check_output(['system_profiler', 'SPDisplaysDataType'], text=True)
                    metal_supported = 'Metal: Supported' in gpu_output
                    
                    if metal_supported:
                        print("Metal is supported but MPS is not available")
                        gpu_info['available'] = True
                        
                        # Try to get GPU name
                        gpu_name_match = re.search(r'Chipset Model: (.+)', gpu_output)
                        gpu_name = gpu_name_match.group(1) if gpu_name_match else 'Intel GPU'
                        
                        # Add the device with estimated memory
                        gpu_info['devices'].append({
                            'id': 0,
                            'name': gpu_name,
                            'total_memory_gb': 2.0,  # Conservative estimate for older Macs
                            'used_memory_gb': 0.0,
                            'compute_capability': 'Metal'
                        })
                except Exception as e:
                    print(f"Error in fallback Metal detection: {e}")
                    # No GPU support detected
        else:
            # Standard CUDA detection for non-macOS systems
            gpu_info['available'] = torch.cuda.is_available()
            
            if gpu_info['available']:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)
                    # Convert bytes to GB for memory
                    total_memory_gb = round(props.total_memory / (1024**3), 2)
                    # Get current allocated memory
                    allocated_memory = torch.cuda.memory_allocated(i)
                    used_memory_gb = round(allocated_memory / (1024**3), 2)
                    
                    gpu_info['devices'].append({
                        'id': i,
                        'name': props.name,
                        'total_memory_gb': total_memory_gb,
                        'used_memory_gb': used_memory_gb,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
        
        # Get CPU information
        import psutil
        import platform
        
        cpu_info = {
            'count': psutil.cpu_count(logical=True),
            'physical_count': psutil.cpu_count(logical=False),
            'model': platform.processor(),
            'usage_percent': psutil.cpu_percent(interval=0.1)
        }
        
        # Get RAM information
        ram = psutil.virtual_memory()
        ram_info = {
            'total_gb': round(ram.total / (1024**3), 2),
            'used_gb': round(ram.used / (1024**3), 2),
            'available_gb': round(ram.available / (1024**3), 2),
            'percent_used': ram.percent
        }
        
        # Return all hardware info
        return jsonify({
            'status': 'success',
            'hardware': {
                'gpu': gpu_info,
                'cpu': cpu_info,
                'ram': ram_info
            }
        })
    except ImportError as e:
        return jsonify({
            'status': 'error',
            'message': f'Missing required package: {str(e)}',
            'hardware': {
                'gpu': {'available': False, 'devices': []},
                'cpu': {},
                'ram': {}
            }
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'hardware': {
                'gpu': {'available': False, 'devices': []},
                'cpu': {},
                'ram': {}
            }
        }), 500

@app.route('/api/lora/apply', methods=['POST'])
def apply_lora():
    """Apply LoRA to a model"""
    try:
        data = request.json
        print(f"DEBUG - apply_lora received data: {data}")
        model_id = data.get('model_id')
        quantization = data.get('quantization')  # None, '4bit', or '8bit'
        
        if not model_id:
            print("ERROR - Missing model_id in apply_lora request")
            return jsonify({"status": "error", "message": "Missing model_id"}), 400
        
        print(f"DEBUG - apply_lora processing model_id: {model_id}, quantization: {quantization}")
        
        # Normalize the model ID and handle different formats
        sanitized_model_id = model_id
        
        # First, check if we need to add 'ollama:' prefix
        if not model_id.startswith('ollama:') and not model_id.startswith('huggingface:'):
            # Assume this is an Ollama model and add the prefix
            print(f"DEBUG - Adding 'ollama:' prefix to model_id: {model_id}")
            sanitized_model_id = f"ollama:{model_id}"
        
        # Check if the model ID contains colons (problematic for HF models)
        if ':' in sanitized_model_id and sanitized_model_id.startswith('ollama:'):
            print(f"DEBUG - Detected colon in model ID: {sanitized_model_id}")
            base_name = sanitized_model_id.split(':')[1].split('.')[0]
            print(f"DEBUG - Using base name: {base_name}")
            
            # Map common models to safe IDs (without colons)
            # IMPORTANT: Don't use 'huggingface:' prefix here, we'll handle that in load_base_model
            if base_name.startswith('llama3'):
                print(f"DEBUG - Mapping to TinyLlama")
                # Use a guaranteed safe model ID
                sanitized_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif base_name.startswith('llama2'):
                print(f"DEBUG - Mapping to OpenLLaMA")
                sanitized_model_id = "openlm-research/open_llama_3b_v2"
            elif base_name.startswith('mistral'):
                print(f"DEBUG - Mapping to open mistral")
                sanitized_model_id = "eachadea/legacy-mistral-7b-v0.1"
            else:
                print(f"DEBUG - Using default TinyLlama model")
                sanitized_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                
            # Now prefix with a marker to know this is a direct HF model
            sanitized_model_id = f"hf_direct:{sanitized_model_id}"
        
        print(f"DEBUG - Final sanitized model_id: {sanitized_model_id}")
            
        # Load model
        try:
            print(f"DEBUG - Calling load_base_model with {sanitized_model_id}")
            lora_impl.load_base_model(sanitized_model_id, quantization)
            print("DEBUG - Model loaded successfully")
        except Exception as model_error:
            print(f"ERROR - Failed to load model: {model_error}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"Failed to load model: {str(model_error)}"}), 500
        
        # Apply LoRA adapter with current config
        try:
            print("DEBUG - Applying LoRA adapter")
            lora_impl.add_lora_adapter()
            print("DEBUG - LoRA adapter applied successfully")
        except Exception as adapter_error:
            print(f"ERROR - Failed to apply LoRA adapter: {adapter_error}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"Failed to apply LoRA adapter: {str(adapter_error)}"}), 500
        
        # Return trainable parameters info
        print("DEBUG - Calculating parameter statistics")
        total_params = sum(p.numel() for p in lora_impl.model.parameters())
        trainable_params = sum(p.numel() for p in lora_impl.model.parameters() if p.requires_grad)
        percentage = 100 * trainable_params / total_params
        
        print(f"DEBUG - LoRA applied successfully. Trainable: {trainable_params} ({percentage:.4f}%)")
        
        return jsonify({
            "status": "success", 
            "message": "LoRA adapter applied successfully",
            "model_info": {
                "base_model_name": model_id,
                "model_type": lora_impl.model.config.model_type,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "trainable_percentage": percentage
            }
        })
    except Exception as e:
        print(f"ERROR - Unhandled exception in apply_lora: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route('/api/lora/train', methods=['POST'])
def train_lora():
    """Train model with LoRA"""
    try:
        global training_state
        
        print("=== LoRA Training API called ===")
        
        # Reset training state
        training_state = {
            'status': 'initializing',
            'progress': 0,
            'logs': [{
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                'message': 'Initializing training...', 
                'level': 'info'
            }],
            'adapter_id': None,
            'error': None,
            'loss': None
        }
        print("Training state reset to initializing")
        
        data = request.json
        model_id = data.get('model_id')
        dataset = data.get('dataset')
        training_args = data.get('training_args', {})
        output_dir = data.get('output_dir', 'lora_adapter')
        
        # Default to 4-bit quantization to reduce memory usage
        quantization = data.get('quantization', '4bit')
        
        # Set default smaller batch size to reduce memory usage
        if 'batch_size' not in training_args:
            training_args['batch_size'] = 1
            
        # Set smaller max_length to reduce memory usage
        if 'max_length' not in training_args:
            training_args['max_length'] = 256
            
        print(f"Using quantization: {quantization}")
        print(f"Using batch size: {training_args['batch_size']}")
        print(f"Using max length: {training_args['max_length']}")
        
        if not model_id:
            training_state['status'] = 'error'
            training_state['error'] = 'Missing model_id'
            return jsonify({"status": "error", "message": "Missing model_id"}), 400
        if not dataset:
            training_state['status'] = 'error'
            training_state['error'] = 'Missing dataset'
            return jsonify({"status": "error", "message": "Missing dataset"}), 400
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate dataset format before starting training
        print(f"Validating dataset: type={type(dataset)}, length={len(dataset) if hasattr(dataset, '__len__') else 'unknown'}")
        
        if not dataset or len(dataset) == 0:
            return jsonify({"status": "error", "message": "Empty dataset. Please provide training data."}), 400
            
        # Show first example for debugging
        if len(dataset) > 0:
            print(f"First dataset example: {dataset[0]}")
            
            # Check if dataset contains required fields
            if isinstance(dataset[0], dict):
                if not any(key in dataset[0] for key in ['instruction', 'prompt', 'question', 'input', 'text']):
                    return jsonify({
                        "status": "error", 
                        "message": "Dataset format error. Each example must contain at least one of: 'instruction', 'prompt', 'question', 'input', or 'text'."
                    }), 400
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=train_model_thread,
            args=(model_id, dataset, training_args, output_dir, quantization)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "success", 
            "message": "LoRA training started",
            "output_dir": output_dir
        })
    except Exception as e:
        training_state['status'] = 'error'
        training_state['error'] = str(e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/training_status', methods=['GET'])
def get_training_status():
    """Get LoRA training status"""
    try:
        return jsonify({
            "status": training_state['status'],
            "progress": training_state['progress'],
            "logs": training_state['logs'][-10:] if training_state['logs'] else [],
            "adapter_id": training_state['adapter_id'],
            "error": training_state['error'],
            "loss": training_state['loss']
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/abort', methods=['POST'])
def abort_training_endpoint():
    """Abort ongoing LoRA training"""
    try:
        global abort_training, training_state
        
        # Set the abort flag to signal training should stop
        abort_training = True
        
        # Update training state
        training_state['status'] = 'aborted'
        training_state['logs'].append({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': 'Training aborted by user request',
            'level': 'warning'
        })
        
        print("=== Training abort requested by user ===")
        
        return jsonify({
            "status": "success",
            "message": "Training abort signal sent. The process will terminate shortly."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def train_model_thread(model_id, dataset, training_args, output_dir, quantization='none'):
    """Function to run training in a separate thread"""
    global training_state, abort_training
    
    # Reset the abort flag when starting a new training session
    abort_training = False
    
    print(f"=== Starting training thread ===")
    print(f"Model ID: {model_id}")
    print(f"Dataset size: {len(dataset) if dataset else 'No dataset'}")
    print(f"Training args: {training_args}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {quantization}")
    
    try:
        # Set training status
        training_state['status'] = 'training'
        training_state['logs'].append({
            'timestamp': get_formatted_timestamp(),
            'message': f'Loading model {model_id}...',
            'level': 'info'
        })
        
        # Process the model ID to handle special characters
        sanitized_model_id = model_id
        
        # First, check if we need to add 'ollama:' prefix
        if not model_id.startswith('ollama:') and not model_id.startswith('huggingface:'):
            # Assume this is an Ollama model and add the prefix
            print(f"DEBUG - Adding 'ollama:' prefix to model_id: {model_id}")
            sanitized_model_id = f"ollama:{model_id}"
        
        # Check if the model ID contains colons (problematic for HF models)
        if ':' in sanitized_model_id and sanitized_model_id.startswith('ollama:'):
            print(f"DEBUG - Detected colon in model ID: {sanitized_model_id}")
            base_name = sanitized_model_id.split(':')[1].split('.')[0]
            print(f"DEBUG - Using base name: {base_name}")
            
            # Map common models to safe IDs (without colons)
            if base_name.startswith('llama3'):
                print(f"DEBUG - Mapping to TinyLlama")
                sanitized_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            elif base_name.startswith('llama2'):
                print(f"DEBUG - Mapping to OpenLLaMA")
                sanitized_model_id = "openlm-research/open_llama_3b_v2"
            elif base_name.startswith('mistral'):
                print(f"DEBUG - Mapping to open mistral")
                sanitized_model_id = "eachadea/legacy-mistral-7b-v0.1"
            else:
                print(f"DEBUG - Using default TinyLlama model")
                sanitized_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                
            # Now prefix with a marker to know this is a direct HF model
            sanitized_model_id = f"hf_direct:{sanitized_model_id}"
        
        print(f"DEBUG - Final sanitized model_id for training: {sanitized_model_id}")
            
        # Initialize LoRA implementation
        lora_impl = LoRAImplementation(model_id=sanitized_model_id)
        
        # Load base model with quantization if specified
        if quantization != 'none':
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': f'Using {quantization} quantization...',
                'level': 'info'
            })
        
        lora_impl.load_base_model(quantization=None if quantization == 'none' else quantization)
        
        # Apply LoRA adapter
        training_state['logs'].append({
            'timestamp': get_formatted_timestamp(),
            'message': 'Applying LoRA adapter...',
            'level': 'info'
        })
        
        lora_impl.add_lora_adapter()
        
        try:
            # Set up training dataset - with better error handling for datasets
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': 'Preparing dataset...',
                'level': 'info'
            })
            
            try:
                # Try importing datasets and creating a proper Dataset object
                try:
                    from datasets import Dataset
                    train_dataset = Dataset.from_list(dataset)
                    print("Successfully created datasets.Dataset")
                except (ImportError, ModuleNotFoundError):
                    # Fall back to using a custom dataset class if datasets library is not available
                    print("datasets library not available, using custom dataset implementation")
                    from torch.utils.data import Dataset as TorchDataset
                    
                    class CustomDataset(TorchDataset):
                        def __init__(self, data_items):
                            self.data = data_items
                            
                        def __len__(self):
                            return len(self.data)
                            
                        def __getitem__(self, idx):
                            return self.data[idx]
                    
                    train_dataset = CustomDataset(dataset)
                    print("Created custom dataset with torch.utils.data.Dataset")
            except Exception as dataset_error:
                print(f"Error creating dataset: {dataset_error}")
                training_state['logs'].append({
                    'timestamp': get_formatted_timestamp(),
                    'message': f'Error creating dataset: {str(dataset_error)}',
                    'level': 'error'
                })
                raise dataset_error
            
            # Convert to format expected by transformers
            def preprocess_function(examples):
                inputs = []
                outputs = []
                
                print(f"Dataset examples type: {type(examples)}")
                if len(examples) > 0:
                    print(f"First example type: {type(examples[0])}")
                    print(f"First example: {examples[0]}")
                
                for ex in examples:
                    # First handle if the example is a string
                    if isinstance(ex, str):
                        # For string inputs, we'll use the string as input and default output
                        inputs.append(ex)
                        outputs.append("")  # Empty output for string inputs
                        continue
                        
                    # Then handle dictionary-like objects
                    try:
                        # Check the format and extract input and output
                        if hasattr(ex, 'get'):
                            # This is a dictionary-like object with a get method
                            if 'instruction' in ex:
                                # Instruction format
                                instruction = ex.get('instruction', '')
                                input_text = ex.get('input', '')
                                if input_text:
                                    combined_input = f"{instruction}\n{input_text}"
                                else:
                                    combined_input = instruction
                                inputs.append(combined_input)
                                outputs.append(ex.get('output', ''))
                            elif 'prompt' in ex:
                                # Prompt-completion format
                                inputs.append(ex.get('prompt', ''))
                                outputs.append(ex.get('completion', ''))
                            elif 'question' in ex:
                                # Q&A format
                                inputs.append(ex.get('question', ''))
                                outputs.append(ex.get('answer', ''))
                            else:
                                # Fallback for dictionary without known keys
                                inputs.append(str(ex))
                                outputs.append('')
                        elif hasattr(ex, '__getitem__'):
                            # This might be a tuple or list format like (input, output)
                            if len(ex) >= 2:
                                inputs.append(str(ex[0]))
                                outputs.append(str(ex[1]))
                            else:
                                inputs.append(str(ex))
                                outputs.append('')
                        else:
                            # Any other object type, convert to string
                            inputs.append(str(ex))
                            outputs.append('')
                    except Exception as ex_error:
                        print(f"Error processing example {ex}: {ex_error}")
                        # Add a placeholder for this example
                        inputs.append("Error processing example")
                        outputs.append('')
                
                # Handle potential character encoding issues
                safe_inputs = []
                safe_outputs = []
                
                for i, (inp, out) in enumerate(zip(inputs, outputs)):
                    try:
                        # Try to encode and then decode to catch any encoding issues
                        inp_safe = inp.encode('utf-8', errors='ignore').decode('utf-8')
                        out_safe = out.encode('utf-8', errors='ignore').decode('utf-8')
                        safe_inputs.append(inp_safe)
                        safe_outputs.append(out_safe)
                    except Exception as char_error:
                        print(f"Warning: Skipping example {i} due to encoding issues: {char_error}")
                        # Add placeholder for problematic examples
                        safe_inputs.append("[Encoding error - example skipped]")
                        safe_outputs.append("")
                
                # Tokenize inputs and outputs
                max_length = training_args.get('max_length', 512)
                
                tokenized_inputs = lora_impl.tokenizer(safe_inputs, truncation=True, padding='max_length', max_length=max_length)
                tokenized_outputs = lora_impl.tokenizer(safe_outputs, truncation=True, padding='max_length', max_length=max_length)
                
                # Create input_ids, attention_mask, and labels
                result = {
                    'input_ids': tokenized_inputs.input_ids,
                    'attention_mask': tokenized_inputs.attention_mask,
                    'labels': tokenized_outputs.input_ids
                }
                
                return result
            
            # Process dataset
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': 'Processing dataset...',
                'level': 'info'
            })
            
            # Check if the dataset is empty
            if len(train_dataset) == 0:
                error_msg = "Dataset is empty. Please provide training data."
                training_state['logs'].append({
                    'timestamp': get_formatted_timestamp(),
                    'message': error_msg,
                    'level': 'error'
                })
                training_state['status'] = 'error'
                training_state['error'] = error_msg
                return
                
            # Process the dataset directly without using the map function
            try:
                # First handle any potential encoding issues
                for i, example in enumerate(train_dataset):
                    print(f"Processing example {i+1}/{len(train_dataset)}...")
                
                # Process each example manually to avoid the dataset mapping issues
                inputs = []
                outputs = []
                
                for i, example in enumerate(train_dataset):
                    try:
                        # Extract the input and output based on the format
                        if isinstance(example, dict):
                            if 'instruction' in example:
                                instruction = example.get('instruction', '')
                                input_text = example.get('input', '')
                                if input_text:
                                    combined_input = f"{instruction}\n{input_text}"
                                else:
                                    combined_input = instruction
                                inputs.append(combined_input)
                                outputs.append(example.get('output', ''))
                            elif 'prompt' in example:
                                inputs.append(example.get('prompt', ''))
                                outputs.append(example.get('completion', ''))
                            elif 'question' in example:
                                inputs.append(example.get('question', ''))
                                outputs.append(example.get('answer', ''))
                            else:
                                inputs.append(str(example))
                                outputs.append('')
                        else:
                            inputs.append(str(example))
                            outputs.append('')
                            
                        # Handle any character encoding issues
                        inputs[-1] = inputs[-1].encode('utf-8', errors='ignore').decode('utf-8')
                        outputs[-1] = outputs[-1].encode('utf-8', errors='ignore').decode('utf-8')
                        
                    except Exception as ex_error:
                        print(f"Error processing example {i}: {ex_error}")
                        # Skip problematic examples
                        continue
                
                print(f"Successfully processed {len(inputs)} examples")
                
                if len(inputs) == 0:
                    error_msg = "No valid examples found after processing the dataset."
                    training_state['logs'].append({
                        'timestamp': get_formatted_timestamp(),
                        'message': error_msg,
                        'level': 'error'
                    })
                    training_state['status'] = 'error'
                    training_state['error'] = error_msg
                    return
                    
                # Tokenize the inputs and outputs
                max_length = training_args.get('max_length', 512)
                tokenized_inputs = lora_impl.tokenizer(inputs, truncation=True, padding='max_length', max_length=max_length)
                tokenized_outputs = lora_impl.tokenizer(outputs, truncation=True, padding='max_length', max_length=max_length)
                
                # Create dataset dictionary for Trainer
                from torch.utils.data import Dataset as TorchDataset
                
                class SimpleDataset(TorchDataset):
                    def __init__(self, input_ids, attention_mask, labels):
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                        self.labels = labels
                        
                    def __len__(self):
                        return len(self.input_ids)
                        
                    def __getitem__(self, idx):
                        return {
                            'input_ids': self.input_ids[idx],
                            'attention_mask': self.attention_mask[idx],
                            'labels': self.labels[idx]
                        }
                
                processed_dataset = SimpleDataset(
                    input_ids=tokenized_inputs.input_ids,
                    attention_mask=tokenized_inputs.attention_mask,
                    labels=tokenized_outputs.input_ids
                )
                
                print(f"Created dataset with {len(processed_dataset)} examples")
                training_state['logs'].append({
                    'timestamp': get_formatted_timestamp(),
                    'message': f"Successfully prepared dataset with {len(processed_dataset)} examples",
                    'level': 'info'
                })
                
            except Exception as map_error:
                error_details = str(map_error)
                error_msg = f"Error processing dataset: {error_details if error_details else 'Empty dataset or format error'}"
                print(error_msg)
                print(f"Dataset type: {type(train_dataset)}")
                print(f"Dataset length: {len(train_dataset)}")
                print(f"First few items: {str(train_dataset[:5] if hasattr(train_dataset, '__getitem__') else 'No items')}")
                
                training_state['logs'].append({
                    'timestamp': get_formatted_timestamp(),
                    'message': error_msg,
                    'level': 'error'
                })
                training_state['status'] = 'error'
                training_state['error'] = error_msg
                return
            
            # Start training
            from transformers import Trainer, TrainingArguments
            
            batch_size = training_args.get('batch_size', 4)
            epochs = training_args.get('epochs', 3)
            learning_rate = training_args.get('learning_rate', 2e-5)
            
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': f'Starting training with batch_size={batch_size}, epochs={epochs}, lr={learning_rate}',
                'level': 'info'
            })
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=training_args.get('weight_decay', 0.01),
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=10,
                save_strategy="epoch",
                report_to="none",  # Disable wandb and other reporting
            )
            
            # Import TrainerCallback class for proper inheritance
            from transformers.trainer_callback import TrainerCallback
            
            # Custom callback to update training status
            class CustomCallback(TrainerCallback):
                # Initialize with required methods
                def on_init_end(self, args, state, control, **kwargs):
                    # Called when the Trainer is initialized
                    training_state['logs'].append({
                        'timestamp': get_formatted_timestamp(),
                        'message': 'Trainer initialized',
                        'level': 'info'
                    })
                    return control
                
                def on_train_begin(self, args, state, control, **kwargs):
                    # Called when training begins
                    training_state['logs'].append({
                        'timestamp': get_formatted_timestamp(),
                        'message': 'Training started',
                        'level': 'info'
                    })
                    training_state['progress'] = 0
                    return control
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    # Called when training logs are updated
                    if logs and "loss" in logs:
                        # Update training state
                        training_state['loss'] = logs["loss"]
                        training_state['logs'].append({
                            'timestamp': get_formatted_timestamp(),
                            'message': f'Step {state.global_step}: loss={logs["loss"]:.4f}',
                            'level': 'info'
                        })
                        
                        # Calculate progress
                        total_steps = state.max_steps
                        if total_steps > 0:
                            progress = min(100, int((state.global_step / total_steps) * 100))
                            training_state['progress'] = progress
                        
                        # Check for abort signal
                        global abort_training
                        if abort_training:
                            training_state['logs'].append({
                                'timestamp': get_formatted_timestamp(),
                                'message': 'Aborting training due to user request',
                                'level': 'warning'
                            })
                            training_state['status'] = 'aborted'
                            # Return stop signal to trainer
                            control.should_training_stop = True
                            print("Abort signal detected - stopping training")
                    return control
                    
                def on_step_end(self, args, state, control, **kwargs):
                    # Check for abort signal on every step
                    global abort_training
                    if abort_training:
                        training_state['logs'].append({
                            'timestamp': get_formatted_timestamp(),
                            'message': 'Aborting training due to user request',
                            'level': 'warning'
                        })
                        training_state['status'] = 'aborted'
                        # Return stop signal to trainer
                        control.should_training_stop = True
                        print("Abort signal detected - stopping training")
                    return control
                    
                def on_train_end(self, args, state, control, **kwargs):
                    # Called when training ends
                    global abort_training
                    if abort_training:
                        training_state['logs'].append({
                            'timestamp': get_formatted_timestamp(),
                            'message': 'Training aborted by user',
                            'level': 'warning'
                        })
                        training_state['status'] = 'aborted'
                    else:
                        training_state['logs'].append({
                            'timestamp': get_formatted_timestamp(),
                            'message': 'Training completed',
                            'level': 'success'
                        })
                        training_state['status'] = 'completed'
                    training_state['progress'] = 100
                    return control
            
            # Create trainer
            trainer = Trainer(
                model=lora_impl.model,
                args=training_args,
                train_dataset=processed_dataset,
                callbacks=[CustomCallback()]
            )
            
            # Train the model
            trainer.train()
            
            # Save the adapter
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': 'Training complete. Saving adapter...',
                'level': 'info'
            })
            
            lora_impl.save_adapter(output_dir)
            
            # Generate a unique adapter ID
            adapter_id = f"lora-adapter-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            training_state['adapter_id'] = adapter_id
            adapter_dir = os.path.join(output_dir, adapter_id)
            os.makedirs(adapter_dir, exist_ok=True)
            
            # Save model-adapter mapping for auto-loading
            try:
                print(f"Saving model-adapter mapping for {model_id} -> {adapter_id}")
                save_model_adapter_mapping(model_id, adapter_id)
            except Exception as mapping_error:
                print(f"Warning: Could not save model-adapter mapping: {mapping_error}")
            
            # Update training state
            training_state['status'] = 'completed'
            training_state['progress'] = 100
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': f'Adapter saved to {output_dir}',
                'level': 'success'
            })
            
        except Exception as e:
            # Update training state on error
            training_state['status'] = 'error'
            training_state['error'] = str(e)
            training_state['logs'].append({
                'timestamp': get_formatted_timestamp(),
                'message': f'Error during training: {str(e)}',
                'level': 'error'
            })
            raise e
            
    except Exception as e:
        # Update training state on error
        training_state['status'] = 'error'
        training_state['error'] = str(e)
        training_state['logs'].append({
            'timestamp': get_formatted_timestamp(),
            'message': f'Error: {str(e)}',
            'level': 'error'
        })

        gpu_info['message'] = "PyTorch is not installed"
        gpu_info['error'] = "Please install PyTorch to enable GPU detection"
    else:
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info['available'] = True
                gpu_info['count'] = gpu_count
                gpu_info['devices'] = []
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024  # Convert to GB
                    gpu_info['devices'].append({
                        'id': i,
                        'name': gpu_name,
                        'total_memory_gb': f"{total_memory:.2f}"
                    })
            else:
                gpu_info['available'] = False
                gpu_info['message'] = "CUDA is not available"
        except Exception as e:
            gpu_info['available'] = False
            gpu_info['error'] = str(e)
            gpu_info['message'] = "Error detecting GPU hardware"
            print(f"Error detecting GPU hardware: {e}")
    
    # Get CPU info
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
    except Exception:
        cpu_count = 1  # Fallback value
    
    return jsonify({
        "status": "success",
        "hardware": {
            "gpu": gpu_info,
            "cpu": {
                "count": cpu_count
            },
            "torch_available": TORCH_AVAILABLE,
            "lora_available": LORA_AVAILABLE
        }
    })

@app.route('/api/lora/quantization_options', methods=['GET'])
def get_quantization_options():
    """Get available quantization options based on hardware"""
    options = ["none"]  # Full precision is always available
    
    if torch.cuda.is_available():
        # Check if bitsandbytes is installed
        try:
            import bitsandbytes
            options.extend(["4bit", "8bit"])
        except ImportError:
            pass
    
    return jsonify({
        "status": "success",
        "quantization_options": options
    })

# Commenting out duplicate endpoint
# @app.route('/api/lora/training_status', methods=['GET'])
# def get_training_status():
#     """Get the current status of LoRA training"""
#     # Add a message to indicate that the training is complete or failed
#     if training_state['status'] == 'completed':
#         training_state['message'] = "Training completed successfully! The LoRA adapter is ready to use."
#     elif training_state['status'] == 'error':
#         training_state['message'] = f"Training failed: {training_state['error']}"
#     elif training_state['status'] == 'training':
#         training_state['message'] = "Training in progress..."
#     elif training_state['status'] == 'initializing':
#         training_state['message'] = "Initializing training..."
#     
#     return jsonify(training_state)

# Functions to manage model-adapter mappings
def get_model_adapter_mappings():
    """Get the current model-adapter mappings"""
    try:
        if os.path.exists(MODEL_ADAPTER_MAPPING_FILE):
            with open(MODEL_ADAPTER_MAPPING_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading model-adapter mappings: {e}")
        return {}

def save_model_adapter_mapping(model_id, adapter_id):
    """Save a model-adapter mapping"""
    try:
        # Load existing mappings
        mappings = get_model_adapter_mappings()
        
        # Update with new mapping
        mappings[model_id] = adapter_id
        
        # Save back to file
        with open(MODEL_ADAPTER_MAPPING_FILE, 'w') as f:
            json.dump(mappings, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving model-adapter mapping: {e}")
        return False

def delete_model_adapter_mapping(model_id):
    """Delete a model-adapter mapping"""
    try:
        # Load existing mappings
        mappings = get_model_adapter_mappings()
        
        # Remove mapping if it exists
        if model_id in mappings:
            del mappings[model_id]
            
            # Save back to file
            with open(MODEL_ADAPTER_MAPPING_FILE, 'w') as f:
                json.dump(mappings, f, indent=2)
                
            return True
        return False
    except Exception as e:
        print(f"Error deleting model-adapter mapping: {e}")
        return False

@app.route('/api/lora/model_adapter_mappings', methods=['GET'])
def api_get_model_adapter_mappings():
    """API endpoint to get all model-adapter mappings"""
    try:
        mappings = get_model_adapter_mappings()
        return jsonify({"status": "success", "mappings": mappings})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/model_adapter_mapping', methods=['POST'])
def api_save_model_adapter_mapping():
    """API endpoint to save a model-adapter mapping"""
    try:
        data = request.json
        model_id = data.get('model_id')
        adapter_id = data.get('adapter_id')
        
        if not model_id or not adapter_id:
            return jsonify({"status": "error", "message": "Missing model_id or adapter_id"}), 400
        
        success = save_model_adapter_mapping(model_id, adapter_id)
        
        if success:
            return jsonify({"status": "success", "message": f"Mapping saved for {model_id}"})
        else:
            return jsonify({"status": "error", "message": "Failed to save mapping"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/model_adapter_mapping/<model_id>', methods=['DELETE'])
def api_delete_model_adapter_mapping(model_id):
    """API endpoint to delete a model-adapter mapping"""
    try:
        success = delete_model_adapter_mapping(model_id)
        
        if success:
            return jsonify({"status": "success", "message": f"Mapping deleted for {model_id}"})
        else:
            return jsonify({"status": "error", "message": "No mapping found for this model"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/get_adapter_for_model/<model_id>', methods=['GET'])
def api_get_adapter_for_model(model_id):
    """API endpoint to get the adapter ID for a specific model"""
    try:
        mappings = get_model_adapter_mappings()
        
        if model_id in mappings:
            return jsonify({"status": "success", "adapter_id": mappings[model_id]})
        else:
            return jsonify({"status": "error", "message": "No adapter mapping found for this model"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/save_adapter', methods=['POST'])
def save_adapter():
    """Save the current adapter weights and configuration in PEFT format"""
    try:
        data = request.json
        adapter_name = data.get('adapter_name')
        
        if not adapter_name:
            return jsonify({"status": "error", "message": "Missing adapter_name"}), 400
        
        # Check if we have a loaded model with adapter
        if not lora_impl or not lora_impl.model:
            return jsonify({"status": "error", "message": "No model with adapter loaded. Please apply LoRA first."}), 400
        
        # Create the save directory
        save_dir = os.path.join('lora_adapter', adapter_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the adapter in PEFT format
        try:
            # Save adapter weights and config
            lora_impl.save_adapter(save_dir)
            
            # Also save the tokenizer for convenience
            if lora_impl.tokenizer:
                lora_impl.tokenizer.save_pretrained(save_dir)
            
            return jsonify({
                "status": "success", 
                "message": f"Adapter saved successfully to {save_dir}",
                "save_path": save_dir
            })
        except Exception as save_error:
            return jsonify({"status": "error", "message": f"Failed to save adapter: {str(save_error)}"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/list_adapters', methods=['GET'])
def list_adapters():
    """List all available LoRA adapters in the lora_adapter folder"""
    try:
        adapter_dir = 'lora_adapter'
        adapters = []
        
        if os.path.exists(adapter_dir):
            # List all directories in lora_adapter
            for item in os.listdir(adapter_dir):
                item_path = os.path.join(adapter_dir, item)
                # Check if it's a directory and contains adapter_config.json
                if os.path.isdir(item_path):
                    config_path = os.path.join(item_path, 'adapter_config.json')
                    if os.path.exists(config_path):
                        # Read adapter config to get base model info
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            adapters.append({
                                'name': item,
                                'path': item_path,
                                'base_model': config.get('base_model_name_or_path', 'Unknown'),
                                'created': os.path.getmtime(item_path)
                            })
                        except Exception as e:
                            print(f"Error reading adapter config for {item}: {e}")
                            adapters.append({
                                'name': item,
                                'path': item_path,
                                'base_model': 'Unknown',
                                'created': os.path.getmtime(item_path)
                            })
        
        # Sort by creation time (newest first)
        adapters.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'adapters': adapters
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/lora/load_adapter_to_model', methods=['POST'])
def load_adapter_to_model():
    """Load a LoRA adapter onto a base model"""
    try:
        data = request.json
        model_id = data.get('model_id')
        adapter_name = data.get('adapter_name')
        
        if not model_id or not adapter_name:
            return jsonify({"status": "error", "message": "Missing model_id or adapter_name"}), 400
        
        adapter_path = os.path.join('lora_adapter', adapter_name)
        
        # Check if adapter exists
        if not os.path.exists(adapter_path) or not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            return jsonify({"status": "error", "message": f"Adapter '{adapter_name}' not found"}), 404
        
        try:
            global lora_impl
            
            # Initialize new LoRA implementation if needed
            if not lora_impl:
                lora_impl = LoRAImplementation()
            
            # Process model ID (same logic as in apply_lora)
            sanitized_model_id = model_id
            
            if not model_id.startswith('ollama:') and not model_id.startswith('huggingface:'):
                sanitized_model_id = f"ollama:{model_id}"
            
            if ':' in sanitized_model_id and sanitized_model_id.startswith('ollama:'):
                base_name = sanitized_model_id.split(':')[1].split('.')[0]
                
                if base_name.startswith('llama3'):
                    sanitized_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                elif base_name.startswith('llama2'):
                    sanitized_model_id = "openlm-research/open_llama_3b_v2"
                elif base_name.startswith('mistral'):
                    sanitized_model_id = "eachadea/legacy-mistral-7b-v0.1"
                else:
                    sanitized_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                    
                sanitized_model_id = f"hf_direct:{sanitized_model_id}"
            
            print(f"Loading model {sanitized_model_id} with adapter {adapter_name}")
            
            # Load base model first
            lora_impl.load_base_model(sanitized_model_id, quantization='4bit')  # Default to 4bit for memory efficiency
            
            # Load the adapter onto the model
            from peft import PeftModel
            lora_impl.model = PeftModel.from_pretrained(lora_impl.model, adapter_path)
            
            print(f"Successfully loaded adapter {adapter_name} onto model {model_id}")
            
            return jsonify({
                "status": "success",
                "message": f"Adapter '{adapter_name}' loaded successfully onto model '{model_id}'",
                "adapter_name": adapter_name,
                "model_id": model_id
            })
            
        except Exception as load_error:
            print(f"Error loading adapter: {load_error}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": f"Failed to load adapter: {str(load_error)}"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/lora/merge_and_export', methods=['POST'])
def merge_and_export():
    """Merge LoRA adapter with base model and export as a full model"""
    try:
        data = request.json
        export_name = data.get('export_name')
        export_path = data.get('export_path', 'merged_models')  # Default to merged_models directory
        
        if not export_name:
            return jsonify({"status": "error", "message": "Missing export_name"}), 400
        
        # Check if we have a loaded model with adapter
        if not lora_impl or not lora_impl.model:
            return jsonify({"status": "error", "message": "No model with adapter loaded. Please apply LoRA and complete training first."}), 400
        
        # Create the export directory
        full_export_path = os.path.join(export_path, export_name)
        os.makedirs(full_export_path, exist_ok=True)
        
        try:
            # Log the merge operation
            print(f"Starting merge and export to {full_export_path}")
            
            # Merge the adapter and unload (this creates a standard model)
            merged_model = lora_impl.merge_and_export()
            
            # Save the merged model
            print("Saving merged model...")
            merged_model.save_pretrained(full_export_path)
            
            # Save the tokenizer
            if lora_impl.tokenizer:
                print("Saving tokenizer...")
                lora_impl.tokenizer.save_pretrained(full_export_path)
            
            # Create a model card / README
            model_card_content = f"""# {export_name}

This is a fine-tuned model created by merging a LoRA adapter with the base model.

## Base Model
{lora_impl.model_id if lora_impl.model_id else 'Unknown'}

## Training Configuration
- LoRA Rank: {lora_impl.lora_config.get('rank', 8)}
- LoRA Alpha: {lora_impl.lora_config.get('alpha', 16)}
- LoRA Dropout: {lora_impl.lora_config.get('dropout', 0.05)}
- Target Modules: {lora_impl.lora_config.get('target_modules', ['q_proj', 'v_proj'])}

## Usage
This model can be loaded like any standard Hugging Face model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{full_export_path}")
tokenizer = AutoTokenizer.from_pretrained("{full_export_path}")
```

Generated with MF Vibe-Tuning
"""
            
            with open(os.path.join(full_export_path, 'README.md'), 'w') as f:
                f.write(model_card_content)
            
            return jsonify({
                "status": "success", 
                "message": f"Model successfully merged and exported to {full_export_path}",
                "export_path": full_export_path,
                "model_size_mb": sum(os.path.getsize(os.path.join(full_export_path, f)) 
                                   for f in os.listdir(full_export_path) 
                                   if os.path.isfile(os.path.join(full_export_path, f))) / (1024 * 1024)
            })
            
        except Exception as export_error:
            return jsonify({"status": "error", "message": f"Failed to merge and export model: {str(export_error)}"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)

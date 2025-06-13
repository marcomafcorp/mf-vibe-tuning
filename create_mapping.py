import json
import os

# Define the model ID and adapter ID
model_id = "llama3.2:1b"  # Change this to your actual model ID
adapter_id = "lora-adapter-20250529100546"  # Your adapter ID from training

# Create the mapping
mapping = {model_id: adapter_id}

# Save to file
with open('model_adapter_mappings.json', 'w') as f:
    json.dump(mapping, f, indent=2)
    
print(f"Created mapping for {model_id} -> {adapter_id}")
print(f"Saved to model_adapter_mappings.json")

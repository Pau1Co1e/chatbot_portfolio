from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, AutoTokenizer
import torch
import os

# Path to the downloaded model
downloaded_model_path = "/Users/paulcoleman/.llama/checkpoints/Llama3.2-1B-Instruct-int4-qlora-eo8"

# Output directory for Hugging Face model
output_dir = "converted_llama_hf"

# Select device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Path to the downloaded model
downloaded_model_path = "/Users/paulcoleman/.llama/checkpoints/Llama3.2-1B-Instruct-int4-qlora-eo8"

# Explicitly load tokenizer.model file
tokenizer_path = os.path.join(downloaded_model_path, "tokenizer.model")
if not os.path.exists(tokenizer_path):
    raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
tokenizer = AutoTokenizer.from_pretrained(downloaded_model_path)

# try:
#     # Attempt to load tokenizer
#     tokenizer = LlamaTokenizer.from_pretrained(downloaded_model_path, legacy=True)
#     print(f"Tokenizer successfully loaded: {type(tokenizer)}")
# except Exception as e:
#     print(f"Failed to initialize tokenizer: {e}")
#     raise

# Load configuration
config_path = os.path.join(downloaded_model_path, "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

config = LlamaConfig.from_pretrained(config_path)

# Load model weights
model_weights_path = os.path.join(downloaded_model_path, "consolidated.00.pth")
if not os.path.exists(model_weights_path):
    raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

# Load the state dictionary
state_dict = torch.load(model_weights_path, map_location=device)

# Clean up state dict keys (if necessary)
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_key = k[len("model.") :]
    else:
        new_key = k
    new_state_dict[new_key] = v

# Initialize model
model = LlamaForCausalLM(config=config)
model.load_state_dict(new_state_dict, strict=False)  # Allow missing/unexpected keys

# Save in Hugging Face format
os.makedirs(output_dir, exist_ok=True)
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print(f"Model successfully converted and saved to {output_dir}")

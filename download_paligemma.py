
from transformers import AutoTokenizer
import os
import huggingface_hub

# Set your Hugging Face token
os.environ["HF_TOKEN"] = "your token"

# Create directory and download model
model_name = "google/paligemma-3b-pt-224"
save_directory = "E:/VLMs/vlmverse/paligemma_weights"

print(f"Downloading model {model_name}...")

# First download just the tokenizer (smaller)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
tokenizer.save_pretrained(save_directory)
print("Tokenizer saved successfully")

# Then download the model files directly without loading them into memory
huggingface_hub.snapshot_download(
    repo_id=model_name,
    local_dir=save_directory,
    token=os.environ["HF_TOKEN"],
    local_dir_use_symlinks=False
)

print("Download complete!")
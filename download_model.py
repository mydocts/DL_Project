import os

# Set mirror endpoint
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

target_dir = os.path.abspath("resources/IP2P")
print(f"Downloading model to {target_dir} using mirror https://hf-mirror.com...")
snapshot_download(repo_id="timbrooks/instruct-pix2pix", local_dir=target_dir, local_dir_use_symlinks=False, resume_download=True)
print("Download complete.")

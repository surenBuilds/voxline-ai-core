"""Download the default Voxline chat model once, into the project directory."""

import os
from pathlib import Path

# Some networks cannot reach Hugging Face's optional Xet transfer service.
# The regular HTTP download route is slower but more broadly compatible.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download


REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
DESTINATION = Path("models/Qwen2.5-0.5B-Instruct")


if __name__ == "__main__":
    print(f"Downloading {REPOSITORY} to {DESTINATION}...", flush=True)
    snapshot_download(repo_id=REPOSITORY, local_dir=DESTINATION, resume_download=True)
    print("Download complete.", flush=True)

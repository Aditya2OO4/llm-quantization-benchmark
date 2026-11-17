"""Download the original 16-bit TinyLlama 1.1B from Hugging Face (~2.2 GB).

Usage:
    python step2_download_model.py
"""

from huggingface_hub import snapshot_download

from config import HF_MODEL_ID, raw_model_dir


def main() -> None:
    target = raw_model_dir()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HF_MODEL_ID} to {target} (about 2.2 GB) ...")
    snapshot_download(
        repo_id=HF_MODEL_ID,
        local_dir=str(target),
        ignore_patterns=["*.gguf", "*.bin"],  # only the safetensors are needed
    )
    print("Done. Next: see step3_quantize.md")


if __name__ == "__main__":
    main()

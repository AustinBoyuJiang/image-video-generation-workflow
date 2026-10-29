"""Download the exact model files used by this workflow; no weights enter Git."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "model-lock.json").read_text())
QWEN_CONFIG_PATTERNS = [
    "model_index.json", "LICENSE", "README.md", "processor/*", "scheduler/*",
    "text_encoder/*", "transformer/*", "vae/*",
]


def verify_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    if path.stat().st_size != expected["size"]:
        raise RuntimeError(f"Model size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        raise RuntimeError(f"Model SHA256 mismatch: {path}")
    print(f"Verified {path.relative_to(ROOT)}", flush=True)


def process_model(name, verify_only=False):
    entry = LOCK[name]
    target = ROOT / ("qwen-image-2.1" if name == "qwen" else "h3-models")
    if not verify_only:
        patterns = QWEN_CONFIG_PATTERNS if name == "qwen" else list(entry["weights"])
        print(f"Downloading {entry['repo']} at {entry['revision']} to {target}", flush=True)
        snapshot_download(
            repo_id=entry["repo"], revision=entry["revision"],
            local_dir=target, allow_patterns=patterns, max_workers=2,
        )
    for filename, expected in entry["weights"].items():
        verify_file(target / filename, expected)
    if name == "qwen" and not (target / "model_index.json").is_file():
        raise FileNotFoundError("Qwen model_index.json is missing")
    print(f"{name.upper()} model ready", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "qwen", "h3"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    names = ("qwen", "h3") if args.model == "all" else (args.model,)
    for name in names:
        process_model(name, args.verify_only)


if __name__ == "__main__":
    main()

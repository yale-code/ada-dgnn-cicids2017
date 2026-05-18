#!/usr/bin/env python3
"""Test kagglehub and download CICIDS2017 dataset."""

import kagglehub
import os
import sys
import zipfile
from pathlib import Path

DEST_DIR = Path("/root/.openclaw/workspace/datasets/cicids2017")
DATASETS = [
    "sateeshkumar6289/cicids-2017-dataset",
    "cicdataset/cicids2017",
    "naveengill/cicids2017-dataset",
]

print("=== Testing kagglehub ===")
try:
    print(f"kagglehub version: {kagglehub.__version__}")
    # Quick test: try to fetch info for a known dataset
    print("kagglehub imported successfully.")
except Exception as e:
    print(f"kagglehub import failed: {e}")
    sys.exit(1)

print("\n=== Attempting dataset downloads ===")

for ds in DATASETS:
    print(f"\n--- Trying: {ds} ---")
    try:
        # kagglehub.dataset_download downloads to a cache dir; we can specify force_download
        # but to control destination, we download then move/copy if needed.
        # kagglehub returns the path to the downloaded file/folder.
        path = kagglehub.dataset_download(ds, force_download=True)
        print(f"Downloaded to cache path: {path}")

        # If it's a zip, extract
        downloaded_path = Path(path)
        if downloaded_path.is_file() and downloaded_path.suffix == ".zip":
            print("Extracting zip...")
            with zipfile.ZipFile(downloaded_path, 'r') as z:
                z.extractall(DEST_DIR)
            print(f"Extracted to {DEST_DIR}")
        elif downloaded_path.is_dir():
            # Move/copy directory contents
            import shutil
            for item in downloaded_path.iterdir():
                dest_item = DEST_DIR / item.name
                if dest_item.exists():
                    if dest_item.is_dir():
                        shutil.rmtree(dest_item)
                    else:
                        dest_item.unlink()
                shutil.move(str(item), str(dest_item))
            print(f"Moved contents to {DEST_DIR}")
        else:
            # Single file, just copy
            import shutil
            dest_file = DEST_DIR / downloaded_path.name
            shutil.copy2(str(downloaded_path), str(dest_file))
            print(f"Copied file to {dest_file}")

        print("Success.")
        break
    except Exception as e:
        print(f"Failed: {e}")
else:
    print("\nAll dataset paths failed.")
    sys.exit(1)

# List files
print("\n=== Files in destination ===")
for root, dirs, files in os.walk(DEST_DIR):
    level = root.replace(str(DEST_DIR), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        fpath = Path(root) / file
        size = fpath.stat().st_size
        print(f"{subindent}{file} ({size:,} bytes)")

# Report total size
total_bytes = sum(f.stat().st_size for f in DEST_DIR.rglob('*') if f.is_file())
total_mb = total_bytes / (1024 * 1024)
print(f"\n=== Total downloaded data: {total_mb:.2f} MB ({total_bytes:,} bytes) ===")

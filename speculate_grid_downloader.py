#!/usr/bin/env python3
"""
Download and decompress spectral grid files from HuggingFace datasets.

Usage:
    python speculate_grid_downloader.py <dataset-id> [filename]
    
Arguments:
    dataset-id: The dataset ID (e.g., speculate_cv_bl_grid_v87f)
    filename: (optional) Specific file to download (e.g., run0.spec.xz)
    
Examples:
    # Download all files from a dataset
    python speculate_grid_downloader.py speculate_cv_bl_grid_v87f
    
    # Download a specific file
    python speculate_grid_downloader.py speculate_cv_bl_grid_v87f run0.spec.xz
"""

import os
import sys
import lzma
import shutil
from tqdm import tqdm

# Disable ALL progress bars from huggingface_hub
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_VERBOSITY'] = 'error'

from huggingface_hub import hf_hub_download, list_repo_files
import logging

# Suppress huggingface_hub logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# --- Parse command-line arguments ---
if len(sys.argv) < 2:
    print("Error: Dataset ID is required")
    print(__doc__)
    sys.exit(1)

DATASET_ID = sys.argv[1]
SPECIFIC_FILE = sys.argv[2] if len(sys.argv) > 2 else None

# --- Configuration ---
ORG_ID = "Sirocco-rt"
REPO_ID = ORG_ID + "/" + DATASET_ID
REPO_TYPE = "dataset"
EXTRACTION_DIR = "raw_grids/" + DATASET_ID + "/"

# Get a list of all files in the repo
all_files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

# Filter for only the files you want
spec_files = [f for f in all_files if f.startswith("run") and f.endswith(".spec.xz")]

# If a specific file was requested, filter to just that file
if SPECIFIC_FILE:
    # Ensure the filename ends with .spec.xz
    if not SPECIFIC_FILE.endswith(".spec.xz"):
        SPECIFIC_FILE += ".spec.xz"
    
    if SPECIFIC_FILE in spec_files:
        spec_files = [SPECIFIC_FILE]
        print(f"Downloading specific file: {SPECIFIC_FILE}")
    else:
        print(f"Error: File '{SPECIFIC_FILE}' not found in dataset")
        print(f"Available files start with 'run' and end with '.spec.xz'")
        sys.exit(1)
else:
    print(f"Found {len(spec_files)} files to download...")

# Step 1: Download all files
print("\n=== Downloading all files ===")
downloaded_files = []
for filename in tqdm(spec_files, desc="Downloading", unit="files"):
    local_path = hf_hub_download(
        repo_id=REPO_ID, 
        filename=filename, 
        repo_type=REPO_TYPE
    )
    downloaded_files.append((filename, local_path))

print("\n=== Decompressing all files ===")
os.makedirs(EXTRACTION_DIR, exist_ok=True)

# Step 2: Decompress all downloaded files
for filename, local_path in tqdm(downloaded_files, desc="Decompressing", unit="file"):
    output_filename = filename[:-3]
    output_file_path = os.path.join(EXTRACTION_DIR, output_filename)
    
    with lzma.open(local_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

print("\n✓ All files downloaded and decompressed.")
print(f"Files saved to: {os.path.abspath(EXTRACTION_DIR)}")

#!/bin/bash
# Script to download COCO 2017 Validation Set and Annotations
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data/coco"

echo "Creating data directory at $DATA_DIR"
mkdir -p "$DATA_DIR"

# 1. Download Val Images
if [ ! -d "$DATA_DIR/val2017" ]; then
    echo "Downloading Val 2017 images..."
    wget -c http://images.cocodataset.org/zips/val2017.zip -O "$DATA_DIR/val2017.zip"
    echo "Unzipping val2017.zip..."
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/val2017.zip"
else
    echo "val2017 directory already exists."
fi

# 2. Download Annotations
if [ ! -d "$DATA_DIR/annotations" ]; then
    echo "Downloading Train/Val 2017 Annotations..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O "$DATA_DIR/annotations_trainval2017.zip"
    echo "Unzipping annotations..."
    unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/annotations_trainval2017.zip"
else
    echo "annotations directory already exists."
fi

echo "Done. Data is ready in $DATA_DIR"

# PetComposer

This module processes raw images into a dataset suitable for training.

## Features
- Resizes images to 64x64 pixels.
- Converts to RGB.
- Splits data into Training (80%) and Testing (20%) sets.

## Usage
1.  Ensure raw data is present in `../shared/data/raw` (run `PetCollector` first).
2.  Run the composer:
    ```bash
    python main.py
    ```
# PetComposer

Processes raw images into a dataset suitable for training.

## Features
- **GUI Interface:** Interactively set Train/Validation/Test split ratios.
- **Normalization:** Resizes images to 64x64 pixels and converts to RGB.
- **Organization:** Splits data into `train`, `val`, and `test` folders in `shared/data/composed`.

## Usage
1.  Ensure raw data is present in `../shared/data/raw` (run `PetCollector` first).
2.  Run the composer:
    ```bash
    python main.py
    ```
3.  Use the sliders to adjust the splits (must sum to 100%) and click **"Process Dataset"**.

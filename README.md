# CatOrDog (Pet Classifier)

This project contains a set of tools for collecting, processing, training, and testing a binary image classifier for Cats and Dogs.

## Modules

### 1. PetCollector
Downloads raw images of cats and dogs from public APIs (`TheCatAPI`, `DogCEO`).
- **Output:** `shared/data/raw/cat`, `shared/data/raw/dog`

### 2. PetComposer
Processes the raw images:
- Resizes to 64x64 pixels.
- Converts to RGB.
- Splits into Training (80%) and Test (20%) sets.
- **Output:** `shared/data/composed/train`, `shared/data/composed/test`

### 3. PetTrainer
Trains a Convolutional Neural Network (CNN) on the processed data.
- **Input:** `shared/data/composed/train`
- **Output:** Trained model in `shared/models/run_TIMESTAMP/pet_cnn.pt`

### 4. PetTester
Evaluates the trained model on the test set.
- **Input:** `shared/models/...`, `shared/data/composed/test`
- **Output:** Confusion matrix and accuracy metrics.

## Workflow

1.  **Collect Data:**
    ```bash
    cd PetCollector
    run.bat
    ```
2.  **Process Data:**
    ```bash
    cd ../PetComposer
    run.bat
    ```
3.  **Train Model:**
    ```bash
    cd ../PetTrainer
    run.bat
    ```
4.  **Test Model:**
    ```bash
    cd ../PetTester
    run.bat
    ```

## Requirements
- Python 3.10+
- Internet connection (for Collector)
- Dependencies installed via `requirements.txt` in each module (handled automatically if using manual pip install, or run `.bat` files).

## Structure
```
catordog/
├── shared/                   # Central storage
│   ├── data/
│   │   ├── raw/              # Downloaded images
│   │   └── composed/         # Processed ready-for-training images
│   └── models/               # Saved models
├── PetCollector/             # Downloader
├── PetComposer/              # Preprocessor
├── PetTrainer/               # Training script
└── PetTester/                # Evaluation tool
```
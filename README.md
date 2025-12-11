# CatOrDog (Pet Classifier)

This project contains a set of tools for collecting, processing, training, and testing a binary image classifier for Cats and Dogs.

## Modules

### 1. PetCollector
GUI application for collecting cat and dog images. It supports both manual import of images from your local machine and automatic downloading of images from public APIs (`TheCatAPI`, `DogCEO`). Downloaded images are named sequentially (e.g., `cat_000.jpg`, `cat_001.jpg`).
- **Output:** `shared/data/raw/cat`, `shared/data/raw/dog`

### 2. PetComposer
GUI application for dataset preparation.
- **Features:** Interactively adjust **Train**, **Validation**, and **Test** split ratios via sliders.
- **Normalization:** Resizes images to 64x64 pixels and converts to RGB.
- **Output:** `shared/data/composed/train`, `shared/data/composed/val`, `shared/data/composed/test`

### 3. PetTrainer
GUI application for training the Convolutional Neural Network (CNN).
- **Features:** GUI with sliders to tune hyperparameters (Epochs, Batch Size, Learning Rate, Dropout, Step Size, LR Gamma, Seed).
- **CLI Support:** Run `python train.py --help` to see available command-line arguments.
- **Output:** Trained model in `shared/models/run_TIMESTAMP/pet_cnn.pt`

### 4. PetTester
Evaluates the trained model on the test set.
- **Input:** `shared/models/...`, `shared/data/composed/test`
- **Output:** Confusion matrix and accuracy metrics.

## Workflow

1.  **Collect Data (using GUI):**
    ```bash
    cd PetCollector
    run.bat
    ```
    *Use the GUI to manually import images or start the automatic download.*
2.  **Process Data (using GUI):**
    ```bash
    cd ../PetComposer
    run.bat
    ```
    *Use the sliders to configure the Train/Validation/Test splits (must sum to 100%) and click "Process Dataset".*
3.  **Train Model (using GUI):**
    ```bash
    cd ../PetTrainer
    run.bat
    ```
    *Adjust hyperparameters using sliders and click "Start Training".*
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
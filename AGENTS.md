# Agent Context & Technical Documentation

## Project Identity
**Name:** CatOrDog (formerly DIE-MNIST)
**Goal:** Binary image classification (Cat vs. Dog) using a modular pipeline.
**Current State:** Functional MVP with GUI data collection, preprocessing, CNN training, and evaluation.

## Technical Stack
*   **Language:** Python 3.10+
*   **ML Framework:** PyTorch (torch, torchvision)
*   **GUI Framework:** Tkinter (PetCollector, PetTester visualization)
*   **Image Processing:** Pillow (PIL)
*   **Data Format:**
    *   **Raw:** JPG/PNG/BMP (various sizes)
    *   **Processed:** 64x64 RGB Tensors

## System Architecture
The project follows a strict "separation of concerns" pipeline. Modules do not import each other; they communicate via the filesystem (`shared/` directory).

### 1. PetCollector
*   **Path:** `PetCollector/`
*   **Entry Point:** `main.py` (GUI Application)
*   **Function:** Acquires data.
    *   **Auto Download:** Fetches from `TheCatAPI` / `DogCEO`.
    *   **Manual Import:** Copies local files.
    *   **Logic:** Implements sequential naming (`cat_000.jpg`, `cat_001.jpg`) to avoid collisions.
*   **Output:** `shared/data/raw/{cat,dog}/`

### 2. PetComposer
*   **Path:** `PetComposer/`
*   **Entry Point:** `main.py` (GUI Application)
*   **Function:** Standardizes data.
    *   **Interactive Split:** GUI sliders to set **Train**, **Validation**, and **Test** ratios (must sum to 100%).
    *   **Normalization:** Resizes images to **64x64** and converts to **RGB**.
*   **Output:** `shared/data/composed/{train,val,test}/{cat,dog}/`

### 3. PetTrainer
*   **Path:** `PetTrainer/`
*   **Entry Point:** `train.py`
*   **Function:** Trains the model.
    *   **Model:** `SimpleCNN` (Custom architecture: 4 Conv blocks -> Flatten -> Linear).
    *   **Input:** `(3, 64, 64)` tensors.
    *   **Classes:** 2 (Cat=0, Dog=1).
*   **Output:** `shared/models/run_TIMESTAMP/pet_cnn.pt` (weights) + `config.json`.

### 4. PetTester
*   **Path:** `PetTester/`
*   **Entry Point:** `main.py`
*   **Function:** Evaluates the model.
    *   Loads the latest model from `shared/models`.
    *   Runs inference on `shared/data/composed/test`.
    *   Generates `results.json` and a Confusion Matrix.

## Context for Next Session
If you are an agent picking up this task, here is what you need to know:

1.  **Git State:** The project is a git repository. The `.gitignore` is set up to exclude large data files and models.
2.  **Environment:** Each module has a `requirements.txt`.
3.  **Recent Changes:**
    *   Refactored from digit recognition to pet classification.
    *   Added GUI to `PetCollector`.
    *   Added GUI to `PetComposer` for dynamic dataset splitting (Train/Val/Test).
    *   Implemented file renaming logic in `PetCollector` to handle incremental additions.
4.  **Potential Next Steps:**
    *   **Data Augmentation:** The current trainer uses raw images. Adding rotation/flip transforms in `PetTrainer` would improve robustness.
    *   **Model Improvement:** The `SimpleCNN` is basic. Consider Transfer Learning (ResNet/MobileNet) if accuracy needs a boost.
    *   **UI Polish:** `PetTester` has a visualization script (`visualize.py`) inherited from the old project; ensure it's fully compatible with the 2-class output.

## Key Paths
*   Raw Data: `shared/data/raw` (gitignored)
*   Processed Data: `shared/data/composed` (gitignored)
*   Models: `shared/models` (gitignored)
*   Logs/Results: `shared/tests`

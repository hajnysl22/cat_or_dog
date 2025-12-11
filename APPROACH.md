# Approach & Architecture

## Overview
The **CatOrDog** project is a strategic refactoring of an existing handwritten digit recognition system ("DIE-MNIST"). The primary goal was to adapt a proven modular machine learning pipeline for a new domain: binary image classification (Cats vs. Dogs).

## Core Philosophy
We retained the **"Separation of Concerns"** principle from the original project. Each stage of the ML lifecycle is handled by a dedicated, isolated module that communicates only through the shared file system. This ensures that changes in data collection do not break the training logic, provided the data contract (folder structure) is maintained.

## Transformation Strategy

### 1. Data Collection (`PetCollector`)
*   **Original:** A drawing canvas for manual digit input.
*   **New Approach:** A dual-mode GUI application.
    *   **Auto-Download:** Fetches images from `TheCatAPI` and `DogCEO`.
    *   **Manual Import:** Allows users to add local datasets.
    *   **Naming Convention:** Implemented sequential file naming (e.g., `dog_001.jpg`) to ensure unique and ordered filenames, preventing collisions between web-downloaded and manually imported files.

### 2. Data Preparation (`PetComposer`)
*   **Original:** Merged BMP files of drawn digits.
*   **New Approach:** A standardization pipeline with GUI control.
    *   **Normalization:** All images are resized to **64x64 pixels** and converted to **RGB**. This uniformity is crucial for the CNN input.
    *   **Interactive Splitting:** Users can dynamically configure the ratio of **Training**, **Validation**, and **Testing** sets via a GUI, ensuring flexible dataset management.

### 3. Model Training (`PetTrainer`)
*   **Original:** A simple CNN for 28x28/32x32 grayscale images (1 channel, 10 classes).
*   **New Approach:** Adapted CNN architecture with extensive configuration.
    *   **GUI Configuration:** A user-friendly interface allows tuning of hyperparameters (Epochs, Batch Size, Learning Rate, Dropout, etc.) via sliders.
    *   **CLI Support:** Enables headless training and automation via command-line arguments.
    *   **Input Layer:** Modified to accept **3 channels** (RGB).
    *   **Feature Extraction:** Deeper convolutional layers to capture complex features of animal fur and shapes compared to simple strokes of digits.
    *   **Output Layer:** Reduced to **2 neurons** (Cat, Dog) for binary classification.

### 4. Evaluation (`PetTester`)
*   **Original:** Multi-class confusion matrix for digits 0-9.
*   **New Approach:** Binary classification metrics.
    *   Visualizes the Confusion Matrix specifically for the two classes.
    *   Calculates Precision, Recall, and F1-Score to handle potential class imbalances.

## Technical Decisions
*   **Framework:** PyTorch was chosen for its flexibility in modifying tensor shapes and model layers.
*   **Storage:** The `shared/` directory acts as the "source of truth," decoupling the modules.
*   **User Experience:** Added GUIs (Tkinter) for the Collector to make dataset creation accessible to non-programmers.

---
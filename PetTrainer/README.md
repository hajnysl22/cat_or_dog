# PetTrainer

Trains a Convolutional Neural Network (CNN) to distinguish between Cats and Dogs.

## Features
- RGB Image support (64x64).
- Binary classification (Cat vs Dog).
- Training logs and model saving.

## Usage
1.  Ensure data is composed in `../shared/data/composed` (run `PetComposer`).
2.  Train the model:
    ```bash
    python train.py --epochs 10
    ```
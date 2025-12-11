# PetTrainer

Trains a Convolutional Neural Network (CNN) to distinguish between Cats and Dogs.

## Features
- **GUI Configuration:** Set Epochs, Batch Size, Learning Rate, Dropout, Step Size, LR Gamma, and Seed.
- **RGB Image support:** Handles 64x64 color images.
- **Binary classification:** Cat vs Dog.
- **Training logs:** Real-time feedback in the GUI and saved model checkpoints.

## Usage
1.  Ensure data is composed in `../shared/data/composed` (run `PetComposer`).
2.  Run the application:
    ```bash
    python main.py
    ```
    *(Or simply click `run.bat`)*

3.  Configure hyperparameters and click **"Start Training"**.

### CLI Usage (Advanced)
You can still run the training script directly from the command line. Supported arguments include:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--dropout`: Dropout rate
- `--step_size`: Step size for scheduler
- `--lr_gamma`: Gamma for scheduler
- `--seed`: Random seed
- `--use_cpu`: Force CPU usage

Example:
```bash
python train.py --epochs 20 --batch_size 32 --learning_rate 0.001 --use_cpu
```

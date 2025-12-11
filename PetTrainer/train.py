"""
Training script for Pet Classifier (Cat vs Dog).

Usage:
    python train.py --data_dir ../shared/data/composed --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

DEFAULT_CONFIG_PATH = Path("config.json")
DEFAULT_MODELS_DIR = Path("..") / "shared" / "models"
DEFAULT_DATA_DIR = Path("..") / "shared" / "data" / "composed"

@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float

@dataclass
class HyperParams:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    val_ratio: float = 0.0 # using directory split usually
    test_ratio: float = 0.0
    step_size: int = 10
    lr_gamma: float = 0.5
    dropout: float = 0.2
    seed: int = 42

    def update_from_dict(self, data: Dict[str, object]) -> None:
        for field in fields(self):
            if field.name in data:
                setattr(self, field.name, data[field.name])

    def apply_overrides(self, overrides: Dict[str, object]) -> None:
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def as_dict(self) -> Dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

class PetDataset(Dataset):
    """Dataset for Cat/Dog images."""

    def __init__(self, root_dir: Path, expected_size: Tuple[int, int] = (64, 64)) -> None:
        self.root_dir = root_dir
        self.expected_size = expected_size
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self._load_samples()

    def _load_samples(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.root_dir}' does not exist.")

        # Find classes
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        if not classes:
            # Fallback if maybe we are pointing to a specific split without class subdirs? 
            # Actually standard structure is root/class/img.
            pass

        for cls_name in classes:
            cls_dir = self.root_dir / cls_name
            idx = self.class_to_idx[cls_name]
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for image_path in sorted(cls_dir.glob(ext)):
                    self.samples.append((image_path, idx))

        if not self.samples:
            # Maybe the root dir itself contains the images (not standard per our composer but handle gracefully?)
            pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        try:
            image = Image.open(image_path).convert("RGB")
            
            if image.size != self.expected_size:
                image = image.resize(self.expected_size)

            # HWC to CHW
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)

            return tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a zero tensor in case of error to avoid crashing
            return torch.zeros((3, *self.expected_size)), torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    """CNN for 64x64 RGB images."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.features = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32 x 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 8 x 8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_loaders_from_subdirectories(
    base_dir: Path,
    batch_size: int,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_dir = base_dir / split
        if split_dir.exists() and split_dir.is_dir():
            subset = PetDataset(split_dir)
            if len(subset) == 0:
                continue
            loaders[split] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=0,
            )
    return loaders

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += inputs.size(0)
        
        progress.set_postfix(loss=running_loss/total, acc=correct/total)

    return running_loss / total if total else 0.0, correct / total if total else 0.0

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += inputs.size(0)

    return running_loss / total if total else 0.0, correct / total if total else 0.0

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--lr_gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    hp = HyperParams()
    if args.epochs: hp.epochs = args.epochs
    if args.batch_size: hp.batch_size = args.batch_size
    if args.learning_rate: hp.learning_rate = args.learning_rate
    if args.dropout: hp.dropout = args.dropout
    if args.step_size: hp.step_size = args.step_size
    if args.lr_gamma: hp.lr_gamma = args.lr_gamma
    if args.seed: hp.seed = args.seed

    set_seed(hp.seed)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    loaders = create_loaders_from_subdirectories(args.data_dir, hp.batch_size)
    if "train" not in loaders:
        print("No training data found in", args.data_dir / "train")
        return

    train_loader = loaders["train"]
    # Infer num classes
    num_classes = len(train_loader.dataset.class_to_idx)
    print(f"Classes: {train_loader.dataset.class_to_idx}")

    model = SimpleCNN(num_classes=num_classes, dropout=hp.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    
    # Check for existing model to resume? No, simplified for this task.
    
    output_dir = args.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    
    for epoch in range(1, hp.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, hp.epochs)
        
        v_loss, v_acc = 0.0, 0.0
        if "test" in loaders: # Using test as validation if val not present
            v_loss, v_acc = evaluate(model, loaders["test"], criterion, device)
            
        print(f"Epoch {epoch}: Train Loss={t_loss:.4f} Acc={t_acc:.4f} | Val Loss={v_loss:.4f} Acc={v_acc:.4f}")
        
        history.append({"epoch": epoch, "train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc})

    # Save model
    torch.save(model.state_dict(), output_dir / "pet_cnn.pt")
    
    # Save class mapping
    with open(output_dir / "classes.json", "w") as f:
        json.dump(train_loader.dataset.class_to_idx, f)
        
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
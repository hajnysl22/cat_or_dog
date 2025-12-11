"""
PetTester - Evaluation tool for Pet Classifier.

CLI tool to evaluate models from PetTrainer on test data.
Calculates metrics including confusion matrix and per-class accuracy.

Usage:
    python main.py                    # Interactive selection
    python main.py --model_dir PATH   # Explicit model
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class SimpleCNN(nn.Module):
    """CNN for 64x64 RGB images."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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


class PetDataset(Dataset):
    """Dataset for Cat/Dog images."""

    def __init__(self, root_dirs: List[Path], expected_size: Tuple[int, int] = (64, 64)) -> None:
        if isinstance(root_dirs, Path):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.expected_size = expected_size
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        # We need to know class mapping. For testing, we usually infer from directory structure 
        # or it should be passed. Here we assume standard structure or specific class folders.
        # But wait, different root_dirs might have different structures?
        # Let's assume standard 'cat'/'dog' folders exist in the root_dirs.
        
        # Hardcoded for now to match training, or we can try to detect.
        # Ideally we should pass the class_to_idx from the model.
        # But the dataset loader needs to know what label 0/1 means when loading.
        # Let's use a standard mapping sorted alphabetically for consistency.
        self.class_to_idx = {"cat": 0, "dog": 1} 
        
        for root_dir in self.root_dirs:
            if not root_dir.exists():
                continue

            for cls_name, idx in self.class_to_idx.items():
                cls_dir = root_dir / cls_name
                if not cls_dir.exists():
                    continue
                    
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    for img_path in sorted(cls_dir.glob(ext)):
                        self.samples.append((img_path, idx))

        if not self.samples:
            # Fallback for flat structure? Not supported for now.
            pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != self.expected_size:
                img = img.resize(self.expected_size)
            
            array = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            return tensor, label, str(img_path)
        except Exception:
            return torch.zeros((3, *self.expected_size)), label, str(img_path)


def load_model(model_dir: Path, device: torch.device) -> Tuple[nn.Module, Dict, Dict]:
    config_path = model_dir / "config.json"
    model_path = model_dir / "pet_cnn.pt" # Changed name
    if not model_path.exists():
        model_path = model_dir / "digit_cnn.pt" # Fallback

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    config = {}
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)

    # Load classes if available
    classes_path = model_dir / "classes.json"
    class_map = {}
    if classes_path.exists():
        with classes_path.open("r") as f:
            class_map = json.load(f)
    else:
        class_map = {"cat": 0, "dog": 1} # Default

    num_classes = len(class_map)
    model = SimpleCNN(num_classes=num_classes, dropout=config.get("dropout", 0.2))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, config, class_map


def evaluate_with_confusion(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, np.ndarray, List[Dict[str, object]]]:
    model.eval()
    running_loss = 0.0
    total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            _, predicted = torch.max(outputs, 1)
            for path, true_label, pred_label in zip(paths, labels.cpu().numpy(), predicted.cpu().numpy()):
                if true_label < num_classes and pred_label < num_classes:
                    confusion[true_label][pred_label] += 1
                predictions.append({
                    "path": path,
                    "true": int(true_label),
                    "pred": int(pred_label)
                })

    average_loss = running_loss / total if total else 0.0
    accuracy = confusion.trace() / total if total else 0.0

    return average_loss, accuracy, confusion, predictions


def save_results(results: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


def print_results(results: Dict, idx_to_class: Dict[int, str]) -> None:
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Average Loss:     {results['average_loss']:.4f}")
    
    print("\nCONFUSION MATRIX")
    confusion = np.array(results["confusion_matrix"])
    num_classes = len(idx_to_class)
    
    # Headers
    print("      ", end="")
    for i in range(num_classes):
        print(f"{idx_to_class.get(i, str(i)):>8}", end="")
    print()
    
    for i in range(num_classes):
        print(f"{idx_to_class.get(i, str(i)):>6}|", end="")
        for j in range(num_classes):
            print(f"{confusion[i, j]:>8}", end="")
        print()
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--data_dir", type=Path, default=Path("..") / "shared" / "data" / "composed" / "test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Find model if not specified
    if args.model_dir is None:
        models_dir = Path("..") / "shared" / "models"
        if not models_dir.exists():
            print("No models found.")
            return
        runs = sorted([d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
        if not runs:
            print("No runs found.")
            return
        args.model_dir = runs[0]
        print(f"Using latest model: {args.model_dir}")

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    model, config, class_map = load_model(args.model_dir, device)
    idx_to_class = {v: k for k, v in class_map.items()}
    num_classes = len(class_map)

    print(f"Loading test data from: {args.data_dir}")
    dataset = PetDataset(args.data_dir)
    if len(dataset) == 0:
        print("No data found.")
        return
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Samples: {len(dataset)}")

    avg_loss, accuracy, confusion, predictions = evaluate_with_confusion(
        model, loader, nn.CrossEntropyLoss(), device, num_classes
    )

    results = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_dir": str(args.model_dir),
        "data_dir": str(args.data_dir),
        "overall_accuracy": float(accuracy),
        "average_loss": float(avg_loss),
        "confusion_matrix": confusion.tolist(),
        "predictions": predictions,
        "class_map": class_map
    }

    print_results(results, idx_to_class)

    if args.output:
        save_results(results, args.output)
    else:
        save_results(results, Path("..") / "shared" / "tests" / f"results_{results['timestamp']}.json")

if __name__ == "__main__":
    main()
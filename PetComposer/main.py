import os
import random
import shutil
from pathlib import Path
from PIL import Image

# Configuration
RAW_DIR = Path("../shared/data/raw")
COMPOSED_DIR = Path("../shared/data/composed")
IMG_SIZE = (64, 64)
SPLIT_RATIO = 0.8  # 80% train, 20% test
CLASSES = ["cat", "dog"]

def setup_directories():
    if COMPOSED_DIR.exists():
        shutil.rmtree(COMPOSED_DIR)
    
    for split in ["train", "test"]:
        for cls in CLASSES:
            (COMPOSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"Created directories in {COMPOSED_DIR}")

def process_images():
    for cls in CLASSES:
        src_dir = RAW_DIR / cls
        if not src_dir.exists():
            print(f"Warning: Source directory {src_dir} does not exist.")
            continue
        
        images = [f for f in src_dir.iterdir() if f.is_file()]
        random.shuffle(images)
        
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]
        
        print(f"Processing {cls}: {len(train_imgs)} train, {len(test_imgs)} test")
        
        for img_path in train_imgs:
            save_image(img_path, COMPOSED_DIR / "train" / cls, IMG_SIZE)
            
        for img_path in test_imgs:
            save_image(img_path, COMPOSED_DIR / "test" / cls, IMG_SIZE)

def save_image(src_path, dest_dir, size):
    try:
        with Image.open(src_path) as img:
            img = img.convert('RGB')
            img = img.resize(size, Image.Resampling.LANCZOS)
            dest_path = dest_dir / src_path.name
            img.save(dest_path)
    except Exception as e:
        print(f"Failed to process {src_path}: {e}")

if __name__ == "__main__":
    setup_directories()
    process_images()
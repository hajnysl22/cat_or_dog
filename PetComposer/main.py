import os
import random
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image

# Configuration
RAW_DIR = Path("../shared/data/raw")
COMPOSED_DIR = Path("../shared/data/composed")
IMG_SIZE = (64, 64)
CLASSES = ["cat", "dog"]

class PetComposerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PetComposer")
        self.root.geometry("500x450")
        
        # Header
        header = tk.Label(root, text="Dataset Composer", font=("Arial", 16, "bold"), pady=10)
        header.pack()

        # Split Configuration Frame
        split_frame = tk.LabelFrame(root, text="Data Splits (%)", padx=15, pady=10)
        split_frame.pack(fill="x", padx=20, pady=10)

        # Train Split
        tk.Label(split_frame, text="Train:").grid(row=0, column=0, sticky="w")
        self.train_val = tk.IntVar(value=70)
        self.train_scale = tk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.train_val, command=self.update_totals)
        self.train_scale.grid(row=0, column=1, sticky="ew", padx=10)
        self.lbl_train = tk.Label(split_frame, text="70%")
        self.lbl_train.grid(row=0, column=2)

        # Validation Split
        tk.Label(split_frame, text="Validation:").grid(row=1, column=0, sticky="w")
        self.val_val = tk.IntVar(value=15)
        self.val_scale = tk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.val_val, command=self.update_totals)
        self.val_scale.grid(row=1, column=1, sticky="ew", padx=10)
        self.lbl_val = tk.Label(split_frame, text="15%")
        self.lbl_val.grid(row=1, column=2)

        # Test Split
        tk.Label(split_frame, text="Test:").grid(row=2, column=0, sticky="w")
        self.test_val = tk.IntVar(value=15)
        self.test_scale = tk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.test_val, command=self.update_totals)
        self.test_scale.grid(row=2, column=1, sticky="ew", padx=10)
        self.lbl_test = tk.Label(split_frame, text="15%")
        self.lbl_test.grid(row=2, column=2)

        split_frame.columnconfigure(1, weight=1)

        # Total Indicator
        self.lbl_total = tk.Label(root, text="Total: 100%", font=("Arial", 10, "bold"), fg="green")
        self.lbl_total.pack(pady=5)

        # Action Button
        self.btn_process = tk.Button(root, text="Process Dataset", command=self.start_processing, bg="#4caf50", fg="white", font=("Arial", 11, "bold"), height=2)
        self.btn_process.pack(fill="x", padx=40, pady=10)

        # Log
        self.log_text = tk.Text(root, height=8, state='disabled', bg="#f0f0f0", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)

        self.update_totals()

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def update_totals(self, _=None):
        t = self.train_val.get()
        v = self.val_val.get()
        te = self.test_val.get()
        
        self.lbl_train.config(text=f"{t}%")
        self.lbl_val.config(text=f"{v}%")
        self.lbl_test.config(text=f"{te}%")
        
        total = t + v + te
        self.lbl_total.config(text=f"Total: {total}%")
        
        if total == 100:
            self.lbl_total.config(fg="green")
            self.btn_process.config(state="normal")
        else:
            self.lbl_total.config(fg="red")
            self.btn_process.config(state="disabled")

    def start_processing(self):
        t = self.train_val.get()
        v = self.val_val.get()
        te = self.test_val.get()
        
        if t + v + te != 100:
            messagebox.showerror("Error", "Splits must sum to 100%")
            return

        self.btn_process.config(state="disabled")
        threading.Thread(target=self.run_process, args=(t/100, v/100, te/100), daemon=True).start()

    def run_process(self, train_ratio, val_ratio, test_ratio):
        self.log(f"Starting composition: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")
        
        try:
            self.setup_directories()
            
            for cls in CLASSES:
                src_dir = RAW_DIR / cls
                if not src_dir.exists():
                    self.log(f"Warning: {src_dir} not found.")
                    continue
                
                images = [f for f in src_dir.iterdir() if f.is_file()]
                random.shuffle(images)
                
                total = len(images)
                n_train = int(total * train_ratio)
                n_val = int(total * val_ratio)
                # Remaining goes to test to avoid rounding errors
                
                train_imgs = images[:n_train]
                val_imgs = images[n_train:n_train+n_val]
                test_imgs = images[n_train+n_val:]
                
                self.log(f"Processing {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
                
                for img in train_imgs: self.save_image(img, COMPOSED_DIR / "train" / cls)
                for img in val_imgs: self.save_image(img, COMPOSED_DIR / "val" / cls)
                for img in test_imgs: self.save_image(img, COMPOSED_DIR / "test" / cls)
                
            self.log("Dataset composition complete!")
            messagebox.showinfo("Success", "Dataset created successfully!")
            
        except Exception as e:
            self.log(f"Error: {e}")
            print(e)
        finally:
            self.btn_process.config(state="normal")

    def setup_directories(self):
        if COMPOSED_DIR.exists():
            shutil.rmtree(COMPOSED_DIR)
        
        for split in ["train", "val", "test"]:
            for cls in CLASSES:
                (COMPOSED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    def save_image(self, src_path, dest_dir):
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
                dest_path = dest_dir / src_path.name
                img.save(dest_path)
        except Exception as e:
            self.log(f"Failed to save {src_path.name}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PetComposerApp(root)
    root.mainloop()

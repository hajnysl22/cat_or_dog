import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import requests
import time
import re

# Configuration
CATS_URL = "https://api.thecatapi.com/v1/images/search"
DOGS_URL = "https://dog.ceo/api/breeds/image/random"
DATA_DIR = Path("../shared/data/raw")

class PetCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PetCollector")
        self.root.geometry("600x500")
        
        self.setup_directories()
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header = tk.Label(root, text="Pet Data Collector", font=("Arial", 18, "bold"), pady=10)
        header.pack()

        # Stats Frame
        stats_frame = tk.Frame(root)
        stats_frame.pack(pady=10)
        
        self.cat_count_label = tk.Label(stats_frame, text="Cats: 0", font=("Arial", 12))
        self.cat_count_label.pack(side=tk.LEFT, padx=20)
        
        self.dog_count_label = tk.Label(stats_frame, text="Dogs: 0", font=("Arial", 12))
        self.dog_count_label.pack(side=tk.LEFT, padx=20)
        
        # Manual Import Frame
        import_frame = tk.LabelFrame(root, text="Manual Import", padx=10, pady=10)
        import_frame.pack(fill="x", padx=20, pady=10)
        
        btn_cat = tk.Button(import_frame, text="Import Cat Images", command=self.import_cats, bg="#e1bee7", height=2)
        btn_cat.pack(fill="x", pady=5)
        
        btn_dog = tk.Button(import_frame, text="Import Dog Images", command=self.import_dogs, bg="#bbdefb", height=2)
        btn_dog.pack(fill="x", pady=5)
        
        # Auto Download Frame
        download_frame = tk.LabelFrame(root, text="Auto Download (Web)", padx=10, pady=10)
        download_frame.pack(fill="x", padx=20, pady=10)
        
        self.spin_count = tk.Spinbox(download_frame, from_=1, to=500, width=5)
        self.spin_count.delete(0, "end")
        self.spin_count.insert(0, 50)
        self.spin_count.pack(side=tk.LEFT, padx=5)
        
        tk.Label(download_frame, text="images per class").pack(side=tk.LEFT)
        
        self.btn_download = tk.Button(download_frame, text="Start Download", command=self.start_download_thread, bg="#c8e6c9")
        self.btn_download.pack(side=tk.RIGHT)

        # Log Area
        self.log_text = tk.Text(root, height=10, state='disabled', bg="#f0f0f0", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)

        self.update_counts()

    def setup_directories(self):
        (DATA_DIR / "cat").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "dog").mkdir(parents=True, exist_ok=True)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def update_counts(self):
        n_cats = len(list((DATA_DIR / "cat").glob("*")))
        n_dogs = len(list((DATA_DIR / "dog").glob("*")))
        self.cat_count_label.config(text=f"Cats: {n_cats}")
        self.dog_count_label.config(text=f"Dogs: {n_dogs}")

    def get_next_filename(self, directory, prefix, ext):
        """Finds the next available filename like prefix_001.ext"""
        # List all files matching the pattern
        existing_files = list(directory.glob(f"{prefix}_*{ext}"))
        
        max_idx = -1
        pattern = re.compile(rf"{prefix}_(\d+)")
        
        for f in directory.glob("*"):
            if f.suffix.lower() == ext.lower():
                match = pattern.match(f.stem)
                if match:
                    try:
                        idx = int(match.group(1))
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        pass
        
        return f"{prefix}_{max_idx + 1:03d}{ext}"

    def import_images(self, species):
        files = filedialog.askopenfilenames(
            title=f"Select {species} images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if not files:
            return
            
        dest_dir = DATA_DIR / species.lower()
        prefix = species.lower()
        count = 0
        
        for src in files:
            try:
                src_path = Path(src)
                ext = src_path.suffix.lower()
                if not ext: ext = ".jpg"
                
                new_name = self.get_next_filename(dest_dir, prefix, ext)
                shutil.copy2(src, dest_dir / new_name)
                count += 1
            except Exception as e:
                self.log(f"Error copying {src}: {e}")
        
        self.log(f"Imported {count} {species} images.")
        self.update_counts()

    def import_cats(self):
        self.import_images("Cat")

    def import_dogs(self):
        self.import_images("Dog")

    def start_download_thread(self):
        count = int(self.spin_count.get())
        threading.Thread(target=self.run_download, args=(count,), daemon=True).start()

    def run_download(self, count):
        self.btn_download.config(state='disabled')
        self.log(f"Starting download of {count} cats and {count} dogs...")
        
        self.fetch_cats(count)
        self.fetch_dogs(count)
        
        self.log("Download complete.")
        self.btn_download.config(state='normal')
        self.root.after(0, self.update_counts)

    def download_image(self, url, save_path):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception:
            pass
        return False

    def fetch_cats(self, count):
        saved = 0
        attempts = 0
        prefix = "cat"
        dest_dir = DATA_DIR / "cat"
        
        while saved < count and attempts < count * 2:
            try:
                resp = requests.get(CATS_URL).json()
                url = resp[0]['url']
                ext = os.path.splitext(url)[1].lower()
                if not ext: ext = ".jpg"
                
                filename = self.get_next_filename(dest_dir, prefix, ext)
                filepath = dest_dir / filename
                
                if self.download_image(url, filepath):
                    saved += 1
                    if saved % 5 == 0:
                        self.log(f"Saved cat {saved}/{count} as {filename}")
                        self.root.after(0, self.update_counts)
                time.sleep(0.5)
            except Exception:
                pass
            attempts += 1

    def fetch_dogs(self, count):
        saved = 0
        attempts = 0
        prefix = "dog"
        dest_dir = DATA_DIR / "dog"
        
        while saved < count and attempts < count * 2:
            try:
                resp = requests.get(DOGS_URL).json()
                if resp['status'] == 'success':
                    url = resp['message']
                    ext = os.path.splitext(url)[1].lower()
                    if not ext: ext = ".jpg"
                    
                    filename = self.get_next_filename(dest_dir, prefix, ext)
                    filepath = dest_dir / filename
                    
                    if self.download_image(url, filepath):
                        saved += 1
                        if saved % 5 == 0:
                            self.log(f"Saved dog {saved}/{count} as {filename}")
                            self.root.after(0, self.update_counts)
                time.sleep(0.5)
            except Exception:
                pass
            attempts += 1

if __name__ == "__main__":
    root = tk.Tk()
    app = PetCollectorApp(root)
    root.mainloop()
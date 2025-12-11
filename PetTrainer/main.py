import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import sys
from pathlib import Path

class PetTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PetTrainer")
        self.root.geometry("600x850") # Increased height again

        # Header
        header = tk.Label(root, text="Model Training Configuration", font=("Arial", 16, "bold"), pady=10)
        header.pack()

        # Parameters Frame
        params_frame = tk.LabelFrame(root, text="Hyperparameters", padx=15, pady=10)
        params_frame.pack(fill="x", padx=20, pady=10)

        self.entries = {}
        
        # Format: (Label, Default, Data Type, CLI Argument, Min, Max, Resolution)
        params = [
            ("Epochs", 20, int, "--epochs", 1, 100, 1),
            ("Batch Size", 32, int, "--batch_size", 1, 128, 1),
            ("Learning Rate", 0.001, float, "--learning_rate", 0.0001, 0.01, 0.0001),
            ("Dropout", 0.2, float, "--dropout", 0.0, 0.9, 0.05),
            ("Step Size", 10, int, "--step_size", 1, 50, 1),
            ("LR Gamma", 0.5, float, "--lr_gamma", 0.1, 1.0, 0.05),
            ("Seed", 42, int, "--seed", 0, 9999, 1),
        ]

        for i, (label, default, dtype, cli_arg, min_val, max_val, res) in enumerate(params):
            # Label
            tk.Label(params_frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=5)
            
            # Variable
            if dtype is int:
                var = tk.IntVar(value=default)
            else:
                var = tk.DoubleVar(value=default)

            # Slider
            scale = tk.Scale(params_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                             resolution=res, variable=var, showvalue=0)
            scale.grid(row=i, column=1, sticky="ew", padx=10)

            # Entry
            entry = tk.Entry(params_frame, textvariable=var, width=8)
            entry.grid(row=i, column=2, sticky="e", padx=5)
            
            # Store reference
            self.entries[label] = (var, cli_arg)
            
        params_frame.columnconfigure(1, weight=1)

        # Advanced Frame
        adv_frame = tk.LabelFrame(root, text="System", padx=15, pady=10)
        adv_frame.pack(fill="x", padx=20, pady=10)
        
        self.use_cpu = tk.BooleanVar(value=False)
        tk.Checkbutton(adv_frame, text="Force CPU", variable=self.use_cpu).pack(anchor="w")

        # Action Button
        self.btn_train = tk.Button(root, text="Start Training", command=self.start_training, bg="#2196f3", fg="white", font=("Arial", 11, "bold"), height=2)
        self.btn_train.pack(fill="x", padx=40, pady=10)

        # Log Area
        self.log_text = tk.Text(root, height=12, state='disabled', bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.process = None

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def start_training(self):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Running", "Training is already in progress.")
            return

        # Build command
        cmd = [sys.executable, "train.py"]
        
        try:
            for label, (var, cli_arg) in self.entries.items():
                value = var.get()
                cmd.extend([cli_arg, str(value)])
            
            if self.use_cpu.get():
                cmd.append("--use_cpu")
                
        except ValueError:
            messagebox.showerror("Error", f"Invalid input values.")
            return

        self.log(f"Starting: {' '.join(cmd)}")
        self.btn_train.config(state='disabled', text="Training in progress...")
        
        threading.Thread(target=self.run_process, args=(cmd,), daemon=True).start()

    def run_process(self, cmd):
        try:
            # We capture stdout and stderr
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            for line in self.process.stdout:
                self.root.after(0, self.log, line.strip())

            self.process.wait()
            ret = self.process.returncode
            
            self.root.after(0, lambda: self.training_finished(ret))

        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, lambda: self.training_finished(-1))

    def training_finished(self, return_code):
        self.btn_train.config(state='normal', text="Start Training")
        if return_code == 0:
            messagebox.showinfo("Success", "Training completed successfully!")
            self.log("=== DONE ===")
        else:
            messagebox.showerror("Failed", f"Training failed with code {return_code}")
            self.log(f"=== FAILED (Code {return_code}) ===")

if __name__ == "__main__":
    root = tk.Tk()
    app = PetTrainerApp(root)
    root.mainloop()

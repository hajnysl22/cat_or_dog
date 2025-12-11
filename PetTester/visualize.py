"""
PetTester Visualizer - GUI for visualizing test results

Displays results from a JSON file created by PetTester:
- Overall metrics with color indicator
- Confusion matrix heatmap (interactive)
- Per-class bar charts
- Top confusions list
- Export charts
"""

from __future__ import annotations

import json
import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, filedialog
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

# Set matplotlib to use TkAgg backend and configure for clean shutdown
matplotlib.use('TkAgg')
plt.ioff()  # Turn off interactive mode to prevent lingering processes

# Dark UI palette matching other apps
BG_COLOR = "#2b2b2b"
FG_COLOR = "#d8dde6"
CANVAS_BORDER = "#43464a"
BUTTON_BG = "#3b3f42"
BUTTON_FG = "#e6e9ef"
BUTTON_ACTIVE_BG = "#4a4f53"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_BORDER = "#5a5d60"


def load_results(results_path: Path) -> Dict:
    """Loads JSON with test results."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_accuracy_color(accuracy: float) -> str:
    """Returns color based on accuracy level."""
    if accuracy >= 0.90:
        return "#4CAF50"  # Green
    elif accuracy >= 0.70:
        return "#FFC107"  # Yellow
    else:
        return "#F44336"  # Red


class ResultsVisualizer:
    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.results = load_results(results_path)

        self.window = tk.Tk()
        self.window.title("PetTester - Test Results")
        self.window.geometry("1400x900")
        self.window.config(bg=BG_COLOR, padx=15, pady=15)

        # Matplotlib figures
        self.confusion_fig = None
        self.confusion_ax = None
        self.confusion_canvas = None

        self.metrics_fig = None
        self.metrics_ax = None

        # Set protocol for window closing
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._build_ui()

        # Center window after building GUI
        self.center_window()

    def _build_ui(self):
        """Builds the entire GUI."""
        # Header
        self._create_header()

        # Overall metrics panel
        self._create_overall_panel()

        # Main content area (confusion matrix + per-class charts)
        content_frame = tk.Frame(self.window, bg=BG_COLOR)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left: Confusion matrix
        left_frame = tk.Frame(content_frame, bg=BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self._create_confusion_heatmap(left_frame)

        # Right: Per-class charts + confusions list
        right_frame = tk.Frame(content_frame, bg=BG_COLOR)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._create_perclass_charts(right_frame)
        self._create_confusions_list(right_frame)

        # Bottom: Export button
        self._create_export_button()

    def _create_header(self):
        """Creates header with model info."""
        header_frame = tk.Frame(self.window, bg=BG_COLOR)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        title = tk.Label(
            header_frame,
            text="📊 Model Test Results",
            font=("Arial", 20, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        title.pack(anchor="w")

        # Model info
        model_dir = Path(self.results["model_dir"]).name
        data_dir = Path(self.results["data_dir"]).name
        timestamp = self.results["timestamp"]

        info_text = f"Model: {model_dir}  |  Data: {data_dir}  |  Time: {timestamp}"
        info_label = tk.Label(
            header_frame,
            text=info_text,
            font=("Arial", 11),
            bg=BG_COLOR,
            fg="#aaaaaa",
        )
        info_label.pack(anchor="w", pady=(5, 0))

    def _create_overall_panel(self):
        """Creates panel with overall metrics."""
        panel = tk.Frame(self.window, bg=BUTTON_BG, relief="flat", bd=0)
        panel.pack(fill=tk.X, pady=(0, 15))

        # Padding inside panel
        inner = tk.Frame(panel, bg=BUTTON_BG)
        inner.pack(padx=20, pady=15)

        accuracy = self.results["overall_accuracy"]
        loss = self.results["average_loss"]
        total = self.results["total_samples"]

        # Accuracy with color
        acc_color = get_accuracy_color(accuracy)
        acc_label = tk.Label(
            inner,
            text=f"Overall Accuracy: {accuracy*100:.2f}%",
            font=("Arial", 18, "bold"),
            bg=BUTTON_BG,
            fg=acc_color,
        )
        acc_label.pack(side=tk.LEFT, padx=20)

        # Loss
        loss_label = tk.Label(
            inner,
            text=f"Average Loss: {loss:.4f}",
            font=("Arial", 14),
            bg=BUTTON_BG,
            fg=FG_COLOR,
        )
        loss_label.pack(side=tk.LEFT, padx=20)

        # Total samples
        total_label = tk.Label(
            inner,
            text=f"Samples: {total}",
            font=("Arial", 14),
            bg=BUTTON_BG,
            fg=FG_COLOR,
        )
        total_label.pack(side=tk.LEFT, padx=20)

    def _create_confusion_heatmap(self, parent):
        """Creates confusion matrix as heatmap."""
        label = tk.Label(
            parent,
            text="Confusion Matrix (click cell for examples)",
            font=("Arial", 13, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        label.pack(anchor="w", pady=(0, 5))

        confusion = np.array(self.results["confusion_matrix"])
        num_classes = confusion.shape[0]

        # Create figure with dark background
        self.confusion_fig = Figure(figsize=(6, 6), facecolor=BG_COLOR)
        self.confusion_ax = self.confusion_fig.add_subplot(111)
        self.confusion_ax.set_facecolor(BG_COLOR)

        # Plot heatmap
        im = self.confusion_ax.imshow(confusion, cmap='YlGn', aspect='auto')

        # Colorbar
        cbar = self.confusion_fig.colorbar(im, ax=self.confusion_ax)
        cbar.ax.yaxis.set_tick_params(color=FG_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=FG_COLOR)

        # Axes
        self.confusion_ax.set_xticks(np.arange(num_classes))
        self.confusion_ax.set_yticks(np.arange(num_classes))
        self.confusion_ax.set_xticklabels(range(num_classes), color=FG_COLOR)
        self.confusion_ax.set_yticklabels(range(num_classes), color=FG_COLOR)

        # Labels
        self.confusion_ax.set_xlabel("Prediction", color=FG_COLOR, fontsize=11)
        self.confusion_ax.set_ylabel("True Label", color=FG_COLOR, fontsize=11)

        # Add text annotations
        for i in range(num_classes):
            for j in range(num_classes):
                text_color = "black" if confusion[i, j] > confusion.max() / 2 else "white"
                self.confusion_ax.text(j, i, str(confusion[i, j]),
                                     ha="center", va="center", color=text_color, fontsize=10)

        self.confusion_fig.tight_layout()

        # Embed in tkinter
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_fig, master=parent)
        self.confusion_canvas.draw()
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Make interactive
        self.confusion_canvas.mpl_connect('button_press_event', self._on_confusion_click)

    def _create_perclass_charts(self, parent):
        """Creates bar charts for per-class metrics."""
        label = tk.Label(
            parent,
            text="Per-Class Metrics",
            font=("Arial", 13, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        label.pack(anchor="w", pady=(0, 5))

        metrics = self.results.get("per_class_metrics", {})
        if not metrics:
             tk.Label(parent, text="No per-class metrics available", bg=BG_COLOR, fg=FG_COLOR).pack()
             return

        classes = sorted([int(k) for k in metrics.keys()])

        accuracies = [metrics[str(c)]["accuracy"] for c in classes]
        precisions = [metrics[str(c)]["precision"] for c in classes]
        recalls = [metrics[str(c)]["recall"] for c in classes]
        f1_scores = [metrics[str(c)]["f1_score"] for c in classes]

        # Create figure
        self.metrics_fig = Figure(figsize=(6, 4), facecolor=BG_COLOR)
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        self.metrics_ax.set_facecolor(BG_COLOR)

        x = np.arange(len(classes))
        width = 0.2

        self.metrics_ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#4CAF50')
        self.metrics_ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#2196F3')
        self.metrics_ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#FF9800')
        self.metrics_ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#9C27B0')

        self.metrics_ax.set_xlabel("Class", color=FG_COLOR, fontsize=10)
        self.metrics_ax.set_ylabel("Value", color=FG_COLOR, fontsize=10)
        self.metrics_ax.set_xticks(x)
        self.metrics_ax.set_xticklabels(classes, color=FG_COLOR)
        self.metrics_ax.tick_params(colors=FG_COLOR)
        self.metrics_ax.legend(facecolor=BUTTON_BG, edgecolor=CANVAS_BORDER,
                              labelcolor=FG_COLOR, fontsize=9)
        self.metrics_ax.set_ylim(0, 1.1)
        self.metrics_ax.grid(axis='y', alpha=0.3, color=CANVAS_BORDER)

        # Set spine colors
        for spine in self.metrics_ax.spines.values():
            spine.set_edgecolor(CANVAS_BORDER)

        self.metrics_fig.tight_layout()

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(self.metrics_fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    def _create_confusions_list(self, parent):
        """Creates a list of most frequent confusions."""
        label = tk.Label(
            parent,
            text="Most Frequent Errors",
            font=("Arial", 13, "bold"),
            bg=BG_COLOR,
            fg=FG_COLOR,
        )
        label.pack(anchor="w", pady=(10, 5))

        # Find top confusions (off-diagonal)
        confusion = np.array(self.results["confusion_matrix"])
        confusions = []

        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                if i != j and confusion[i, j] > 0:
                    confusions.append((i, j, confusion[i, j]))

        confusions.sort(key=lambda x: x[2], reverse=True)
        top_confusions = confusions[:5]

        # Display in a frame
        list_frame = tk.Frame(parent, bg=BUTTON_BG, relief="flat", bd=0)
        list_frame.pack(fill=tk.X, pady=(0, 10))

        if not top_confusions:
            no_errors = tk.Label(
                list_frame,
                text="🎉 No Errors!",
                font=("Arial", 12),
                bg=BUTTON_BG,
                fg="#4CAF50",
            )
            no_errors.pack(padx=15, pady=15)
        else:
            for idx, (true_cls, pred_cls, count) in enumerate(top_confusions, 1):
                text = f"{idx}. Class {true_cls} → {pred_cls}: {count}× wrong"
                lbl = tk.Label(
                    list_frame,
                    text=text,
                    font=("Courier", 11),
                    bg=BUTTON_BG,
                    fg=FG_COLOR,
                    anchor="w",
                )
                lbl.pack(fill=tk.X, padx=15, pady=3)

    def _create_export_button(self):
        """Creates button to export charts."""
        btn_frame = tk.Frame(self.window, bg=BG_COLOR)
        btn_frame.pack(pady=(10, 0))

        export_btn = tk.Button(
            btn_frame,
            text="💾 Export Charts",
            command=self._export_charts,
            width=20,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BUTTON_BORDER,
            font=("Arial", 12),
        )
        export_btn.pack()

    def _on_confusion_click(self, event):
        """Handler for clicking on confusion matrix."""
        if event.inaxes != self.confusion_ax:
            return

        # Get clicked cell
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        confusion = np.array(self.results["confusion_matrix"])

        if row < 0 or row >= confusion.shape[0] or col < 0 or col >= confusion.shape[1]:
            return

        count = confusion[row, col]

        if count == 0:
            messagebox.showinfo(
                "No Cases",
                f"No samples were classified as:\nTrue: {row}, Predicted: {col}"
            )
            return

        # Show examples
        self._show_confusion_examples(row, col, count)

    def _show_confusion_examples(self, true_class: int, pred_class: int, total_count: int):
        """Shows random examples of specific confusion."""
        # Find all predictions matching this confusion
        predictions = self.results.get("predictions", [])

        matching = [
            p for p in predictions
            if p["true"] == true_class and p["pred"] == pred_class
        ]

        if not matching:
            messagebox.showwarning(
                "No Data",
                f"Prediction data not available.\n"
                f"(You might be using older results version)"
            )
            return

        # Sample random examples (max 6)
        num_examples = min(6, len(matching))
        examples = random.sample(matching, num_examples)

        # Create window
        ex_window = tk.Toplevel(self.window)
        ex_window.title(f"Examples: {true_class} → {pred_class}")
        ex_window.config(bg=BG_COLOR, padx=20, pady=20)
        ex_window.transient(self.window)

        # Header
        if true_class == pred_class:
            header_text = f"✓ Correctly classified as {true_class} ({total_count}× total)"
            header_color = "#4CAF50"
        else:
            header_text = f"✗ True: {true_class}, Model said: {pred_class} ({total_count}× total)"
            header_color = "#F44336"

        header = tk.Label(
            ex_window,
            text=header_text,
            font=("Arial", 14, "bold"),
            bg=BG_COLOR,
            fg=header_color,
        )
        header.pack(pady=(0, 15))

        # Grid of examples
        grid_frame = tk.Frame(ex_window, bg=BG_COLOR)
        grid_frame.pack()

        for idx, example in enumerate(examples):
            row_idx = idx // 3
            col_idx = idx % 3

            try:
                img_path = Path(example["path"])
                img = Image.open(img_path).convert("L")
                img = img.resize((96, 96), Image.NEAREST)
                photo = ImageTk.PhotoImage(img)

                canvas = tk.Canvas(
                    grid_frame,
                    width=96,
                    height=96,
                    bg="black",
                    highlightthickness=2,
                    highlightbackground=CANVAS_BORDER,
                )
                canvas.create_image(0, 0, image=photo, anchor="nw")
                canvas.image = photo  # Keep reference
                canvas.grid(row=row_idx, column=col_idx, padx=5, pady=5)

            except Exception as e:
                # If can't load image, show error
                error_canvas = tk.Canvas(
                    grid_frame,
                    width=96,
                    height=96,
                    bg="#1e1e1e",
                    highlightthickness=2,
                    highlightbackground=CANVAS_BORDER,
                )
                error_canvas.create_text(
                    48, 48,
                    text="Error\nloading",
                    fill=FG_COLOR,
                    font=("Arial", 10),
                )
                error_canvas.grid(row=row_idx, column=col_idx, padx=5, pady=5)

        # Close button
        close_btn = tk.Button(
            ex_window,
            text="Close",
            command=ex_window.destroy,
            width=12,
            bg=BUTTON_BG,
            fg=BUTTON_FG,
            activebackground=BUTTON_ACTIVE_BG,
            activeforeground=BUTTON_ACTIVE_FG,
            relief="flat",
            font=("Arial", 11),
        )
        close_btn.pack(pady=(15, 0))

    def _export_charts(self):
        """Exports charts as PNG files."""
        output_dir = filedialog.askdirectory(
            title="Select directory to export charts",
            initialdir=self.results_path.parent,
        )

        if not output_dir:
            return

        output_path = Path(output_dir)
        timestamp = self.results["timestamp"]

        try:
            # Export confusion matrix
            confusion_path = output_path / f"confusion_matrix_{timestamp}.png"
            self.confusion_fig.savefig(
                confusion_path,
                dpi=300,
                facecolor=BG_COLOR,
                edgecolor='none',
                bbox_inches='tight'
            )

            # Export metrics
            metrics_path = output_path / f"per_class_metrics_{timestamp}.png"
            self.metrics_fig.savefig(
                metrics_path,
                dpi=300,
                facecolor=BG_COLOR,
                edgecolor='none',
                bbox_inches='tight'
            )

            messagebox.showinfo(
                "Export Complete",
                f"Charts exported to:\n{output_path}\n\n"
                f"- {confusion_path.name}\n"
                f"- {metrics_path.name}"
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export charts:\n{e}")

    def _on_closing(self):
        """Handler for window closing - cleans up resources."""
        try:
            # Close all matplotlib figures
            plt.close('all')

            # Destroy the window
            self.window.quit()
            self.window.destroy()
        except Exception:
            pass

    def center_window(self):
        """Centers window on screen."""
        # Ensure window has correct size
        self.window.update_idletasks()

        # Get screen and window dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        # Calculate central position
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)

        # Set position
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def show(self):
        """Shows the window."""
        import sys
        self.window.mainloop()
        # Clean up matplotlib to ensure all threads are closed
        plt.close('all')
        # Flush output streams to prevent hanging
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    """Standalone visualization launch."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize results from PetTester")
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to JSON results file",
    )

    args = parser.parse_args()

    if args.results is None:
        # Find latest results file
        results_files = sorted(Path(".").glob("test_results_*.json"), reverse=True)
        if not results_files:
            print("Error: No results file found.")
            print("Run: python visualize.py --results <path_to_JSON>")
            return
        args.results = results_files[0]
        print(f"Loading latest results: {args.results}")

    viz = ResultsVisualizer(args.results)
    viz.show()


if __name__ == "__main__":
    main()

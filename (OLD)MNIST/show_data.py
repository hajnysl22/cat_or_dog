"""
MNIST Data Visualization

Tento skript stáhne MNIST data (pokud ještě nejsou) a zobrazí
náhodné vzorky v interaktivním okně.

Spuštění:
    python show_data.py

Zobrazí grid s 50 náhodnými obrázky (5 z každé číslice 0-9)
a základní statistiky o datasetu.
"""

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

# ============================================================================
# POMOCNÁ FUNKCE - CENTROVÁNÍ OKEN
# ============================================================================

def center_figure(fig):
    """Centruje matplotlib okno na obrazovku."""
    try:
        # Získat Tk window handle z matplotlib figure
        manager = fig.canvas.manager
        window = manager.window

        # Vynutit kompletní update včetně vykreslení
        fig.canvas.draw()
        window.update()

        # Získat rozměry obrazovky a okna
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window_width = window.winfo_width()
        window_height = window.winfo_height()

        # Vypočítat centrální pozici (střed okna uprostřed obrazovky)
        x = max(0, (screen_width - window_width) // 2)
        y = max(0, (screen_height - window_height) // 2)

        # Nastavit pozici
        window.geometry(f"+{x}+{y}")
    except Exception as e:
        # Pokud backend není TkAgg nebo jiný problém, ignorujeme
        print(f"⚠️  Nepodařilo se centrovat okno: {e}")

# ============================================================================
# KONFIGURACE
# ============================================================================

DATA_DIR = "./data/"
SAMPLES_PER_CLASS = 6  # Kolik vzorků z každé číslice zobrazit

# Vytvoření složky pro data
Path(DATA_DIR).mkdir(exist_ok=True)

print("=" * 60)
print("MNIST Data Viewer")
print("=" * 60)

# ============================================================================
# STAŽENÍ A NAČTENÍ DAT
# ============================================================================

print("\nNačítám MNIST dataset...")
print("(První spuštění stáhne ~10MB dat)")

# Základní transformace - jen převod na tensor
transform = transforms.ToTensor()

# Načtení trénovacích dat
train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

# Načtení testovacích dat
test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

print(f"✓ Trénovací data: {len(train_dataset)} vzorků")
print(f"✓ Testovací data: {len(test_dataset)} vzorků")

# ============================================================================
# STATISTIKY
# ============================================================================

print("\nPočítám statistiky...")

# Spočítáme počet vzorků pro každou číslici
train_counts = [0] * 10
test_counts = [0] * 10

for _, label in train_dataset:
    train_counts[label] += 1

for _, label in test_dataset:
    test_counts[label] += 1

print("\nRozdělení tříd (train):")
for digit in range(10):
    bar = "█" * (train_counts[digit] // 100)
    print(f"  {digit}: {train_counts[digit]:5d} vzorků {bar}")

print(f"\nCelkem: {sum(train_counts)} trénovacích + {sum(test_counts)} testovacích vzorků")

# ============================================================================
# VIZUALIZACE
# ============================================================================

print(f"\nVytvářím vizualizaci ({SAMPLES_PER_CLASS} vzorků z každé číslice)...")

# Vytvoření figure - 6 řádků (vzorky) × 10 sloupců (číslice)
fig, axes = plt.subplots(SAMPLES_PER_CLASS, 10, figsize=(15, 10))
fig.suptitle('MNIST Dataset - Náhodné vzorky z každé číslice', fontsize=18, y=0.98)

# Pro každou číslici najdeme náhodné vzorky
for digit in range(10):
    # Najdeme indexy všech vzorků dané číslice
    indices = [i for i, (_, label) in enumerate(train_dataset) if label == digit]

    # Vybereme náhodné vzorky
    np.random.seed(42)  # Pro reprodukovatelnost
    selected_indices = np.random.choice(indices, SAMPLES_PER_CLASS, replace=False)

    # Zobrazíme vybrané vzorky
    for row, idx in enumerate(selected_indices):
        img, label = train_dataset[idx]
        img = img.squeeze().numpy()  # 28x28 numpy array

        ax = axes[row, digit]
        ax.imshow(img, cmap='gray')
        ax.axis('off')

        # Popisek číslice nad prvním řádkem
        if row == 0:
            ax.text(14, -5, f'{digit}', fontsize=16, fontweight='bold',
                   ha='center', va='bottom')

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.02, wspace=0.02)

# ============================================================================
# ZOBRAZENÍ
# ============================================================================

print("\n" + "=" * 60)
print("✓ Vizualizace připravena!")
print("=" * 60)
print("\nZavřete okno pro ukončení programu.")
print("\nTip: Můžete upravit SAMPLES_PER_CLASS v kódu pro více/méně vzorků.")

# Centrovat okno
center_figure(fig)

plt.show()

print("\n✓ Hotovo!")

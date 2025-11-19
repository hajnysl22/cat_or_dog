"""
MNIST Model Visualization

Tento skript načte natrénovaný model a vizualizuje jeho architekturu,
naučené konvoluční filtry a feature maps.

Spuštění:
    python show_model.py

Předpoklad: Model již existuje (./model/mnist_model.pt)
Pokud model neexistuje, spusťte nejdřív: python mnist.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Import SimpleMNIST třídy z mnist.py
from mnist import SimpleMNIST

# ============================================================================
# POMOCNÁ FUNKCE - CENTROVÁNÍ A Z-ORDER OKEN
# ============================================================================

def center_and_order_figures(figures):
    """
    Centruje matplotlib okna na obrazovku a nastaví správný Z-order.
    První figure v listu bude nahoře (viditelný jako první).
    """
    try:
        for fig in figures:
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

        # Nastavit Z-order: poslední na zásobník = první viditelný
        # Proto procházíme v opačném pořadí
        for fig in reversed(figures):
            manager = fig.canvas.manager
            window = manager.window
            window.lift()
            window.focus_force()
    except Exception as e:
        # Pokud backend není TkAgg nebo jiný problém, ignorujeme
        print(f"⚠️  Nepodařilo se centrovat okna: {e}")

# ============================================================================
# KONTROLA EXISTENCE MODELU
# ============================================================================

model_path = Path("./model/mnist_model.pt")

if not model_path.exists():
    print("=" * 60)
    print("❌ Model nebyl nalezen!")
    print("=" * 60)
    print("\nModel se očekává v: ./model/mnist_model.pt")
    print("\nNejdřív spusťte trénink:")
    print("  python mnist.py")
    print("\nPo dokončení tréninku spusťte tento skript znovu.")
    print("=" * 60)
    sys.exit(0)

# ============================================================================
# NAČTENÍ MODELU
# ============================================================================

print("=" * 60)
print("MNIST Model Visualization")
print("=" * 60)
print(f"\nNačítám model z: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMNIST()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print(f"✓ Model načten (zařízení: {device})")

# ============================================================================
# TEXT SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("Model Architecture Summary")
print("=" * 60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: SimpleMNIST (28×28 input)")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {total_params - trainable_params:,}")

print("\nArchitecture:")
print("  Conv2d(1→32, 3×3) + MaxPool → 14×14")
print("  Conv2d(32→64, 3×3) + MaxPool → 7×7")
print("  Flatten + Linear(3136→128) + Dropout(0.5)")
print("  Linear(128→10)")

# ============================================================================
# VIZUALIZACE CONV FILTRŮ - VRSTVA 1
# ============================================================================

print("\n" + "=" * 60)
print("Vizualizuji konvoluční filtry...")
print("=" * 60)

# Layer 1: 32 filtrů 3×3
conv1_weights = model.conv1.weight.data.cpu().numpy()  # [32, 1, 3, 3]
n_filters1 = conv1_weights.shape[0]

fig1, axes1 = plt.subplots(4, 8, figsize=(12, 6))
fig1.suptitle('Conv Layer 1 - Learned Filters (32 filters, 3×3)', fontsize=14)

for i in range(n_filters1):
    ax = axes1[i // 8, i % 8]
    filter_img = conv1_weights[i, 0, :, :]  # [3, 3]
    ax.imshow(filter_img, cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i}', fontsize=8)
    ax.axis('off')

plt.tight_layout()

# ============================================================================
# VIZUALIZACE CONV FILTRŮ - VRSTVA 2
# ============================================================================

# Layer 2: 64 filtrů 3×3 (pro každý z 32 input channels)
# Zobrazíme průměr přes input channels
conv2_weights = model.conv2.weight.data.cpu().numpy()  # [64, 32, 3, 3]
n_filters2 = conv2_weights.shape[0]

# Průměr přes input channels
conv2_weights_avg = conv2_weights.mean(axis=1)  # [64, 3, 3]

fig2, axes2 = plt.subplots(8, 8, figsize=(12, 12))
fig2.suptitle('Conv Layer 2 - Learned Filters (64 filters, 3×3, averaged over input channels)',
              fontsize=14)

for i in range(n_filters2):
    ax = axes2[i // 8, i % 8]
    filter_img = conv2_weights_avg[i, :, :]  # [3, 3]
    ax.imshow(filter_img, cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i}', fontsize=8)
    ax.axis('off')

plt.tight_layout()

# ============================================================================
# FEATURE MAPS - UKÁZKOVÁ ČÍSLICE
# ============================================================================

print("\nVytvářím feature maps pro ukázkovou číslici...")

# Načteme ukázkovou číslici z MNIST
# DŮLEŽITÉ: Musí mít stejnou normalizaci jako při tréninku!
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean a std
])
test_dataset = datasets.MNIST(root='./data/', train=False, download=False, transform=transform)

# Vybereme náhodnou číslici
np.random.seed(42)
idx = np.random.randint(len(test_dataset))
sample_img, sample_label = test_dataset[idx]
sample_img = sample_img.unsqueeze(0).to(device)  # [1, 1, 28, 28]

print(f"Ukázková číslice: {sample_label}")

# Forward pass s uložením intermediate outputs
with torch.no_grad():
    # Po conv1 + pool
    x = model.relu(model.conv1(sample_img))
    feat1 = x.clone()
    x = model.pool(x)

    # Po conv2 + pool
    x = model.relu(model.conv2(x))
    feat2 = x.clone()

# Vizualizace
fig3, axes3 = plt.subplots(3, 10, figsize=(15, 5))
fig3.suptitle(f'Feature Maps - Sample digit: {sample_label}', fontsize=14)

# Originální obrázek v prvním řádku (zleva)
axes3[0, 0].imshow(sample_img.cpu().squeeze(), cmap='gray')
axes3[0, 0].set_title('Input', fontsize=10)
axes3[0, 0].axis('off')
for col in range(1, 10):
    axes3[0, col].axis('off')

# Feature maps z conv1 (vybereme prvních 10)
feat1_np = feat1.cpu().squeeze().numpy()  # [32, 28, 28]

print("\nDiagnostika Conv1 Feature Maps (prvních 10):")
for col in range(10):
    feat_map = feat1_np[col]
    fmin, fmax = feat_map.min(), feat_map.max()
    fmean, fstd = feat_map.mean(), feat_map.std()
    n_unique = len(np.unique(feat_map))
    n_zeros = np.sum(feat_map == 0)
    n_total = feat_map.size

    print(f"  C1-{col}: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}, std={fstd:.4f}, "
          f"unique={n_unique}, zeros={n_zeros}/{n_total} ({100*n_zeros/n_total:.1f}%)")

    # Robustní vizualizace
    if fstd < 1e-6:  # Konstantní nebo skoro konstantní
        axes3[1, col].text(0.5, 0.5, 'Dead\nfilter', ha='center', va='center',
                          transform=axes3[1, col].transAxes, fontsize=8)
        axes3[1, col].set_facecolor('#1a1a1a')
    else:
        # Percentilová normalizace (eliminuje outliers)
        vmin_robust = np.percentile(feat_map, 2)
        vmax_robust = np.percentile(feat_map, 98)

        # Ochrana proti min==max
        if vmax_robust - vmin_robust < 1e-6:
            vmax_robust = vmin_robust + 1e-6

        axes3[1, col].imshow(feat_map, cmap='RdBu_r', vmin=vmin_robust, vmax=vmax_robust)

    axes3[1, col].set_title(f'C1-{col}', fontsize=8)
    axes3[1, col].axis('off')

# Feature maps z conv2 (vybereme prvních 10)
feat2_np = feat2.cpu().squeeze().numpy()  # [64, 14, 14]

print("\nDiagnostika Conv2 Feature Maps (prvních 10):")
for col in range(10):
    feat_map = feat2_np[col]
    fmin, fmax = feat_map.min(), feat_map.max()
    fmean, fstd = feat_map.mean(), feat_map.std()
    n_unique = len(np.unique(feat_map))
    n_zeros = np.sum(feat_map == 0)
    n_total = feat_map.size

    print(f"  C2-{col}: min={fmin:.4f}, max={fmax:.4f}, mean={fmean:.4f}, std={fstd:.4f}, "
          f"unique={n_unique}, zeros={n_zeros}/{n_total} ({100*n_zeros/n_total:.1f}%)")

    # Robustní vizualizace
    if fstd < 1e-6:  # Konstantní nebo skoro konstantní
        axes3[2, col].text(0.5, 0.5, 'Dead\nfilter', ha='center', va='center',
                          transform=axes3[2, col].transAxes, fontsize=8)
        axes3[2, col].set_facecolor('#1a1a1a')
    else:
        # Percentilová normalizace (eliminuje outliers)
        vmin_robust = np.percentile(feat_map, 2)
        vmax_robust = np.percentile(feat_map, 98)

        # Ochrana proti min==max
        if vmax_robust - vmin_robust < 1e-6:
            vmax_robust = vmin_robust + 1e-6

        axes3[2, col].imshow(feat_map, cmap='RdBu_r', vmin=vmin_robust, vmax=vmax_robust)

    axes3[2, col].set_title(f'C2-{col}', fontsize=8)
    axes3[2, col].axis('off')

plt.tight_layout()

# ============================================================================
# ZOBRAZENÍ
# ============================================================================

print("\n" + "=" * 60)
print("✓ Vizualizace připravena!")
print("=" * 60)
print("\nZobrazuji 3 okna:")
print("  1. Conv Layer 1 Filters (32 filters)")
print("  2. Conv Layer 2 Filters (64 filters)")
print("  3. Feature Maps (ukázková číslice)")
print("\nZavřete okna pro ukončení programu.")

# Shromáždit všechny figures pro centrování a správný Z-order
figures = [fig1, fig2, fig3]

# Centrovat a nastavit správné pořadí
center_and_order_figures(figures)

plt.show()

print("\n✓ Hotovo!")

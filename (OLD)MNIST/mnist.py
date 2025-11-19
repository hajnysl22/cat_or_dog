"""
(OLD) MNIST - Klasický přístup

Tradiční "ML Hello World" s hotovým MNIST datasetem.
Tento skript demonstruje klasický workflow: stažení hotových dat,
trénink modelu a testování - vše v jednom souboru.

Klasický MNIST workflow:
1. Stažení hotového datasetu (60k train, 10k test)
2. Vytvoření jednoduchého modelu
3. Trénink
4. Testování

Spuštění:
    python mnist.py

První spuštění automaticky stáhne ~10MB MNIST data do ./data/
Model se automaticky uloží do ./model/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# ============================================================================
# MODEL - Jednoduchá CNN
# ============================================================================

class SimpleMNIST(nn.Module):
    """Velmi jednoduchá konvoluční síť pro MNIST (28×28 obrázky)."""

    def __init__(self):
        super(SimpleMNIST, self).__init__()
        # Konvoluční vrstvy
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                           # 28x28 -> 14x14 -> 7x7

        # Plně propojené vrstvy
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Konvoluční část
        x = self.relu(self.conv1(x))
        x = self.pool(x)              # 28x28 -> 14x14
        x = self.relu(self.conv2(x))
        x = self.pool(x)              # 14x14 -> 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Plně propojená část
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ============================================================================
# HLAVNÍ PROGRAM - spustí se pouze při přímém spuštění skriptu
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # PŘÍPRAVA SLOŽEK
    # ========================================================================

    # Vytvoření složek pro data a model (pokud neexistují)
    Path("./data").mkdir(exist_ok=True)
    Path("./model").mkdir(exist_ok=True)

    # ========================================================================
    # KONFIGURACE
    # ========================================================================

    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("(OLD) MNIST - Klasický přístup s hotovými daty")
    print("=" * 60)
    print(f"Používám zařízení: {DEVICE}")


    # ========================================================================
    # DATA - Stažení a načtení MNIST
    # ========================================================================

    print("\nStahuji MNIST dataset...")
    print("(První spuštění stáhne ~10MB dat)")

    # Transformace: převod na tensor a normalizace
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean a std
    ])

    # Stažení trénovací sady
    train_dataset = datasets.MNIST(
        root='./data/',
        train=True,
        download=True,
        transform=transform
    )

    # Stažení testovací sady
    test_dataset = datasets.MNIST(
        root='./data/',
        train=False,
        download=True,
        transform=transform
    )

    # DataLoadery pro dávkové zpracování
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Trénovací vzorky: {len(train_dataset)} (60,000 hotových vzorků)")
    print(f"Testovací vzorky: {len(test_dataset)} (10,000 hotových vzorků)")


    # ========================================================================
    # INICIALIZACE
    # ========================================================================

    model = SimpleMNIST().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nModel vytvořen: {sum(p.numel() for p in model.parameters())} parametrů")


    # ========================================================================
    # TRÉNINK
    # ========================================================================

    print(f"\nZačínám trénink na {EPOCHS} epoch...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Přesun dat na GPU/CPU
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistiky
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Progress každých 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        # Průměrné metriky za epochu
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] dokončena - "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


    # ========================================================================
    # TESTOVÁNÍ
    # ========================================================================

    print("\nTestuji model na testovací sadě...")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total
    print(f"\n{'='*60}")
    print(f"VÝSLEDEK: Test Accuracy: {test_accuracy:.2f}%")
    print(f"{'='*60}")

    # Typická očekávaná accuracy pro tento jednoduchý model: 98-99%
    if test_accuracy > 99:
        print("🏆 Výborný výsledek!")
    elif test_accuracy > 97:
        print("✅ Solidní výsledek!")
    else:
        print("⚠️  Model by mohl být lepší, zkuste více epoch nebo jinou architekturu.")


    # ========================================================================
    # ULOŽENÍ MODELU (automatické)
    # ========================================================================

    print("\nUkládám model...")
    model_path = './model/mnist_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model uložen do {model_path}")
    print("\nPro načtení použijte:")
    print("  model = SimpleMNIST()")
    print(f"  model.load_state_dict(torch.load('{model_path}'))")
    print("  model.eval()")

    print("\n" + "="*60)
    print("✅ Hotovo!")
    print("")
    print("   Tento 'OLD' přístup je rychlý a jednoduchý,")
    print("   ale naučí vás jen používat hotové nástroje.")
    print("")
    print("   Pro skutečné pochopení ML procesu:")
    print("   vytvořte vlastní dataset a celou pipeline od nuly!")
    print("="*60)

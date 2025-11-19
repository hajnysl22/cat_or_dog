# (OLD) MNIST - Klasický přístup

Tradiční implementace "ML Hello World" s hotovým MNIST datasetem. Tento skript demonstruje klasický způsob práce s ML - stažení hotových dat a rychlé natrénování modelu.

## 📜 Historie MNIST

**MNIST** (Modified National Institute of Standards and Technology database) vytvořil **Yann LeCun** a kolektiv v roce **1998**.

### Původní účel

- Benchmark pro testování algoritmů rozpoznávání ručně psaných číslic
- První úspěšné nasazení CNN architektury (**LeNet-5**)
- Standardizovaný dataset pro porovnání ML modelů

### Co MNIST obsahuje

- **60,000** trénovacích obrázků
- **10,000** testovacích obrázků
- Velikost: **28×28 pixelů**
- Formát: **Černé písmo na bílém pozadí**
- Zdroj: Americké poštovní formuláře a studentské písemky

### Proč "OLD"?

V moderní ML komunitě je MNIST považován za **přetrénovaný benchmark**:

- ✅ Skvělý pro výuku základů
- ❌ Příliš jednoduchý - modely dosahují 99.8% accuracy
- ❌ Nerealistický - reálný svět je složitější
- 🔄 Modernější alternativy: Fashion-MNIST, EMNIST, CIFAR-10

## 🆚 Srovnání: OLD vs. DIE-MNIST

| Aspekt | (OLD) MNIST | DIE-MNIST |
|--------|-------------|-------------|
| **Data** | Hotová ke stažení | Vlastní sběr/generování |
| **Velikost datasetu** | 60k train / 10k test | Dle potřeby |
| **Pipeline** | `datasets.MNIST()` | Celý workflow od nuly |
| **Kontrola** | Žádná (black box) | Plná kontrola nad procesem |
| **Učení** | Použití API | End-to-end ML pipeline |
| **Cíl** | Rychlý prototyp | Porozumění celému procesu |

## 🚀 Spuštění

```bash
cd (OLD)MNIST
python mnist.py
```

**První spuštění:**

- Automaticky stáhne ~10MB MNIST data do `./data/`
- Natrénuje model (~5 minut CPU / ~1 minuta GPU)
- Uloží model do `./model/mnist_model.pt`

**Další spuštění:**

- Data už jsou stažená, trénink začne okamžitě

## 🔍 Vizualizace dat a modelu

### Prohlížení datasetu

```bash
python show_data.py
```

Zobrazí interaktivní okno s náhodnými vzorky z MNIST datasetu (6 vzorků z každé číslice) včetně statistik o rozdělení tříd.

### Vizualizace natrénovaného modelu

```bash
python show_model.py
```

Po natrénování modelu (spuštěním `mnist.py`) můžete vizualizovat:

- **Architekturu** - textový summary s počtem parametrů
- **Konvoluční filtry** - naučené váhy 1. a 2. vrstvy jako obrázky
- **Feature maps** - co model "vidí" při zpracování ukázkové číslice

Skript automaticky zkontroluje existenci modelu a v případě potřeby navede ke spuštění tréninku.

### ⚠️ Proč jsou některé filtry "mrtvé"?

Při vizualizaci feature maps uvidíte, že část filtrů je označena jako "Dead filter". To je **záměrná demonstrace problému starých architektur**.

**Dying ReLU problém:**

- Tento model **nepoužívá BatchNorm** (jako staré CNN z roku 1998)
- ReLU aktivace může "zabít" neurony, které dostávají vždy negativní vstupy
- Výsledek: **40-80% filtrů je neaktivních** (produkují jen nuly)

**Proč model přesto funguje?**

- MNIST je **velmi jednoduchý** dataset
- Zbylých 20-60% filtrů stačí na 96-98% accuracy
- Na složitějších datech by model selhal

**Moderní řešení:**

- DIE-MNIST v tomto projektu používá **BatchNorm2d** za každou Conv vrstvou
- BatchNorm normalizuje aktivace → snižuje dying ReLU na <20%
- Výsledek: **efektivnější model** s lepší accuracy

To je důvod, proč je označení "(OLD)" přesné - ukazuje historický problém a jeho moderní řešení.

## 📊 Očekávané výsledky

Po 5 epochách by model měl dosáhnout:

- **Test Accuracy: 98-99%**
- **Training time: ~5 minut** (CPU) / **~1 minuta** (GPU)
- **Model size: ~600k parametrů**

## 🎓 Co se naučíte

Spuštěním tohoto skriptu:

1. ✅ Vidíte tradiční ML workflow (stáhnout → trénovat → testovat)
2. ✅ Pochopíte, co MNIST je a proč byl důležitý
3. ✅ Získáte baseline benchmark (~98-99% accuracy)
4. ✅ Pochopíte rozdíl mezi "použít API" vs "postavit od nuly"

## 💡 Použití jako benchmark

Po natrénování vlastního modelu můžete porovnat výsledky:

```python
# (OLD) MNIST baseline: 98-99% accuracy (60k vzorků, 28×28)
# Váš custom model: X% accuracy (Y vzorků, různé rozlišení)
```

**Otázky k zamyšlení:**

- Dosáhl váš model podobné accuracy s méně daty?
- Jak ovlivňuje kvalitu a množství vlastních dat?
- Je vaše vlastní pipeline efektivnější než hotové řešení?

## 📚 Reference

- **Original paper:** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.*
- **Dataset:** <http://yann.lecun.com/exdb/mnist/>
- **LeNet-5 architecture:** První úspěšná CNN pro rozpoznávání číslic

## 🎯 Závěr

Tento "starý" přístup je:

- ✅ Rychlý - funguje za pár minut
- ✅ Jednoduchý - minimální kód
- ✅ Spolehlivý - otestované řešení

Ale neposkytuje:

- ❌ Porozumění celé pipeline
- ❌ Kontrolu nad daty a procesem
- ❌ Zkušenost s real-world ML problémy

**DIE-MNIST přístup učí:**

- ✅ Jak sbírat a připravovat vlastní data
- ✅ Jak navrhovat celý ML workflow
- ✅ Jak řešit problémy v každé fázi
- ✅ Reálný proces, ne jen `import dataset`

---

**"The old way works. The DIE way teaches."** 🎓

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>

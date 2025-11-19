# DigitTester

CLI nástroj pro kvantifikované testování natrénovaných modelů z DigitLearner. Vypočítá detailní metriky včetně confusion matrix, per-class accuracy, precision, recall a F1-score.

**Nové v této verzi:**

- 📊 **Grafická vizualizace výsledků** - Automaticky zobrazí interaktivní GUI s grafy
- 🎯 **Interaktivní výběr modelu** - Vyberte z seznamu dostupných modelů
- 📁 **Volba testovacích dat** - Testujte na celém datasetu nebo jen test split
- 🖱️ **Klikací confusion matrix** - Klikněte na buňku a uvidíte příklady chyb
- 💾 **Export grafů** - Uložte vizualizace jako PNG

## Instalace

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nejrychlejší je spustit `run.bat`, který všechno připraví automaticky.

## Spuštění

### Interaktivní režim (doporučeno)

Nejjednodušší použití s interaktivním výběrem:

```bash
run.bat
```

Skript se vás zeptá:

1. **Který model testovat?** - Zobrazí seznam všech modelů s metrikami
2. **Jaká data použít?** - Kompletní dataset (doporučeno) nebo jen test split

Po testování se automaticky otevře vizualizační okno s grafy.

### S explicitními parametry

```bash
run.bat --model_dir ../shared/models/run_YYYYMMDD_HHMMSS --data_dir ../shared/data/composed/test
```

Nebo přímé spuštění Pythonu:

```bash
python main.py --model_dir ../shared/models/run_20251024_100000 --data_dir ../shared/data/composed
```

## Příkazové řádkové parametry

### Základní parametry

- `--model_dir PATH` - Cesta ke složce s natrénovaným modelem (musí obsahovat `config.json` a `digit_cnn.pt`). Pokud není zadána, automaticky se použije nejnovější model z `../shared/models/`
- `--data_dir PATH` - Cesta k testovacím datům (struktura `0/`, `1/`, ..., `9/` s BMP soubory). Default: interaktivní výběr

### Volitelné parametry

- `--batch_size N` - Velikost batche pro evaluaci (default: 64)
- `--use_cpu` - Vynutit použití CPU i když je GPU dostupné
- `--output PATH` - Cesta k výstupnímu JSON souboru (default: `../shared/tests/test_results_TIMESTAMP.json`)

## Struktura dat

Testovací data očekávají stejnou strukturu jako DigitLearner:

```
test_data/
├── 0/
│   ├── 0001.bmp
│   ├── 0002.bmp
│   └── ...
├── 1/
│   ├── 0001.bmp
│   └── ...
...
└── 9/
    ├── 0001.bmp
    └── ...
```

## Interaktivní vizualizace

Po dokončení testování se automaticky otevře GUI okno s vizualizací výsledků.

### Co vizualizace zobrazuje

1. **Overall Score Panel**
   - Celková accuracy s barevným indikátorem (zelená >90%, žlutá 70-90%, červená <70%)
   - Average loss
   - Celkový počet testovacích vzorků

2. **Confusion Matrix Heatmap** (INTERAKTIVNÍ!)
   - Barevná mapa záměn (zelená = správně, červená = chyby)
   - **Klikněte na buňku** → Zobrazí se okno s 4-6 náhodnými příklady té konkrétní chyby
   - Vidíte skutečné obrázky, které model plete

3. **Per-Class Bar Charts**
   - Seskupené sloupcové grafy pro Accuracy, Precision, Recall, F1-Score
   - Snadné porovnání výkonu pro každou číslici

4. **Top 5 Confusions**
   - Seznam nejčastějších chyb modelu
   - Například: "3 → 8: 12× (model často plete trojku za osmičku)"

5. **Export Button**
   - Tlačítko "💾 Exportovat grafy"
   - Uloží confusion matrix a per-class charts jako PNG (300 DPI)

### Použití vizualizace

- Zavřete okno → Skript automaticky skončí
- Klikejte na confusion matrix → Prozkoumejte konkrétní chyby
- Export grafů → Uložte pro prezentaci/reporty

## Výstup

### Konzolový výstup

Aplikace vypíše do konzole:

1. **Overall Metrics**
   - Overall Accuracy
   - Average Loss
   - Total Samples
   - Model a data paths
   - Použité zařízení (CPU/GPU)

2. **Per-Class Metrics**
   - Accuracy per class
   - Precision per class
   - Recall per class
   - F1-Score per class
   - Number of samples per class

3. **Confusion Matrix**
   - Řádky = skutečné třídy
   - Sloupce = predikované třídy

### JSON výstup

Všechny metriky se automaticky ukládají do centrální složky `../shared/tests/` jako JSON soubor s časovým razítkem ve formátu `test_results_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "20251024_120000",
  "model_dir": "/path/to/model",
  "data_dir": "/path/to/test/data",
  "device": "cuda",
  "batch_size": 64,
  "overall_accuracy": 0.9523,
  "average_loss": 0.1847,
  "total_samples": 1000,
  "num_classes": 10,
  "per_class_metrics": {
    "0": {
      "precision": 0.96,
      "recall": 0.94,
      "f1_score": 0.95,
      "accuracy": 0.94,
      "samples": 100
    },
    ...
  },
  "confusion_matrix": [[94, 1, 0, ...], ...],
  "model_config": {...}
}
```

## Příklad použití

### 1. Nejjednodušší použití (interaktivní)

Použijte workflow skript v rootu projektu:

```bash
cd ..
start_collect.bat  # nebo start_dream.bat
```

Workflow automaticky projde celým procesem a na konci otevře vizualizaci.

Nebo testujte samostatně:

```bash
cd DigitTester
run.bat
```

**Interaktivní výběr:**

1. Vyberte model ze seznamu (Enter = nejnovější)
2. Vyberte testovací data (Enter = kompletní dataset)
3. Počkejte na dokončení testování
4. Automaticky se otevře vizualizace s grafy

### 2. Testování s vlastními parametry

```bash
run.bat --model_dir ../shared/models/run_20251024_100000 --batch_size 128 --use_cpu --output my_results.json
```

### 3. Srovnání více modelů

```bash
python main.py --model_dir ../shared/models/run_A --output results_model_A.json
python main.py --model_dir ../shared/models/run_B --output results_model_B.json
python main.py --model_dir ../shared/models/run_C --output results_model_C.json
```

Potom porovnejte výsledky v JSON souborech.

## Metriky

### Overall Accuracy

Celková přesnost modelu na všech testovacích vzorcích:

```
accuracy = (počet správně klasifikovaných) / (celkový počet vzorků)
```

### Per-Class Metrics

**Precision** (přesnost):

```
precision = TP / (TP + FP)
```

Jak často je predikce dané třídy správná?

**Recall** (úplnost):

```
recall = TP / (TP + FN)
```

Kolik vzorků dané třídy model dokázal najít?

**F1-Score** (harmonický průměr):

```
f1_score = 2 * (precision * recall) / (precision + recall)
```

Vyvážená metrika kombinující precision a recall.

**Per-Class Accuracy**:

```
accuracy = TP / (všechny vzorky dané třídy)
```

Přesnost pro konkrétní třídu.

### Confusion Matrix

Matice záměn ukazuje, jak často model pletl jednotlivé třídy:

- Řádek `i`, sloupec `j` = kolikrát model predikoval třídu `j`, když správně byla třída `i`
- Diagonála = správně klasifikované vzorky
- Mimo diagonálu = záměny

## Kompatibilita

- Plně kompatibilní s modely z **DigitLearner**
- Podporuje data z **DigitCollector**, **DigitDreamer** a **DigitComposer**
- Automatická detekce GPU/CPU
- Podporuje SimpleCNN architekturu (jedinou v DigitLearner)

## Známá omezení

- **Pouze SimpleCNN**: Podporuje pouze SimpleCNN architekturu z DigitLearner
- **32×32 obrázky**: Data musí být 32×32 px BMP soubory
- **10 tříd**: Fixně nastaveno pro číslice 0-9
- **Bez augmentace**: Testování probíhá bez augmentace dat

## Tipy

1. **Použijte interaktivní výběr dat**:
   - Testujte na **kompletním datasetu** (doporučeno) pro realistický pohled
   - Testujte na **test split** pro reprodukci výsledků z tréninku
   - Testujte na **train+val** pro detekci přeučení

2. **Prozkoumejte chyby ve vizualizaci**:
   - Klikejte na confusion matrix → Uvidíte konkrétní příklady chyb
   - Identifikujte systematické chyby (např. 3 pletena za 8)
   - Rozhodněte, zda potřebujete více trénovacích dat pro konkrétní páry

3. **Porovnávejte modely**:
   - Testujte více modelů s různými hyperparametry
   - Porovnejte JSON výsledky nebo vizualizace vedle sebe
   - Sledujte per-class metriky - některé modely jsou lepší na konkrétní číslice

4. **Používejte workflow skripty**:
   - `start_collect.bat` nebo `start_dream.bat` v rootu projektu
   - Automaticky projdou celým procesem od sběru dat po vizualizaci

5. **Exportujte grafy**:
   - Tlačítko v vizualizaci → Uložte jako PNG pro reporty/prezentace

## Centrální struktura

Tento nástroj je součástí ekosystému DIE-MNIST (Digital Identification Exercise - MNIST), který používá centrální adresářovou strukturu:

- **Testovací data**: `../shared/data/composed/test/` (vytvořené pomocí DigitComposer)
- **Modely**: `../shared/models/run_YYYYMMDD_HHMMSS/` (vytvořené pomocí DigitLearner)
- **Automatická detekce**: Při spuštění bez parametrů se automaticky najde nejnovější model

Viz hlavní README pro kompletní workflow.

## Rozdíl mezi DigitTester a DigitTeaser

- **DigitTester** (tento nástroj) - Kvantifikované testování, detailní metriky, CLI
- **DigitTeaser** (dříve DigitTester) - Interaktivní GUI aplikace pro "poškádlení" modelu kreslením číslic

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>

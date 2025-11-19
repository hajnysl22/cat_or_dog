# DIE (**D**igital **I**dentification **E**xercise) MNIST

Tento projekt obsahuje šest na sebe navazujících aplikací pro sběr, slučování, syntetické generování, učení modelu a testování nad vlastní variantou datasetu ručně psaných číslic.

## Aplikace

### DigitCollector

Desktopová aplikace pro kreslení a ukládání číslic. Automaticky vyvažuje zastoupení tříd - méně zastoupené číslice mají vyšší pravděpodobnost výběru. Data se ukládají do `shared/data/collected/`.

### DigitComposer

Slučuje datasety z více instancí DigitCollector do jednoho s automatickým rozdělením na train/val/test sady. Podporuje import z adresářů i ZIP archivů. Inteligentní přečíslování souborů zachovává trasovatelnost původu každého vzorku. Nastavitelné poměry rozdělení a seed pro reprodukovatelnost.

### DigitDreamer

Generuje syntetické číslice z geometrických tvarů. Výstupní formát kompatibilní s DigitCollector. Umožňuje generovat dataset do podsložek `train/`, `val/`, `test` nebo navázat na existující obsah.

### DigitLearner

Trénuje konvoluční síť SimpleCNN na datech z předchozích nástrojů. Ukládá model i průběh učení. Obsahuje GUI editor konfigurace (`marshall`).

### DigitTeaser

Interaktivní GUI aplikace pro real-time vizualizaci klasifikace. Umožňuje kreslit číslice a okamžitě vidět pravděpodobnosti klasifikace zobrazené jako pruhové grafy s vizuálním zvýrazněním nejvyšší hodnoty. Ideální pro "poškádlení" modelu a intuitivní pochopení jeho chování.

### DigitTester

Nástroj pro kvantifikované testování natrénovaných modelů s interaktivní grafickou vizualizací. Nabízí výběr modelu a testovacích dat, vypočítá detailní metriky (overall accuracy, per-class precision/recall/F1-score, confusion matrix) a automaticky zobrazí výsledky v GUI. Klikací confusion matrix umožňuje prozkoumat konkrétní příklady chyb modelu. Výsledky lze exportovat jako grafy (PNG) nebo JSON. Vhodný pro seriózní evaluaci a porovnávání modelů.

## Struktura projektu

Projekt používá centrální adresářovou strukturu pro data a modely, což zajišťuje přehlednost a snadnou správu:

```
DIE-MNIST/
├── shared/                   # Centrální úložiště pro sdílená data a modely
│   ├── data/                 # Datové úložiště
│   │   ├── collected/        # ← DigitCollector ukládá sem
│   │   │   ├── 0/
│   │   │   ├── 1/
│   │   │   └── ... (0-9)
│   │   ├── synthetic/        # ← DigitDreamer ukládá sem
│   │   │   ├── train/0-9/
│   │   │   ├── val/0-9/
│   │   │   └── test/0-9/
│   │   └── composed/         # ← DigitComposer ukládá sem
│   │       ├── train/0-9/    # → DigitLearner čte odtud
│   │       ├── val/0-9/
│   │       └── test/0-9/     # → DigitTester čte odtud
│   ├── models/               # Úložiště natrénovaných modelů
│   │   ├── run_20251024_120000/  # ← DigitLearner ukládá sem
│   │   │   ├── config.json
│   │   │   ├── digit_cnn.pt  # → DigitTeaser/DigitTester čtou odtud
│   │   │   ├── training_history.json
│   │   │   └── test_metrics.json
│   │   └── ...
│   └── tests/                # Výsledky testování
│       ├── test_results_20251024_120000.json  # ← DigitTester ukládá sem
│       └── ...
└── [nástroje s izolovanými .venv]
```

### Workflow

1. **DigitCollector** → vytváří ručně kreslená data v `shared/data/collected/`
2. **DigitDreamer** → generuje syntetická data v `shared/data/synthetic/`
3. **DigitComposer** → slučuje collected + synthetic → `shared/data/composed/`
4. **DigitLearner** → trénuje na `composed/` → ukládá model do `shared/models/`
5. **DigitTeaser** → načítá model z `shared/models/` pro interaktivní testování
6. **DigitTester** → načítá model z `shared/models/` a data z `composed/test/` pro metriky

**Automatizace:** Kroky 1-4 nebo 2-4 lze automaticky provést jedním příkazem pomocí `start_collect.bat` nebo `start_dream.bat`.

### Výhody

- ✅ **Jasná struktura** - vše na jednom místě
- ✅ **Oddělení nástrojů od dat** - snadné zálohování a cleanup
- ✅ **Automatické cesty** - nástroje fungují "out of the box"
- ✅ **CLI override** - pokročilí uživatelé mohou přepsat defaulty

## Rychlé spuštění

### Automatizované workflow (doporučeno)

Pro kompletní průchod celou pipeline od začátku do konce jsou k dispozici dva workflow skripty:

#### start_collect.bat

Workflow s **ručně kreslenými** daty:

```bash
start_collect.bat
```

Automaticky provede: **DigitCollector** → **DigitComposer** → **DigitLearner** → **DigitTester**

#### start_dream.bat

Workflow se **syntetickými** daty:

```bash
start_dream.bat
```

Automaticky provede: **DigitDreamer** → **DigitComposer** → **DigitLearner** → **DigitTester**

Workflow skripty automaticky řetězí jednotlivé nástroje, přeskakují pauzy mezi kroky a na konci zobrazí vizualizaci výsledků testování.

### Ruční spuštění jednotlivých nástrojů

Pro nezávislé spuštění konkrétního nástroje:

- `DigitCollector\run.bat` – sběr ručně psaných číslic
- `DigitComposer\run.bat` – sloučení datasetů
- `DigitDreamer\run.bat [volby]` – generování syntetických dat (např. `run.bat --samples 500 --split`)
- `DigitLearner\run.bat [train|marshall] [volby]` – trénink modelu nebo editor konfigurace
- `DigitTeaser\run.bat` – interaktivní testování modelu kreslením
- `DigitTester\run.bat [volby]` – kvantifikované testování s vizualizací

Každý skript automaticky vytvoří a znovu použije izolované virtuální prostředí (`.venv`). Je potřeba mít nainstalovaný Python 3.10+ dostupný v proměnné `PATH`.

---

<sub>Dokumentace vygenerována AI asistentem Claude Code (Anthropic) – říjen 2025</sub>

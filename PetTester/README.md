# PetTester

Evaluates the trained Cat vs Dog classifier.

## Usage

### Interactive Mode (Recommended)
Run the tester without arguments to interactively select a trained model from `../shared/models`:

```bash
python main.py
```

It will:
1. List available models (latest first).
2. Ask you to select one.
3. Run evaluation on test data (`../shared/data/composed/test`).
4. Display metrics in the console.
5. Save results to a JSON file in `../shared/tests/`.
6. Ask if you want to visualize the results.

### Non-Interactive Mode
You can specify the model directory explicitly:

```bash
python main.py --model_dir ../shared/models/run_YYYYMMDD_HHMMSS
```

## Features

- **Model Selection:** Automatically finds and lists trained models.
- **Evaluation:** Calculates accuracy, loss, and confusion matrix.
- **Results Storage:** Saves detailed results (including per-image predictions) to JSON.
- **Visualization:** Includes a GUI tool (`visualize.py`) to explore results:
    - Overall metrics dashboard.
    - Interactive confusion matrix.
    - Per-class precision/recall/F1-score charts.
    - Visual examples of misclassified images.
    - Export charts to PNG.

### Standalone Visualization
To view previous results without running evaluation again:

```bash
python visualize.py --results ../shared/tests/results_YYYYMMDD_HHMMSS.json
```

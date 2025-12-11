# PetCollector

GUI application for collecting cat and dog images.

## Features
- **Manual Import:** Select images from your local computer and add them to the dataset.
- **Auto Download:** Automatically download random images from public APIs (TheCatAPI, DogCEO).
- **Live Stats:** See how many images you have collected in real-time.

## Usage

1.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the application:
    ```bash
    python main.py
    ```
    *(Or simply click `run.bat`)*

3.  Use the GUI to:
    - Click "Import Cat Images" to browse and add files.
    - Click "Start Download" to fetch from the web.
    - Images are saved to `../shared/data/raw/cat` and `../shared/data/raw/dog`.

import os
import requests
import time
from pathlib import Path

# Configuration
CATS_URL = "https://api.thecatapi.com/v1/images/search"
DOGS_URL = "https://dog.ceo/api/breeds/image/random"
COUNT = 100
DATA_DIR = Path("../shared/data/raw")

def setup_directories():
    (DATA_DIR / "cat").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "dog").mkdir(parents=True, exist_ok=True)
    print(f"Directories created at {DATA_DIR.resolve()}")

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def fetch_cats(count):
    print("Fetching cats...")
    saved = 0
    attempts = 0
    while saved < count and attempts < count * 2:
        try:
            resp = requests.get(CATS_URL).json()
            url = resp[0]['url']
            ext = os.path.splitext(url)[1]
            if not ext: ext = ".jpg"
            
            filename = f"cat_{saved:03d}{ext}"
            filepath = DATA_DIR / "cat" / filename
            
            if download_image(url, filepath):
                saved += 1
                print(f"Saved cat {saved}/{count}", end='\r')
            time.sleep(0.5) # Be nice to the API
        except Exception as e:
            print(f"Error fetching cat metadata: {e}")
        attempts += 1
    print(f"\nDone fetching cats. Saved {saved} images.")

def fetch_dogs(count):
    print("Fetching dogs...")
    saved = 0
    attempts = 0
    while saved < count and attempts < count * 2:
        try:
            resp = requests.get(DOGS_URL).json()
            if resp['status'] == 'success':
                url = resp['message']
                ext = os.path.splitext(url)[1]
                if not ext: ext = ".jpg"
                
                filename = f"dog_{saved:03d}{ext}"
                filepath = DATA_DIR / "dog" / filename
                
                if download_image(url, filepath):
                    saved += 1
                    print(f"Saved dog {saved}/{count}", end='\r')
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching dog metadata: {e}")
        attempts += 1
    print(f"\nDone fetching dogs. Saved {saved} images.")

if __name__ == "__main__":
    setup_directories()
    fetch_cats(COUNT)
    fetch_dogs(COUNT)
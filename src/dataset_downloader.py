import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil
import random

RAW_DIR = Path("/app/dataset/raw")
PROCESSED_DIR = Path("/app/dataset/processed")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, dest):
    if dest.exists():
        print(f"{dest.name} already exists skipping download.")
        return

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url} (status: {response.status_code})")

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=f"Downloading {dest.name}"
    ) as t:
        for data in response.iter_content(block_size):
            f.write(data)
            t.update(len(data))

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {zip_path.name}"):
            zip_ref.extract(member, extract_to)

def split_real_fake_dataset():
    train_dir, val_dir, test_dir = (
        PROCESSED_DIR / "task1" / "train",
        PROCESSED_DIR / "task1" / "val",
        PROCESSED_DIR / "task1" / "test",
    )

    for split in [train_dir, val_dir, test_dir]:
        for cls in ["real", "fake"]:
            (split / cls).mkdir(parents=True, exist_ok=True)

    train_ratio, val_ratio, test_ratio = (0.7, 0.2, 0.1)

    for cls in ["real", "fake"]:
        files = list((RAW_DIR / "hardfakevsrealfaces" / cls).glob("*.jpg"))
        random.shuffle(files)
        n_total = len(files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        for f in files[:n_train]:
            shutil.copy(f, train_dir / cls / f.name)
        for f in files[n_train:n_train + n_val]:
            shutil.copy(f, val_dir / cls / f.name)
        for f in files[n_train + n_val:]:
            shutil.copy(f, test_dir / cls / f.name)

    print("hardfakevsrealfaces dataset split complete!")

def split_comofod_dataset():
    src_dir = RAW_DIR / "CoMoFoD_small_v2"
    
    images_dir = PROCESSED_DIR / "task2" / "images"
    masks_dir  = PROCESSED_DIR / "task2" / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in src_dir.iterdir() if f.is_file()]

    for image in files:
        name = image.name
        parts = name.split("_")

        if len(parts) == 2 and "M" in name: #maskeler
            shutil.copy(image, masks_dir / name)
        elif len(parts) > 2 and ("F" in name or "O" in name): #gerçek ve manipüle edilmiş resimler
            shutil.copy(image, images_dir / name)

    print("comofod dataset split complete!")
    

download_file("https://www.kaggle.com/api/v1/datasets/download/hamzaboulahia/hardfakevsrealfaces", RAW_DIR / "hardfakevsrealfaces.zip")
unzip_file(RAW_DIR / "hardfakevsrealfaces.zip", RAW_DIR / "hardfakevsrealfaces")
split_real_fake_dataset()



download_file("https://www.kaggle.com/api/v1/datasets/download/tusharchauhan1898/comofod", RAW_DIR / "CoMoFoD_small_v2.zip")
unzip_file(RAW_DIR / "CoMoFoD_small_v2.zip", RAW_DIR)
split_comofod_dataset()
    



   
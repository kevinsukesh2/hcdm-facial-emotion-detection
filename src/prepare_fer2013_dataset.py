"""
Prepare the folder-based FER2013 dataset layout for later model training.

Expected raw dataset layout:
    data/raw/fer2013/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── neutral/
        ├── sad/
        └── surprise/

This script:
- uses raw `train/` as the source for processed `train/` and `val/`
- splits raw train images into 85% train and 15% val
- copies raw `test/` into processed `test/`
- optionally limits images per class
- uses a fixed random seed for reproducibility

This script does not train a model, run webcam inference, or build a dashboard.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path


RAW_FER2013_DIR = Path("data/raw/fer2013")
RAW_TRAIN_DIR = RAW_FER2013_DIR / "train"
RAW_TEST_DIR = RAW_FER2013_DIR / "test"
PROCESSED_FER2013_DIR = Path("data/processed/fer2013_subset")

EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
MAX_IMAGES_PER_CLASS = 1000
RANDOM_SEED = 42


def list_image_files(folder: Path) -> list[Path]:
    """Return sorted image files from a class folder."""
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_expected_structure() -> bool:
    """Validate that the raw FER2013 folder layout exists and is usable."""
    if not RAW_FER2013_DIR.exists():
        print(f"Raw FER2013 folder not found: {RAW_FER2013_DIR}")
        return False

    if not RAW_TRAIN_DIR.exists() or not RAW_TEST_DIR.exists():
        print("Expected raw FER2013 structure was not found.")
        print(f"Missing train or test folder inside: {RAW_FER2013_DIR}")
        return False

    for split_name, split_dir in [("train", RAW_TRAIN_DIR), ("test", RAW_TEST_DIR)]:
        missing_classes = [class_name for class_name in EMOTION_CLASSES if not (split_dir / class_name).exists()]
        if missing_classes:
            print(f"Missing class folders in raw {split_name}: {missing_classes}")
            return False

    return True


def reset_processed_directory() -> None:
    """Recreate only the split folders so tracked docs in the root can remain."""
    PROCESSED_FER2013_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_dir = PROCESSED_FER2013_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

        for class_name in EMOTION_CLASSES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def copy_files(files: list[Path], destination_dir: Path) -> None:
    """Copy image files into the destination class folder."""
    for source_file in files:
        shutil.copy2(source_file, destination_dir / source_file.name)


def prepare_train_and_val(rng: random.Random) -> dict[str, dict[str, int]]:
    """Split raw train images into processed train and val directories."""
    split_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}}

    for class_name in EMOTION_CLASSES:
        raw_class_dir = RAW_TRAIN_DIR / class_name
        image_files = list_image_files(raw_class_dir)

        if not image_files:
            print(f"No images found in raw train class folder: {raw_class_dir}")
            split_counts["train"][class_name] = 0
            split_counts["val"][class_name] = 0
            continue

        rng.shuffle(image_files)
        selected_files = image_files[:MAX_IMAGES_PER_CLASS]

        train_count = int(len(selected_files) * TRAIN_RATIO)
        train_files = selected_files[:train_count]
        val_files = selected_files[train_count:]

        copy_files(train_files, PROCESSED_FER2013_DIR / "train" / class_name)
        copy_files(val_files, PROCESSED_FER2013_DIR / "val" / class_name)

        split_counts["train"][class_name] = len(train_files)
        split_counts["val"][class_name] = len(val_files)

    return split_counts


def prepare_test_split() -> dict[str, int]:
    """Copy raw test images into the processed test directory."""
    split_counts: dict[str, int] = {}

    for class_name in EMOTION_CLASSES:
        raw_class_dir = RAW_TEST_DIR / class_name
        image_files = list_image_files(raw_class_dir)

        if not image_files:
            print(f"No images found in raw test class folder: {raw_class_dir}")
            split_counts[class_name] = 0
            continue

        selected_files = image_files[:MAX_IMAGES_PER_CLASS]
        copy_files(selected_files, PROCESSED_FER2013_DIR / "test" / class_name)
        split_counts[class_name] = len(selected_files)

    return split_counts


def print_summary(train_val_counts: dict[str, dict[str, int]], test_counts: dict[str, int]) -> None:
    """Print a compact summary of the prepared dataset counts."""
    print("\nPrepared image counts by split and class:")
    for split_name in ["train", "val"]:
        print(f"\n{split_name}:")
        for class_name in EMOTION_CLASSES:
            print(f"  {class_name}: {train_val_counts[split_name][class_name]}")

    print("\ntest:")
    for class_name in EMOTION_CLASSES:
        print(f"  {class_name}: {test_counts[class_name]}")


def main() -> None:
    """Prepare the processed FER2013 dataset structure from the raw folder layout."""
    print("Preparing FER2013 dataset")
    print(f"Raw dataset location: {RAW_FER2013_DIR}")
    print(f"Processed dataset location: {PROCESSED_FER2013_DIR}")
    print(f"Emotion classes: {EMOTION_CLASSES}")
    print(f"Max images per class: {MAX_IMAGES_PER_CLASS}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Train/val split from raw train: {int(TRAIN_RATIO * 100)}%/{int(VAL_RATIO * 100)}%")

    if not ensure_expected_structure():
        print("Dataset preparation aborted.")
        return

    rng = random.Random(RANDOM_SEED)
    reset_processed_directory()

    train_val_counts = prepare_train_and_val(rng)
    test_counts = prepare_test_split()
    print_summary(train_val_counts, test_counts)
    print(f"\nProcessed dataset created at: {PROCESSED_FER2013_DIR}")


if __name__ == "__main__":
    main()

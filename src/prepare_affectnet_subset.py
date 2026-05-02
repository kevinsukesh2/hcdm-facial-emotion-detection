"""
Skeleton planning script for preparing a manageable AffectNet subset.

Future purpose:
- read the locally stored AffectNet dataset from the raw data folder
- filter the dataset to the planned 8 emotion classes
- sample a manageable subset per class
- split the subset into train, validation, and test partitions
- write lightweight metadata and prepared outputs into the processed folder

This file is intentionally a skeleton only for Phase 2.
It does not perform real dataset processing yet.
"""

from pathlib import Path


# Planned local input folder for the original AffectNet dataset.
RAW_AFFECTNET_DIR = Path("data/raw/affectnet")

# Planned output folder for the prepared working subset.
PROCESSED_SUBSET_DIR = Path("data/processed/affectnet_subset")

# Planned emotion classes for the project.
EMOTION_CLASSES = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "contempt",
]

# Planned subset size per class for early experiments.
IMAGES_PER_CLASS = 1000

# Planned split ratio for the prepared subset.
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}


def main() -> None:
    """
    Placeholder entry point for future AffectNet subset preparation.

    In a later phase, this function will:
    - validate the expected raw dataset layout
    - load AffectNet annotations
    - select the planned 8 emotion classes
    - sample up to the target number of images per class
    - create train/val/test splits
    - write metadata files and processed outputs
    """
    print("AffectNet subset preparation skeleton")
    print(f"Expected raw dataset location: {RAW_AFFECTNET_DIR}")
    print(f"Planned processed output location: {PROCESSED_SUBSET_DIR}")
    print(f"Planned emotion classes: {EMOTION_CLASSES}")
    print(f"Planned images per class: {IMAGES_PER_CLASS}")
    print(f"Planned split ratios: {SPLIT_RATIOS}")

    if not RAW_AFFECTNET_DIR.exists():
        print("Raw AffectNet dataset not found yet. No processing will be done.")
        return

    print("Raw dataset folder exists, but real processing is not implemented yet.")


if __name__ == "__main__":
    main()

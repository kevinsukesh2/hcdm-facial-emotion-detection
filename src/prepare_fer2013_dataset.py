"""
Skeleton planning script for preparing the FER2013 dataset for training.

Future purpose:
- read the locally stored FER2013 dataset from the raw data folder
- validate the expected FER2013 files or folder layout
- prepare a manageable working subset when needed
- split the prepared dataset into train, validation, and test partitions
- write lightweight metadata and processed outputs into the processed folder

This file is intentionally a skeleton only for Phase 3.
It does not perform real dataset processing, model training, webcam inference,
or dashboard work yet.
"""

from pathlib import Path


# Planned local input folder for the raw FER2013 dataset.
RAW_FER2013_DIR = Path("data/raw/fer2013")

# Planned output folder for the prepared working dataset.
PROCESSED_FER2013_DIR = Path("data/processed/fer2013_subset")

# Planned emotion classes for the project.
EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Planned split names for the prepared dataset.
SPLITS = ["train", "val", "test"]


def main() -> None:
    """
    Placeholder entry point for future FER2013 dataset preparation.

    In a later phase, this function will:
    - check the raw FER2013 dataset layout
    - load source data and labels
    - prepare selected classes for training
    - create train/val/test outputs
    - write metadata files and processed outputs
    """
    print("FER2013 dataset preparation skeleton")
    print(f"Expected raw dataset location: {RAW_FER2013_DIR}")
    print(f"Planned processed output location: {PROCESSED_FER2013_DIR}")
    print(f"Planned emotion classes: {EMOTION_CLASSES}")
    print(f"Planned splits: {SPLITS}")

    if not RAW_FER2013_DIR.exists():
        print("Raw FER2013 dataset folder not found yet. No processing will be done.")
        return

    dataset_entries = list(RAW_FER2013_DIR.iterdir())
    if not dataset_entries:
        print("Raw FER2013 folder exists but does not contain the dataset yet. Exiting safely.")
        return

    print("Raw FER2013 folder has content, but real dataset preparation is not implemented yet.")


if __name__ == "__main__":
    main()

# Real-Time Webcam Emotion Detection Dashboard Using ResNet18 Fine-Tuned on FER2013

## Project Goal

This project will be built step by step to create a Python-based emotion detection system that can:

1. Fine-tune ResNet18 on the FER2013 dataset.
2. Split data into train, validation, and test sets.
3. Evaluate the model using accuracy, loss curves, a confusion matrix, and a classification report.
4. Run the trained model on a laptop webcam in real time.
5. Detect a face, crop it, predict an emotion, and display the predicted emotion with confidence.
6. Later provide a simple dashboard with webcam feed, current emotion, emotion timeline, and session summary.

## Important Note

This system estimates facial expression, not the user's true internal emotion.

## Planned Pipeline

1. Set up the project structure, documentation, dependencies, and version control.
2. Prepare dataset handling for FER2013 without assuming the dataset is already downloaded.
3. Build preprocessing and train/validation/test split utilities.
4. Fine-tune a ResNet18 classifier for facial expression recognition.
5. Evaluate the model with quantitative metrics and visualizations.
6. Load the trained model for real-time inference.
7. Add face detection and live webcam prediction.
8. Build a lightweight dashboard for live monitoring and session review.

## Step-by-Step Roadmap

### Step 1
Project setup, repository initialization, and GitHub preparation.

### Step 2
Initial dataset planning and data ingestion structure.

### Step 3
Dataset acquisition and preparation pivot to FER2013.

### Step 4
Model training configuration and ResNet18 fine-tuning.

### Step 5
Evaluation outputs including plots, confusion matrix, and report.

### Step 6
Real-time webcam inference pipeline.

### Step 7
Dashboard integration and session analytics.

## Dataset Plan

FER2013 is a facial expression recognition dataset.

The raw FER2013 dataset will be stored locally in `data/raw/fer2013/`.

A processed working subset will be stored in `data/processed/fer2013_subset/`.

FER2013 commonly uses these 7 emotion classes:

- `angry`
- `disgust`
- `fear`
- `happy`
- `sad`
- `surprise`
- `neutral`

To keep the first experiments manageable, we will start with a subset of about `1000` images per class.

That subset will later be split into:

- `70%` train
- `15%` validation
- `15%` test

The raw and processed dataset contents will not be uploaded to GitHub.

Only small metadata files and preparation scripts should be tracked by Git.

## Phase 3: Dataset Acquisition and Preparation

In this phase, the raw FER2013 dataset should be placed locally in `data/raw/fer2013/`.

The processed dataset will later be created in `data/processed/fer2013_subset/`.

Raw and processed image data must not be committed to GitHub.

## Webcam Inference

After training the tuned FER2013 model, you can run the simple webcam inference script:

```powershell
& ".\.venv\Scripts\Activate.ps1"
python src/webcam_inference.py
```

The script loads `models/resnet18_fer2013_tuned.pth`, opens the laptop webcam, detects faces, predicts one of the 7 FER2013 emotion classes, and shows the predicted emotion with confidence on screen.

Press `q` to quit the webcam window.

## Current Project Structure

```text
fer2013-webcam-emotion/
|
+-- data/
|   +-- raw/
|   |   +-- fer2013/
|   |       +-- README.md
|   +-- processed/
|   |   +-- fer2013_subset/
|   |       +-- README.md
|   +-- README.md
+-- models/
|   +-- README.md
+-- notebooks/
|   +-- README.md
+-- outputs/
|   +-- README.md
+-- src/
|   +-- README.md
|   +-- prepare_fer2013_dataset.py
+-- README.md
+-- requirements.txt
+-- .gitignore
```

## Current Status

The project now includes dataset preparation, training notebooks, and a simple webcam inference script. The full dashboard is not implemented yet.

# Real-Time Webcam Emotion Detection Dashboard Using ResNet18 Fine-Tuned on AffectNet

## Project Goal

This project will be built step by step to create a Python-based emotion detection system that can:

1. Fine-tune ResNet18 on the AffectNet dataset.
2. Split data into train, validation, and test sets.
3. Evaluate the model using accuracy, loss curves, a confusion matrix, and a classification report.
4. Run the trained model on a laptop webcam in real time.
5. Detect a face, crop it, predict an emotion, and display the predicted emotion with confidence.
6. Later provide a simple dashboard with webcam feed, current emotion, emotion timeline, and session summary.

## Important Note

This system estimates facial expression, not the user's true internal emotion.

## Planned Pipeline

1. Set up the project structure, documentation, dependencies, and version control.
2. Prepare dataset handling for AffectNet without assuming the dataset is already downloaded.
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
Dataset planning and data ingestion structure for AffectNet.

### Step 3
Preprocessing pipeline and train/validation/test split workflow.

### Step 4
Model training configuration and ResNet18 fine-tuning.

### Step 5
Evaluation outputs including plots, confusion matrix, and report.

### Step 6
Real-time webcam inference pipeline.

### Step 7
Dashboard integration and session analytics.

## Dataset Plan

The AffectNet dataset will be stored locally in `data/raw/affectnet/`.

A processed working subset will be stored in `data/processed/affectnet_subset/`.

We will use these 8 emotion classes:

- `neutral`
- `happy`
- `sad`
- `surprise`
- `fear`
- `disgust`
- `anger`
- `contempt`

To keep the first experiments manageable, we will start with a subset of about `1000` images per class.

That subset will later be split into:

- `70%` train
- `15%` validation
- `15%` test

The dataset itself will not be uploaded to GitHub.

Only small metadata files and preparation scripts should be tracked by Git.

## Current Project Structure

```text
affectnet-webcam-emotion/
|
+-- data/
|   +-- raw/
|   |   +-- affectnet/
|   |       +-- README.md
|   +-- processed/
|   |   +-- affectnet_subset/
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
|   +-- prepare_affectnet_subset.py
+-- README.md
+-- requirements.txt
+-- .gitignore
```

## Current Status

The project now includes setup and dataset planning only. No model training, webcam inference, or dashboard implementation has been added yet.

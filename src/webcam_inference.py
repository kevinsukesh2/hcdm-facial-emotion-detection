"""
Simple webcam inference script for FER2013 emotion recognition.

What this script does:
- loads the tuned ResNet18 model trained on FER2013
- opens the laptop webcam with OpenCV
- detects faces with OpenCV's built-in Haar cascade
- crops the detected face
- applies the same style of preprocessing used during training
- predicts one of the 7 FER2013 emotion classes
- displays the predicted emotion and confidence above each face

Press `q` to quit the webcam window.
"""

from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_fer2013_tuned.pth"

FACE_CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform() -> transforms.Compose:
    """Return the image transform used for webcam face crops."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model() -> torch.nn.Module:
    """Load the tuned ResNet18 model and move it to the active device."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's built-in Haar cascade face detector."""
    detector = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector from: {FACE_CASCADE_PATH}")
    return detector


def predict_emotion(model: torch.nn.Module, transform: transforms.Compose, face_bgr) -> tuple[str, float]:
    """Run inference on a cropped face and return the top emotion and confidence."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_image = Image.fromarray(face_rgb)
    input_tensor = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_label = CLASS_NAMES[predicted_index.item()]
    predicted_confidence = confidence.item()
    return predicted_label, predicted_confidence


def main() -> None:
    """Open the webcam, detect faces, and show live FER2013 emotion predictions."""
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")

    model = load_model()
    transform = build_transform()
    face_detector = load_face_detector()

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    print("Webcam started. Press 'q' to quit.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to read a frame from the webcam.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray_frame,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for (x, y, w, h) in faces:
                face_crop = frame[y : y + h, x : x + w]
                if face_crop.size == 0:
                    continue

                emotion, confidence = predict_emotion(model, transform, face_crop)
                label_text = f"{emotion} ({confidence * 100:.1f}%)"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label_text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("FER2013 Webcam Emotion Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

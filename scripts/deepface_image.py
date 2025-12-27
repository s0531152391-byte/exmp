#!/usr/bin/env python3
"""
Analyze a single image with DeepFace for emotion.
Usage:
    python scripts/deepface_image.py --img path/to/photo.jpg
"""
import argparse
from deepface import DeepFace
import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to the image file")
    parser.add_argument("--detector", default="opencv", help="face detector backend (opencv, mtcnn, retinaface, mediapipe, ssd)")
    args = parser.parse_args()

    print("Analyzing:", args.img)
    # actions includes "emotion" for emotion recognition
    obj = DeepFace.analyze(img_path=args.img, actions=["emotion"], detector_backend=args.detector)
    # DeepFace returns dict with 'dominant_emotion' and 'emotion' distributions
    pprint.pprint({
        "dominant_emotion": obj.get("dominant_emotion"),
        "emotions": obj.get("emotion")
    }, width=120)

if __name__ == "__main__":
    main()

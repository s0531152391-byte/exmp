#!/usr/bin/env python3
"""
Real-time webcam emotion recognition using DeepFace (simple).
Press 'q' to quit.
"""
import time
import cv2
from deepface import DeepFace

# Config
camera_index = 0            # 如果电脑有多个摄像头，可改为 1,2...
frame_skip = 6              # 每处理第 frame_skip 帧做一次分析（降低计算）
detector_backend = "opencv" # 可改为 "mtcnn","retinaface","mediapipe","ssd"

def draw_emotion(frame, face_region, emotions, dominant):
    x, y, w, h = face_region
    # draw box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 200, 10), 2)
    # label
    label = f"{dominant}"
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,200,10), 2)
    # draw bars
    start_y = y + h + 10
    max_w = w
    i = 0
    for k, v in emotions.items():
        bar_w = int(max_w * float(v))
        cv2.putText(frame, f"{k[:6]}:{int(v*100)}%", (x, start_y + i*18 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.rectangle(frame, (x, start_y + i*18 + 8), (x + bar_w, start_y + i*18 + 8 + 8), (100,180,250), -1)
        i += 1

def main():
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {camera_index}")
        return

    frame_count = 0
    last_emotions = None
    print("Starting webcam. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_count += 1
            # Resize for speed (optional)
            display = frame.copy()

            if frame_count % frame_skip == 0:
                # run DeepFace analyze on the current frame
                # To speed up, we analyze whole frame; DeepFace will try to detect face(s)
                try:
                    t0 = time.time()
                    result = DeepFace.analyze(frame, actions=["emotion"], detector_backend=detector_backend, enforce_detection=False)
                    # result may be a list if multiple faces; ensure dict
                    if isinstance(result, list) and len(result) > 0:
                        r = result[0]
                    else:
                        r = result
                    emotions = r.get("emotion", {})
                    dominant = r.get("dominant_emotion", "")
                    # If face detection failed, emotions could be empty
                    last_emotions = (emotions, dominant)
                    t1 = time.time()
                    # print timing
                    # print(f"Analysis time: {(t1-t0):.2f}s")
                except Exception as e:
                    # sometimes analysis fails; keep last_emotions
                    print("Analyze error:", e)

            # overlay results if available
            if last_emotions:
                emotions, dominant = last_emotions
                # DeepFace does not always return face box for enforce_detection=False.
                # For a robust demo we'll detect faces via OpenCV cascade to get coordinates for drawing.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
                if len(faces) > 0:
                    # draw for first face
                    (x,y,w,h) = faces[0]
                    draw_emotion(display, (x,y,w,h), emotions, dominant)
                else:
                    # fallback: put label at top-left
                    cv2.putText(display, f"{dominant}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10,200,10), 2)

            cv2.imshow('DeepFace - Emotion (q to quit)', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

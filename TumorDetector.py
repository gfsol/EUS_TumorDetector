import os
from ultralytics import YOLO
import cv2

# === USER INPUT SECTION ===
model_path = input("Enter the path to your YOLO model (.pt file): ").strip()
video_path = input("Enter the path to your input video file: ").strip()

# === Load model ===
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")
model = YOLO(model_path)

# === Preprocessing Function ===
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray)
    img_out = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    return img_out

# === Open video ===
if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file not found at: {video_path}")
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Output video writer ===
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter("detectedVideo.mp4", fourcc, fps, (width, height))

print("Starting tumor detection on video... Press 'q' to exit early.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pFrame = preprocess_frame(frame)
    results = model(pFrame)
    annotatedFrame = results[0].plot()

    resized_frame = cv2.resize(annotatedFrame, (width, height), interpolation=cv2.INTER_AREA)
    output.write(resized_frame)
    cv2.imshow("Video Detection", resized_frame)

    # Optional: center window (works on some OS)
    cv2.moveWindow("Video Detection", (1920 - resized_frame.shape[1]) // 2, (1080 - resized_frame.shape[0]) // 2)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Detection stopped by user.")
        break

cap.release()
output.release()
cv2.destroyAllWindows()
print("Detection complete. Output saved as: detectedVideo.mp4")

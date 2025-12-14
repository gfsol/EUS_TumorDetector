from ultralytics import YOLO

# Replace with your dataset and model paths
MODEL_PATH = "yolo11n.pt"  # Pretrained YOLOv11n model
DATA_CONFIG = "data.yaml"  # YOLO data configuration file

def main():
    model = YOLO(MODEL_PATH)
    model.train(
        data=DATA_CONFIG,
        imgsz=512,
        epochs=100,
        batch=8,
        project="runs",
        name="detect",
        exist_ok=True  # Allow overwriting existing runs
    )

if __name__ == "__main__":
    main()

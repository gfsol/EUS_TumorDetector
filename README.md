# 🧠 Pancreatic Tumor Detection in Endoscopic Ultrasound Images (EUS)

A real-time deep learning system for detecting pancreatic tumors in endoscopic ultrasound (EUS) images using [YOLOv11](https://github.com/ultralytics/ultralytics). This project fine-tunes a YOLOv11n model on the IAEUS dataset and supports both static image and live video inference.

<video width="100%" controls>
  <source src="assets/example_detection.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🫀 Motivation

Early detection of pancreatic tumors is critical — yet extremely challenging.  
Pancreatic cancer has one of the highest mortality rates due to late diagnosis.  

This project was born out of a deep motivation to apply AI for real-world impact:  
assisting radiologists and clinicians by providing a fast, accurate tool that  
can help flag suspicious regions during EUS examinations.

**This is just the beginning** — my long-term goal is to develop deep learning models  
for more common imaging modalities such as **CT scans**, where automated tumor detection  
could be applied in routine checkups, screenings, or post-operative follow-up.

---

## 📌 Project Overview

- **Goal**: Automatically identify tumor regions in grayscale endoscopic ultrasound images, in real-time or static mode.
- **Dataset**: Public IAEUS dataset — 7,825 annotated EUS images from 606 patients.
- **Model**: YOLOv11n (Ultralytics) fine-tuned for 7 tumor-related classes.
- **Hardware**: Trained and tested using NVIDIA RTX 4060 (CUDA 12.1).

---

## 🧪 Dataset

- **Source**: IAEUS — publicly available dataset of EUS images.
- **Size**: 7,825 grayscale images from 606 patients.
- **Format**: YOLO-compatible bounding box annotations in `.txt` files.
- **Classes (7)**:
  - Tumor
  - Difficile
  - Adenocarcinome
  - Tumeur endocrine
  - Hors pancreas
  - To discard
  - Tumor mask

### Directory Structure

```
TumorDetector/
├── data.yaml               # Dataset configuration
├── images/                 # EUS images
│   ├── train/pos
│   ├── train/neg
│   ├── val/pos
│   └── val/neg
├── labels/                 # YOLO-format bounding box annotations
│   ├── train/pos
│   ├── train/neg
│   ├── val/pos
│   └── val/neg
```

---

## ⚙️ Setup & Environment

**Python**: 3.11  
**Virtual Environment**: Created with `venv` or `conda`  
**Libraries**:
- `ultralytics==8.3.227`
- `torch==2.5.1+cu121`
- `opencv-python`
- `numpy`
- `matplotlib`
- `pyyaml`
- `torchvision`, `torchaudio`

### Activate Environment

```bash
# Windows PowerShell
& yolo11_env/Scripts/activate

# or Bash
source yolo11_env/Scripts/activate
```

---

## 🏋️‍♂️ Model Training

Uses YOLOv11n (lightweight) for speed and efficiency.

```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=512 epochs=100 batch=8
```

- **Input size**: 512x512
- **Epochs**: 100
- **Batch size**: 8
- **Outputs**:
  - Trained weights in `runs/detect/<run_name>/weights/`
  - Training logs, plots, label heatmaps

**Sample Output Path**:
```
runs/detect/train17/
├── weights/
│   ├── best.pt
│   └── last.pt
├── labels.jpg
└── results.png
```

---

## 📈 Performance (Validation)

On the IAEUS validation set (852 images, 310 tumor instances):

| Metric   | Score |
|----------|-------|
| Precision | ~0.72 |
| Recall    | ~0.76 |
| mAP@0.5   | ~0.76 |

- **High Confidence** (> 0.7): strong detection
- **Medium** (0.4–0.7): partial detection, shadow, artifact
- **Low** (< 0.4): likely false positive

---

## 🔍 Inference (Images & Video)

Run inference on static images, video files, or live webcam.

### Example: Image

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train17/weights/best.pt")
results = model("test_image.jpg")
results.show()
```

### Example: Video / Webcam

```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train17/weights/best.pt")
cap = cv2.VideoCapture(0)  # Or path to video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=512, conf=0.25)
    annotated = results[0].plot()

    cv2.imshow("Tumor Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧪 Evaluation Without Bounding Boxes

If your test images do **not** include bounding boxes:
- Use model predictions and analyze **confidence scores**
- A detection is considered “positive” if any prediction exceeds a confidence threshold (e.g., 0.5)
- Useful for evaluating datasets with image-level labels only

---

## 🧠 Notes

- Model generalizes well on grayscale EUS images; may underperform on colored or heavily artifacted data.
- Inference is fast and lightweight with YOLOv11n, suitable for real-time applications.
- Modular structure: separate scripts for training (`train.py`), inference (`detector.py`), and evaluation.

---

## 🧩 Future Improvements

- Add temporal smoothing for video inference
- Improve class-wise performance (especially for similar categories)
- Implement Grad-CAM or explainability techniques
- Integrate test-time augmentation (TTA)

---

## 📚 Citation & Dataset

If you use the IAEUS dataset, please cite the original paper:
> [Insert dataset citation and link here]

---

## 📎 License

This project is for **research and educational purposes** only. Not intended for clinical use.

---

## 🙌 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 framework
- Public IAEUS dataset contributors

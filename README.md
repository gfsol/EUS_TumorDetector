# 🧠 Pancreatic Tumor Detection in Endoscopic Ultrasound Images (EUS)

A real-time deep learning system for detecting pancreatic tumors in endoscopic ultrasound (EUS) images using [YOLOv11](https://github.com/ultralytics/ultralytics). This project fine-tunes a YOLOv11n model on the IAEUS dataset and supports both static image and live video inference.

🎥 **Demo video:**  
[Click here to watch the detection demo](https://github.com/gfsol/EUS_TumorDetector/blob/main/assets/example_detection.mp4)

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

### IAEUS – Endoscopic Ultrasound Dataset

- **Official dataset website**: https://iaeus.im-lis.com/
- **Size**: 7,825 grayscale EUS images from 606 patients
- **Annotations**: Bounding boxes provided by the dataset authors
- **Classes (7)**:
  - Tumor
  - Difficile
  - Adenocarcinome
  - Tumeur endocrine
  - Hors pancreas
  - To discard
  - Tumor mask

⚠️ **Important note on data usage**  
Due to licensing and redistribution restrictions, **this repository does NOT include the IAEUS dataset**, neither images nor annotation files.

The dataset was **preprocessed and reorganized** locally to match a YOLO-compatible structure for training and validation. Only the **scripts, configuration files, and folder structure** are provided here to ensure reproducibility.

---

### Dataset Preparation

To reproduce the experiments:

1. Download the dataset from the official source:  
   👉 https://iaeus.im-lis.com/
2. Follow the dataset terms of use and citation requirements.
3. Organize the data locally using the provided structure or preparation scripts.

Example structure used for training:

```
dataset/
├── images/
│   ├── train/pos
│   ├── train/neg
│   ├── val/pos
│   └── val/neg
├── labels/
│   ├── train/pos
│   ├── train/neg
│   ├── val/pos
│   └── val/neg
```

---

## ⚙️ Setup & Environment

**Python**: 3.11  
**Virtual Environment**: Created with `venv` or `conda`  

**Main Libraries**:
- `ultralytics==8.3.227`
- `torch==2.5.1+cu121`
- `opencv-python`
- `numpy`
- `matplotlib`
- `pyyaml`
- `torchvision`, `torchaudio`

---

## 🏋️‍♂️ Model Training

Uses YOLOv11n (lightweight) for speed and efficiency.

```bash
yolo detect train model=yolo11n.pt data=data.yaml imgsz=512 epochs=100 batch=8
```

**Training configuration**:
- Input size: 512×512
- Epochs: 100
- Batch size: 8

**Outputs**:
```
runs/detect/<run_name>/
├── weights/
│   ├── best.pt
│   └── last.pt
├── labels.jpg
└── results.png
```

---

## 📈 Performance (Validation)

Validation on the IAEUS validation split (852 images, 310 tumor instances):

| Metric    | Score |
|-----------|-------|
| Precision | ~0.72 |
| Recall    | ~0.76 |
| mAP@0.5   | ~0.76 |

Confidence interpretation:
- **High** (> 0.7): strong detection
- **Medium** (0.4–0.7): partial tumor or artifact
- **Low** (< 0.4): likely false positive

---

## 🔍 Inference

Once trained, the model can be used to analyze static images or video sequences. The predictions include bounding boxes and confidence scores over tumor regions.

---

## 🧪 Evaluation Without Bounding Boxes

For datasets with image-level labels only:
- A frame is considered positive if **any detection exceeds a confidence threshold**
- Useful for weakly supervised or external validation datasets

---

## 🧠 Notes

- Optimized for grayscale EUS images
- Lightweight and suitable for near real-time inference
- Modular codebase (training, inference, preprocessing separated)

---

## 🧩 Future Improvements

- Temporal smoothing for video predictions
- Extension to CT imaging
- Explainability methods (Grad-CAM)
- Test-time augmentation (TTA)

---

## 📚 Citation

Please cite the **IAEUS dataset authors** when using this work.  
Dataset website: https://iaeus.im-lis.com/

---

## 📎 License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

---

## 🙌 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- IAEUS dataset contributors

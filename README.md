# ThyroNet: Thyroid Nodule Analysis

> AI-powered thyroid cancer detection using deep learning and ensemble machine learning.
> B.Tech Major Project — NMAM Institute of Technology, Nitte — April 2026

---

## What is ThyroNet?

ThyroNet is a complete web-based AI system for thyroid nodule analysis. It combines deep learning image classification with clinical risk assessment into one Flask web application.

**4 integrated modules:**
- Image Analysis — YOLO detection + 8-model CNN ensemble
- Cancer Risk Assessment — 6 ML classifiers on 50,000 patients
- ACR TI-RADS Calculator — standardized ultrasound scoring
- Thyroid Disease Detection — Hypo/Hyper/Normal classification

---

## Results

| Module | Model | Performance |
|--------|-------|-------------|
| YOLO Detection | YOLOv8l / YOLOv9c | mAP50 = 97.4% |
| Image Classification | 8-Model Ensemble | 87% accuracy, Sensitivity 89.4% |
| Cancer Risk | Voting Ensemble | 84.16% accuracy, AUC 90.59% |
| Disease Detection | XGBoost | 93.42% accuracy |
| TI-RADS | Rule-based | 100% deterministic |

---

## Tech Stack

- **Backend:** Python 3.10, Flask, PyTorch 2.2.0
- **Detection:** Ultralytics YOLOv8m, YOLOv8l, YOLOv9c
- **Classifiers:** DenseNet121, ResNet50, ResNet152, MobileNetV3, EfficientNetB4, VGG16, InceptionV3, ConvNeXt
- **ML:** scikit-learn, XGBoost 2.0.3, timm
- **Image Processing:** OpenCV 4.8, Pillow
- **Frontend:** Bootstrap 5.3, JavaScript ES6+

---

## Dataset

- **TN5000** — 5,000 thyroid ultrasound images, VOC XML annotations, 80/20 split
- **thyroid_nodule_clinical.csv** — 50,000 patient records (ATA 2015 guidelines)
- **UCI thyroidDF.csv** — 9,168 real patients

---

## Run Locally

`ash
pip install -r requirements.txt
python app_enhanced.py
`

Open http://localhost:5000

Login: doctor / 	hyronet2024

> Note: Model files (.pt) are not included in this repo due to size (1.2GB total).
> Models were trained on Kaggle using 2x NVIDIA Tesla T4 GPUs.

---

## Project Structure

`
thyronet-ai/
├── app_enhanced.py              # Flask application
├── utils/
│   ├── super_ensemble_predictor.py   # YOLO + 8 CNN classifiers
│   ├── cancer_risk_predictor.py      # 6 ML models
│   ├── thyroid_disease_predictor.py  # Disease detection
│   └── tirads_calculator.py          # ACR TI-RADS 2017
├── templates/                   # HTML templates
├── static/                      # CSS, JS, demo images
├── demo_images/                 # 8 sample ultrasound images
├── thyroidDF.csv                # UCI thyroid dataset
├── thyroid_nodule_clinical.csv  # Cancer risk dataset
└── requirements.txt
`

---

**Institution:** NMAM Institute of Technology, Nitte
**Department:** Information Science and Engineering

---

## Publication

Presented at **PTEMS-2026** — International Conference on Progressive Trends in Engineering, Management and Science, April 2026, Greater Noida, India.

---

*For research and educational purposes only. Not a certified medical device.*

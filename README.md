# 🏥 Thyroid Cancer Risk Prediction System

A production-ready AI web application that predicts Malignant/Benign thyroid cancer risk using both:
- **Image Analysis** (Deep Learning - DenseNet121)
- **Numerical Features** (ML + Deep Learning models)

## ✨ Features

### 🎯 Dual Prediction Modes
1. **Image-Based Prediction**: Upload medical images for instant analysis
2. **Numerical-Based Prediction**: Enter patient data for risk assessment

### 🤖 Advanced ML Pipeline
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - Gradient Boosting

- **Deep Learning Model**:
  - Dense Neural Network (3 hidden layers with dropout)
  - EarlyStopping for optimal training

### 📊 Data Handling
- Automatic class imbalance detection
- SMOTE oversampling for balanced training
- StandardScaler for feature normalization
- Cross-validation for robust evaluation
- Stratified train/test split

### 🎨 Professional Web Interface
- Responsive medical-grade UI
- Drag-and-drop image upload
- Real-time prediction results
- Confidence scores and probability bars
- Color-coded risk indicators (Green=Benign, Red=Malignant)

## 📁 Project Structure

```
project/
├── app.py                      # Flask web application
├── train_models.py             # ML/DL training pipeline
├── utils/
│   ├── image_predictor.py      # Image prediction module
│   └── numerical_predictor.py  # Numerical prediction module
├── models/                     # Saved models directory
│   ├── best_model.pkl/.h5      # Best performing model
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoder.pkl       # Label encoder
│   ├── metadata.json           # Model metadata
│   └── feature_names.json      # Feature list
├── templates/
│   └── index.html              # Web interface
├── static/                     # Static assets
├── uploads/                    # Temporary upload folder
├── requirements.txt            # Python dependencies
├── densenet121_best.pth        # Pre-trained image model
├── thyroid_cancer_risk_data.csv # Training dataset
└── README.md                   # This file
```

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python train_models.py
```

This will:
- Load and preprocess the CSV data
- Handle class imbalance with SMOTE
- Train 6 different models (5 ML + 1 DL)
- Perform cross-validation
- Select the best model (90%+ accuracy target)
- Save models and artifacts to `models/` directory
- Generate confusion matrix and ROC curve plots

### 3. Run Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## 📊 Model Training Details

### Preprocessing Pipeline
1. **Data Loading**: Reads CSV and detects target column
2. **Missing Values**: Filled with median values
3. **Categorical Encoding**: LabelEncoder for categorical features
4. **Class Imbalance**: SMOTE applied if imbalance ratio > 1.5
5. **Feature Scaling**: StandardScaler normalization
6. **Train/Test Split**: 80/20 with stratification

### Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- 5-Fold Cross-Validation

### Best Model Selection
The system automatically selects the model with highest accuracy. All results are logged and saved.

## 🌐 API Endpoints

### 1. Image Prediction
```
POST /predict-image
Content-Type: multipart/form-data

Body: image file

Response:
{
  "prediction": "Benign" | "Malignant",
  "confidence": 95.67,
  "probabilities": {
    "Benign": 95.67,
    "Malignant": 4.33
  },
  "success": true
}
```

### 2. Numerical Prediction
```
POST /predict-numerical
Content-Type: application/json

Body:
{
  "features": {
    "Age": 45,
    "Gender": 1,
    "TSH_Level": 2.5,
    ...
  }
}

Response:
{
  "prediction": "Benign" | "Malignant",
  "confidence": 92.34,
  "probabilities": {
    "Benign": 92.34,
    "Malignant": 7.66
  },
  "model_used": "XGBoost",
  "success": true
}
```

### 3. Get Feature Names
```
GET /get-features

Response:
{
  "success": true,
  "features": ["Age", "Gender", "TSH_Level", ...]
}
```

## 🎯 Usage Examples

### Image Prediction
1. Navigate to "Upload Image" tab
2. Click or drag-and-drop an image
3. Click "Analyze Image"
4. View prediction with confidence scores

### Numerical Prediction
1. Navigate to "Enter Features" tab
2. Fill in all patient data fields
3. Click "Predict Risk"
4. View prediction with probability breakdown

## 📈 Performance Targets

- **Minimum Accuracy**: 90%
- **Class Balance**: Handled via SMOTE
- **Cross-Validation**: 5-fold CV for reliability
- **Model Selection**: Automatic best model selection

## 🔒 Production Features

- Error handling for all endpoints
- File size limits (16MB max)
- File type validation
- Input validation
- Secure file handling
- No debug mode in production
- Environment variable support

## 🚢 Deployment

### Option 1: Render

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: thyroid-prediction
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.10.0
```

2. Deploy:
```bash
git init
git add .
git commit -m "Initial commit"
# Push to GitHub and connect to Render
```

### Option 2: Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Deploy:
```bash
railway login
railway init
railway up
```

### Option 3: Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

2. Build and run:
```bash
docker build -t thyroid-prediction .
docker run -p 5000:5000 thyroid-prediction
```

## 🛠️ Environment Variables

```bash
# Optional configuration
FLASK_ENV=production
PORT=5000
MAX_CONTENT_LENGTH=16777216  # 16MB
```

## 📝 Model Artifacts

After training, the following files are generated:

- `models/best_model.pkl` or `best_model.h5` - Best performing model
- `models/scaler.pkl` - Feature scaler
- `models/label_encoder.pkl` - Label encoder
- `models/metadata.json` - Model information
- `models/feature_names.json` - Feature list
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve plot

## 🔍 Troubleshooting

### Models not found
```bash
# Ensure you've run training first
python train_models.py
```

### Port already in use
```bash
# Change port in app.py or use environment variable
export PORT=8000
python app.py
```

### CUDA/GPU issues
```bash
# Models will automatically fall back to CPU
# No GPU required for inference
```

## 📊 Expected Results

After training, you should see output similar to:

```
📊 Loading dataset...
✅ Target column detected: Diagnosis
📈 Dataset shape: (370, 15)
🎯 Class distribution: {0: 280, 1: 90}
⚖️  Imbalance ratio: 3.11
🔄 Applying SMOTE to balance dataset...
✅ After SMOTE: {0: 280, 1: 280}
✅ Preprocessing complete!

🤖 Training Machine Learning Models...

Training Logistic Regression...
  Accuracy: 0.9189 | Precision: 0.8750 | Recall: 0.8235
  F1-Score: 0.8485 | ROC-AUC: 0.9654 | CV: 0.9054±0.0234

Training Random Forest...
  Accuracy: 0.9459 | Precision: 0.9286 | Recall: 0.8824
  F1-Score: 0.9048 | ROC-AUC: 0.9823 | CV: 0.9324±0.0189

...

✅ Best Model: XGBoost
✅ Accuracy: 0.9595
✅ Model saved to: models/
```

## 🎓 Technical Stack

- **Backend**: Flask
- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **DL**: TensorFlow/Keras, PyTorch
- **Image Processing**: PIL, torchvision
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 standards
- All tests pass
- Documentation is updated

## 📧 Support

For issues or questions, please open a GitHub issue.

---

Built with ❤️ for medical AI applications

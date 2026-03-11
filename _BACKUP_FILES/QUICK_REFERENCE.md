# 🚀 QUICK REFERENCE CARD

## 📋 3-Step Quick Start

```bash
# Step 1: Train numerical models (2-5 minutes)
python quick_train.py

# Step 2: Run web application
python app_enhanced.py

# Step 3: Open browser
# Visit: http://localhost:5000
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `quick_train.py` | Quick training script - RUN THIS FIRST |
| `advanced_train_numerical.py` | Full training pipeline (400+ lines) |
| `app_enhanced.py` | Multi-page web application |
| `thyroid_cancer_risk_data.csv` | Training data (370 patients) |
| `densenet121_best.pth` | Pre-trained image model |

## 🎯 What Gets Trained

### 4 ML Models
1. **Logistic Regression** - Linear baseline
2. **SVM** - Non-linear boundaries
3. **Random Forest** - Ensemble learning
4. **XGBoost** - Best performer (95%+ accuracy)

### Training Features
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ Imbalance handling (SMOTETomek)
- ✅ 5-fold cross-validation
- ✅ Robust scaling
- ✅ Overfitting prevention

## 📊 Expected Results

```
🏆 BEST MODEL: XGBoost
📊 Accuracy: 95.95%
📊 ROC-AUC: 0.9654
✅ TARGET ACHIEVED: 95.95% >= 85.00%
```

## 📁 Generated Files

### After Training
```
models/
├── best_model.pkl              # XGBoost (best)
├── logistic_regression_model.pkl
├── svm_model.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── feature_names.json
└── metadata.json

Visualizations:
├── confusion_matrices_all.png
├── roc_curves_all.png
├── model_comparison_metrics.png
└── best_model_confusion_matrix.png

Reports:
├── model_comparison.csv
└── classification_report_*.txt (4 files)
```

## 🌐 Web Pages

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Landing page |
| Predict | `/predict` | Make predictions |
| Model Info | `/model-info` | Model details |
| Dashboard | `/dashboard` | Performance metrics |
| About | `/about` | System info |

## 🔌 API Endpoints

### Numerical Prediction
```bash
POST /predict-numerical
Content-Type: application/json

{
  "features": {
    "Age": 45,
    "Gender": 1,
    "TSH_Level": 2.5,
    ...
  }
}
```

### Image Prediction
```bash
POST /predict-image
Content-Type: multipart/form-data

Body: image file
```

### Get Features
```bash
GET /get-features
```

## 📊 Model Comparison

| Model | Accuracy | ROC-AUC | Speed |
|-------|----------|---------|-------|
| Logistic Regression | 91.89% | 0.9654 | ⚡⚡⚡ |
| SVM | 93.24% | 0.9823 | ⚡⚡ |
| Random Forest | 94.59% | 0.9823 | ⚡⚡ |
| **XGBoost** | **95.95%** | **0.9654** | **⚡** |

## 🔧 Common Commands

### Training
```bash
# Full training with all features
python quick_train.py

# Check training progress
# Watch console output for progress

# Verify models trained
ls models/
```

### Web Application
```bash
# Start server
python app_enhanced.py

# Access application
# Browser: http://localhost:5000

# Stop server
# Press Ctrl+C
```

### Testing
```bash
# Verify setup
python verify_setup.py

# Test system (after starting app)
python test_system.py
```

## 🐛 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "Predictor not available" | Run `python quick_train.py` first |
| Low accuracy (<85%) | Expand hyperparameter grids |
| Training too slow | Reduce grid size or use fewer folds |
| Port 5000 in use | Change port in `app_enhanced.py` |
| Missing dependencies | Run `pip install -r requirements.txt` |

## 📈 Performance Tips

### Increase Accuracy
- Expand hyperparameter grids
- Feature engineering
- Collect more data
- Try ensemble methods

### Reduce Training Time
- Reduce grid size
- Use RandomizedSearchCV
- Fewer CV folds
- Parallel processing (already enabled)

### Prevent Overfitting
- Increase regularization
- Reduce model complexity
- More cross-validation
- Early stopping

## 🎯 Key Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Accuracy** | Overall correctness | >85% |
| **Precision** | Positive predictive value | >0.85 |
| **Recall** | Sensitivity | >0.85 |
| **F1-Score** | Balance of precision/recall | >0.85 |
| **ROC-AUC** | Discrimination ability | >0.90 |

## 📚 Documentation Files

| File | Content |
|------|---------|
| `FINAL_SUMMARY.md` | Complete system overview |
| `COMPLETE_SETUP_GUIDE.md` | Detailed setup instructions |
| `NUMERICAL_SYSTEM_README.md` | Technical documentation |
| `QUICK_REFERENCE.md` | This file |

## ⚠️ Important Notes

### Medical Disclaimer
- ⚠️ Research and educational purposes only
- ⚠️ NOT for clinical diagnosis
- ⚠️ Consult medical professionals

### Data Privacy
- ✅ No data stored by default
- ✅ Local processing only
- ✅ No external API calls

### System Requirements
- Python 3.8+
- 2-4 GB RAM
- ~50 MB disk space
- Modern web browser

## 🎊 Success Checklist

- [ ] Dependencies installed
- [ ] CSV data file present
- [ ] Run `python quick_train.py`
- [ ] Models trained successfully
- [ ] Run `python app_enhanced.py`
- [ ] Web app accessible
- [ ] Make test prediction
- [ ] Review visualizations

## 📞 Need Help?

1. Check `COMPLETE_SETUP_GUIDE.md`
2. Review error messages
3. Verify all files present
4. Check Python version
5. Ensure dependencies installed

## 🚀 Ready to Start?

```bash
# Just run this:
python quick_train.py
```

---

**ThyroNet AI - Competition-Level Medical Classification**
**Built with ❤️ for medical AI research**

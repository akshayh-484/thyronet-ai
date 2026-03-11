# 📖 Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Image Prediction](#image-prediction)
3. [Numerical Prediction](#numerical-prediction)
4. [API Usage](#api-usage)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Windows Users
```bash
# Double-click start.bat or run:
start.bat
```

### Mac/Linux Users
```bash
# Make script executable
chmod +x start.sh

# Run
./start.sh
```

### Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

# Start application
python app.py
```

Visit: **http://localhost:5000**

---

## Image Prediction

### Step 1: Navigate to Image Tab
Click on "📸 Upload Image" tab

### Step 2: Upload Image
Two methods:
1. **Click** the upload area and select an image
2. **Drag and drop** an image onto the upload area

### Step 3: Analyze
Click "Analyze Image" button

### Step 4: View Results
- **Prediction**: Benign or Malignant
- **Confidence**: Percentage confidence (0-100%)
- **Probability Bars**: Visual representation of probabilities

### Supported Image Formats
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

### Image Requirements
- Maximum size: 16MB
- Recommended: Clear, well-lit medical images
- Automatically resized to 224x224 for processing

---

## Numerical Prediction

### Step 1: Navigate to Features Tab
Click on "📊 Enter Features" tab

### Step 2: Fill in Patient Data
Enter values for all fields:

#### Demographic Features
- **Age**: Patient age (years)
- **Gender**: 0 = Female, 1 = Male

#### Medical History
- **Family_History**: 0 = No, 1 = Yes
- **Radiation_Exposure**: 0 = No, 1 = Yes
- **Iodine_Deficiency**: 0 = No, 1 = Yes
- **Smoking**: 0 = No, 1 = Yes
- **Obesity**: 0 = No, 1 = Yes
- **Diabetes**: 0 = No, 1 = Yes

#### Lab Results
- **TSH_Level**: Thyroid Stimulating Hormone (0-10 mIU/L typical)
- **T3_Level**: Triiodothyronine (0.5-3.5 ng/mL typical)
- **T4_Level**: Thyroxine (4-12 μg/dL typical)

#### Clinical Findings
- **Nodule_Size**: Size in cm (0-5 typical)

### Step 3: Submit
Click "Predict Risk" button

### Step 4: View Results
- **Prediction**: Benign or Malignant
- **Confidence**: Model confidence percentage
- **Probability Breakdown**: Individual class probabilities
- **Model Used**: Which ML model made the prediction

---

## API Usage

### Python Example

```python
import requests

# Numerical Prediction
url = "http://localhost:5000/predict-numerical"
data = {
    "features": {
        "Age": 45,
        "Gender": 1,
        "Family_History": 0,
        "Radiation_Exposure": 0,
        "Iodine_Deficiency": 0,
        "Smoking": 0,
        "Obesity": 0,
        "Diabetes": 0,
        "TSH_Level": 2.5,
        "T3_Level": 1.8,
        "T4_Level": 8.5,
        "Nodule_Size": 2.3
    }
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
```

### JavaScript Example

```javascript
// Numerical Prediction
const url = 'http://localhost:5000/predict-numerical';
const data = {
    features: {
        Age: 45,
        Gender: 1,
        TSH_Level: 2.5,
        T3_Level: 1.8,
        T4_Level: 8.5,
        Nodule_Size: 2.3,
        // ... other features
    }
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log('Prediction:', result.prediction);
    console.log('Confidence:', result.confidence + '%');
});
```

### cURL Example

```bash
# Numerical Prediction
curl -X POST http://localhost:5000/predict-numerical \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Age": 45,
      "Gender": 1,
      "TSH_Level": 2.5,
      "T3_Level": 1.8,
      "T4_Level": 8.5,
      "Nodule_Size": 2.3
    }
  }'

# Image Prediction
curl -X POST http://localhost:5000/predict-image \
  -F "image=@path/to/image.jpg"
```

---

## Understanding Results

### Prediction Classes

#### Benign (Green)
- Non-cancerous
- Low risk
- May still require monitoring

#### Malignant (Red)
- Potentially cancerous
- High risk
- Requires immediate medical attention

### Confidence Score
- **90-100%**: Very high confidence
- **80-89%**: High confidence
- **70-79%**: Moderate confidence
- **Below 70%**: Lower confidence, consider additional tests

### Probability Bars
- Shows likelihood for each class
- Both probabilities sum to 100%
- Longer bar = higher probability

### Important Notes
⚠️ **This system is for research and educational purposes only**
- Not a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- Use as a supplementary decision support tool

---

## Troubleshooting

### Issue: "Server is not running"
**Solution**: 
```bash
python app.py
```

### Issue: "Models not found"
**Solution**: Train models first
```bash
python train_models.py
```

### Issue: Image upload fails
**Possible causes**:
- File too large (max 16MB)
- Invalid file format
- Corrupted image file

**Solution**: 
- Compress image
- Convert to supported format (PNG/JPG)
- Try different image

### Issue: Numerical prediction returns error
**Possible causes**:
- Missing required fields
- Invalid data types
- Out of range values

**Solution**:
- Fill all fields
- Use numeric values only
- Check value ranges

### Issue: Low confidence scores
**Possible causes**:
- Ambiguous input data
- Edge case scenario
- Model uncertainty

**Solution**:
- Verify input data accuracy
- Consider additional tests
- Consult medical professional

### Issue: Slow predictions
**Possible causes**:
- Large image file
- CPU-only processing
- Limited system resources

**Solution**:
- Resize images before upload
- Use smaller batch sizes
- Upgrade hardware

---

## Testing the System

Run the test suite:
```bash
python test_system.py
```

This will test:
- Server connectivity
- Feature endpoint
- Numerical prediction
- Image prediction
- Error handling

---

## Best Practices

### For Image Predictions
1. Use clear, high-quality images
2. Ensure proper lighting
3. Avoid blurry or distorted images
4. Use standard medical imaging formats

### For Numerical Predictions
1. Double-check all input values
2. Use accurate lab results
3. Ensure units are correct
4. Fill all required fields

### General
1. Keep the system updated
2. Retrain models with new data periodically
3. Monitor prediction accuracy
4. Log predictions for audit trail
5. Always verify with medical professionals

---

## Feature Importance

The model considers these factors (in order of importance):
1. **Nodule Size**: Larger nodules = higher risk
2. **TSH Level**: Abnormal levels indicate thyroid dysfunction
3. **Age**: Risk increases with age
4. **Family History**: Genetic predisposition
5. **Radiation Exposure**: Known risk factor
6. **T3/T4 Levels**: Hormone imbalances
7. Other clinical factors

---

## Batch Processing

For processing multiple predictions:

```python
import pandas as pd
import requests

# Load data
df = pd.read_csv('patients.csv')

# Process each row
results = []
for _, row in df.iterrows():
    response = requests.post(
        'http://localhost:5000/predict-numerical',
        json={'features': row.to_dict()}
    )
    results.append(response.json())

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('predictions.csv', index=False)
```

---

## Integration Examples

### Flask Integration
```python
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/check-patient', methods=['POST'])
def check_patient():
    patient_data = request.json
    
    # Call prediction API
    response = requests.post(
        'http://localhost:5000/predict-numerical',
        json={'features': patient_data}
    )
    
    return response.json()
```

### Django Integration
```python
import requests
from django.http import JsonResponse

def predict_view(request):
    if request.method == 'POST':
        patient_data = request.POST.dict()
        
        response = requests.post(
            'http://localhost:5000/predict-numerical',
            json={'features': patient_data}
        )
        
        return JsonResponse(response.json())
```

---

## Performance Tips

1. **Cache Models**: Models are loaded once at startup
2. **Batch Requests**: Process multiple predictions together
3. **Optimize Images**: Resize before upload
4. **Use CDN**: For static assets in production
5. **Enable Compression**: Gzip responses

---

## Security Considerations

1. **Input Validation**: All inputs are validated
2. **File Size Limits**: 16MB maximum
3. **File Type Checking**: Only allowed formats accepted
4. **Error Handling**: Graceful error messages
5. **No Data Storage**: Predictions not stored by default

---

## Support & Feedback

For issues or questions:
1. Check this guide first
2. Review error messages
3. Check application logs
4. Run test suite
5. Open GitHub issue

---

**Happy Predicting! 🏥**

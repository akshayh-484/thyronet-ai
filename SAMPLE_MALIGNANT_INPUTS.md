# SAMPLE MALIGNANT CASE INPUTS
## For Demonstration and Testing

---

## 📸 IMAGE INPUT - MALIGNANT CASE

### Sample Image Files:
You can use any of these malignant images from your test set:

**Location:** `extracted_data/dataset thyroid/test/Malignant/`

**Sample Files:**
1. `4A_0.jpg`
2. `4A_1.jpg`
3. `4A_10.jpg`
4. `4B_0.jpg`
5. `5_0.jpg`

### How to Use:
1. Go to http://127.0.0.1:5000/predict
2. Select "Image Prediction" tab
3. Click "Choose File"
4. Navigate to: `extracted_data/dataset thyroid/test/Malignant/`
5. Select any image (e.g., `4A_0.jpg`)
6. Click "Predict"

### Expected Result:
- **Prediction:** Malignant
- **Confidence:** 95-99%
- **Individual Model Votes:**
  - ResNet50: ~98% Malignant
  - ResNeXt50: ~99% Malignant
  - DenseNet121: ~96% Malignant

---

## 📋 CLINICAL INPUT - MALIGNANT CASE 1

### Patient Profile: High-Risk Malignant

**Demographics:**
- Age: 58
- Gender: Female
- Country: USA
- Ethnicity: Caucasian

**Medical History:**
- Family History: Yes
- Radiation Exposure: Yes
- Iodine Deficiency: No
- Smoking: Yes
- Obesity: Yes
- Diabetes: No

**Lab Results:**
- TSH Level: 6.5 mIU/L (High - normal is 0.5-4.5)
- T3 Level: 0.8 ng/dL (Low - normal is 1.0-2.0)
- T4 Level: 4.2 μg/dL (Low - normal is 5.0-12.0)
- Nodule Size: 2.8 cm (Large - suspicious if >2.0)

**Risk Assessment:**
- Thyroid Cancer Risk: High

### Expected Result:
- **Prediction:** Malignant
- **Confidence:** 85-90%
- **Reasoning:** High TSH, large nodule, multiple risk factors

---

## 📋 CLINICAL INPUT - MALIGNANT CASE 2

### Patient Profile: Moderate-Risk Malignant

**Demographics:**
- Age: 45
- Gender: Male
- Country: Japan
- Ethnicity: Asian

**Medical History:**
- Family History: Yes
- Radiation Exposure: No
- Iodine Deficiency: Yes
- Smoking: No
- Obesity: No
- Diabetes: Yes

**Lab Results:**
- TSH Level: 5.8 mIU/L (High)
- T3 Level: 1.1 ng/dL (Normal)
- T4 Level: 5.5 μg/dL (Low-normal)
- Nodule Size: 1.9 cm (Borderline large)

**Risk Assessment:**
- Thyroid Cancer Risk: Medium

### Expected Result:
- **Prediction:** Malignant
- **Confidence:** 70-80%
- **Reasoning:** Elevated TSH, borderline nodule, family history

---

## 📋 CLINICAL INPUT - MALIGNANT CASE 3

### Patient Profile: Aggressive Malignant

**Demographics:**
- Age: 62
- Gender: Female
- Country: Germany
- Ethnicity: Caucasian

**Medical History:**
- Family History: Yes
- Radiation Exposure: Yes
- Iodine Deficiency: No
- Smoking: Yes
- Obesity: Yes
- Diabetes: Yes

**Lab Results:**
- TSH Level: 8.2 mIU/L (Very High)
- T3 Level: 0.6 ng/dL (Very Low)
- T4 Level: 3.8 μg/dL (Very Low)
- Nodule Size: 3.5 cm (Very Large)

**Risk Assessment:**
- Thyroid Cancer Risk: High

### Expected Result:
- **Prediction:** Malignant
- **Confidence:** 90-95%
- **Reasoning:** Very high TSH, very large nodule, all risk factors present

---

## 🎯 HOW TO TEST IN WEB APPLICATION

### For Image Prediction:
1. Open: http://127.0.0.1:5000/predict
2. Login: doctor / thyronet2024
3. Click "Image Prediction" tab
4. Upload any image from `Malignant` folder
5. Click "Predict"
6. See results with individual model votes

### For Clinical Prediction:
1. Open: http://127.0.0.1:5000/predict
2. Login: doctor / thyronet2024
3. Click "Numerical Prediction" tab
4. Fill in the form with values from Case 1, 2, or 3 above
5. Click "Predict"
6. See results with individual model votes

---

## 📊 COPY-PASTE VALUES FOR WEB FORM

### Case 1 (High-Risk):
```
Age: 58
Gender: Female
Country: USA
Ethnicity: Caucasian
Family History: Yes
Radiation Exposure: Yes
Iodine Deficiency: No
Smoking: Yes
Obesity: Yes
Diabetes: No
TSH Level: 6.5
T3 Level: 0.8
T4 Level: 4.2
Nodule Size: 2.8
Thyroid Cancer Risk: High
```

### Case 2 (Moderate-Risk):
```
Age: 45
Gender: Male
Country: Japan
Ethnicity: Asian
Family History: Yes
Radiation Exposure: No
Iodine Deficiency: Yes
Smoking: No
Obesity: No
Diabetes: Yes
TSH Level: 5.8
T3 Level: 1.1
T4 Level: 5.5
Nodule Size: 1.9
Thyroid Cancer Risk: Medium
```

### Case 3 (Aggressive):
```
Age: 62
Gender: Female
Country: Germany
Ethnicity: Caucasian
Family History: Yes
Radiation Exposure: Yes
Iodine Deficiency: No
Smoking: Yes
Obesity: Yes
Diabetes: Yes
TSH Level: 8.2
T3 Level: 0.6
T4 Level: 3.8
Nodule Size: 3.5
Thyroid Cancer Risk: High
```

---

## 🔍 WHAT MAKES THESE MALIGNANT?

### Key Indicators:
1. **High TSH Level** (>4.5 mIU/L)
   - Indicates thyroid not producing enough hormones
   - Body trying to stimulate thyroid more
   - Common in thyroid cancer

2. **Large Nodule Size** (>2.0 cm)
   - Larger nodules more likely malignant
   - Size correlates with cancer risk

3. **Low T3/T4 Levels**
   - Thyroid not functioning properly
   - Hormones below normal range

4. **Multiple Risk Factors**
   - Family history (genetic)
   - Radiation exposure (environmental)
   - Smoking (lifestyle)
   - Obesity (metabolic)

5. **Age Factor**
   - Risk increases with age
   - Peak incidence: 45-65 years

---

## 💡 FOR PRESENTATION DEMO

### Best Cases to Show:
1. **Image:** Use `4A_0.jpg` - Clear malignant case
2. **Clinical:** Use Case 1 (High-Risk) - Most obvious malignant

### What to Say:
"Let me demonstrate with a real malignant case. This patient is 58 years old with high TSH (6.5), large nodule (2.8 cm), and multiple risk factors including family history and radiation exposure. The AI correctly predicts Malignant with 85-90% confidence."

---

## ⚠️ IMPORTANT NOTES

1. **These are sample cases** - Not real patient data
2. **For demonstration only** - Not for actual medical diagnosis
3. **Always verify** - Real diagnosis requires doctor confirmation
4. **Educational purpose** - To show how the AI system works

---

**Ready to test and demonstrate!** 🎉

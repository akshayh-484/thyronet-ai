# 📊 TEST DATA FOR NUMERICAL PREDICTION

## 🎯 How to Test

1. Go to **http://localhost:5000/predict**
2. Stay on **"Numerical Features"** tab
3. Select prediction method: **"🏆 Ensemble - Best Method"**
4. Enter values from below
5. Click **"Predict Risk"**

---

## ✅ SAMPLE 1: BENIGN CASE (Should predict: Benign)

```
Age: 66
Gender: Male
Country: Russia
Ethnicity: Caucasian
Family_History: No
Radiation_Exposure: Yes
Iodine_Deficiency: No
Smoking: No
Obesity: No
Diabetes: No
TSH_Level: 9.37
T3_Level: 1.67
T4_Level: 6.16
Nodule_Size: 1.08
Thyroid_Cancer_Risk: Low
```

**Expected Result**: Benign with high confidence

---

## ⚠️ SAMPLE 2: MALIGNANT CASE (Should predict: Malignant)

```
Age: 89
Gender: Female
Country: South Korea
Ethnicity: Asian
Family_History: Yes
Radiation_Exposure: Yes
Iodine_Deficiency: No
Smoking: No
Obesity: Yes
Diabetes: No
TSH_Level: 4.7
T3_Level: 0.62
T4_Level: 11.73
Nodule_Size: 0.01
Thyroid_Cancer_Risk: High
```

**Expected Result**: Malignant with high confidence

---

## 🧪 SIMPLE TEST CASE (Easy to remember)

```
Age: 45
Gender: Male (or 1)
Country: USA
Ethnicity: Caucasian
Family_History: No
Radiation_Exposure: No
Iodine_Deficiency: No
Smoking: No
Obesity: No
Diabetes: No
TSH_Level: 2.5
T3_Level: 1.5
T4_Level: 8.0
Nodule_Size: 0.5
Thyroid_Cancer_Risk: Low
```

---

## 📝 Field Explanations

| Field | Type | Values |
|-------|------|--------|
| Age | Number | 0-100 |
| Gender | Text | Male/Female |
| Country | Text | Any country name |
| Ethnicity | Text | Caucasian/Asian/African/Hispanic/Other |
| Family_History | Text | Yes/No |
| Radiation_Exposure | Text | Yes/No |
| Iodine_Deficiency | Text | Yes/No |
| Smoking | Text | Yes/No |
| Obesity | Text | Yes/No |
| Diabetes | Text | Yes/No |
| TSH_Level | Number | 0-20 (typical: 0.5-5.0) |
| T3_Level | Number | 0-5 (typical: 0.8-2.0) |
| T4_Level | Number | 0-20 (typical: 5-12) |
| Nodule_Size | Number | 0-10 cm |
| Thyroid_Cancer_Risk | Text | Low/Medium/High |

---

## 🎨 Prediction Methods to Try

1. **Single Best Model** - Random Forest (77.95%)
2. **🏆 Ensemble - Best Method** - Automatically best (RECOMMENDED)
3. **Ensemble - Voting** - Average of all models
4. **Ensemble - Stacking** - Meta-learner (77.75%)
5. **Ensemble - Weighted** - Performance-weighted
6. **Ensemble - All Methods** - Shows all results

---

## 💡 Tips

- **Try different methods** to see how predictions vary
- **Use "All Methods"** to compare all ensemble approaches
- **Green result** = Benign (low risk)
- **Red result** = Malignant (high risk)
- **Confidence score** shows how certain the model is

---

## 🔍 What to Look For

### Good Prediction:
- High confidence (>70%)
- Correct classification
- Consistent across methods

### Uncertain Prediction:
- Low confidence (<60%)
- Close probabilities (e.g., 52% vs 48%)
- Different methods disagree

---

## 📊 Quick Copy-Paste Values

### Benign Test:
```
66, Male, Russia, Caucasian, No, Yes, No, No, No, No, 9.37, 1.67, 6.16, 1.08, Low
```

### Malignant Test:
```
89, Female, South Korea, Asian, Yes, Yes, No, No, Yes, No, 4.7, 0.62, 11.73, 0.01, High
```

---

## 🎯 Expected Results

| Sample | Expected | Confidence | Method |
|--------|----------|------------|--------|
| Sample 1 | Benign | ~70-80% | Any |
| Sample 2 | Malignant | ~70-80% | Any |
| Simple Test | Benign | ~60-70% | Any |

---

**Now go test your predictions!** 🚀

Open: **http://localhost:5000/predict**

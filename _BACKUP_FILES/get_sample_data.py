import pandas as pd
import json

# Load data
df = pd.read_csv('thyroid_cancer_risk_data.csv')

# Get samples
benign_sample = df[df['Diagnosis'] == 'Benign'].iloc[0].to_dict()
malignant_sample = df[df['Diagnosis'] == 'Malignant'].iloc[0].to_dict()

# Remove Patient_ID and Diagnosis
benign_sample.pop('Patient_ID', None)
benign_sample.pop('Diagnosis', None)
malignant_sample.pop('Patient_ID', None)
malignant_sample.pop('Diagnosis', None)

print("="*80)
print("SAMPLE DATA FOR TESTING NUMERICAL PREDICTION")
print("="*80)

print("\n📊 SAMPLE 1: BENIGN CASE")
print("-"*80)
for key, value in benign_sample.items():
    print(f"{key}: {value}")

print("\n📊 SAMPLE 2: MALIGNANT CASE")
print("-"*80)
for key, value in malignant_sample.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("COPY-PASTE READY FORMAT")
print("="*80)

print("\n✅ BENIGN SAMPLE (Copy these values):")
print(json.dumps(benign_sample, indent=2))

print("\n⚠️ MALIGNANT SAMPLE (Copy these values):")
print(json.dumps(malignant_sample, indent=2))

# Save to file
with open('SAMPLE_TEST_DATA.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SAMPLE DATA FOR TESTING NUMERICAL PREDICTION\n")
    f.write("="*80 + "\n\n")
    
    f.write("SAMPLE 1: BENIGN CASE\n")
    f.write("-"*80 + "\n")
    for key, value in benign_sample.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\nSAMPLE 2: MALIGNANT CASE\n")
    f.write("-"*80 + "\n")
    for key, value in malignant_sample.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("HOW TO USE:\n")
    f.write("="*80 + "\n")
    f.write("1. Go to http://localhost:5000/predict\n")
    f.write("2. Stay on 'Numerical Features' tab\n")
    f.write("3. Copy values from above into the form\n")
    f.write("4. Select prediction method\n")
    f.write("5. Click 'Predict Risk'\n")

print("\n✅ Sample data saved to: SAMPLE_TEST_DATA.txt")

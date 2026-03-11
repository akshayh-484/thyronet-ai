"""
Test the prediction with sample data
"""

from utils.numerical_predictor import NumericalPredictor

# Initialize predictor
predictor = NumericalPredictor()

# Test case from user: Should predict "Benign"
test_data = {
    'Age': 66,
    'Gender': 'Male',
    'Country': 'Russia',
    'Ethnicity': 'Caucasian',
    'Family_History': 'No',
    'Radiation_Exposure': 'Yes',
    'Iodine_Deficiency': 'No',
    'Smoking': 'No',
    'Obesity': 'No',
    'Diabetes': 'No',
    'TSH_Level': 9.37,
    'T3_Level': 1.67,
    'T4_Level': 6.16,
    'Nodule_Size': 1.08,
    'Thyroid_Cancer_Risk': 'Low'
}

print("="*80)
print("🧪 TESTING PREDICTION")
print("="*80)
print("\nInput Data:")
for key, value in test_data.items():
    print(f"  {key}: {value}")

print("\n" + "="*80)
print("PREDICTION RESULT:")
print("="*80)

result = predictor.predict(test_data)

if result['success']:
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']}%")
    print(f"🤖 Model: {result['model_used']}")
    print(f"\nProbabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob}%")
    
    print("\n" + "="*80)
    if result['prediction'] == 'Benign':
        print("✅ CORRECT! Predicted Benign as expected")
    else:
        print("⚠️  Predicted Malignant (Expected: Benign)")
        print("   Note: Model accuracy is 78.90%, not 100%")
    print("="*80)
else:
    print(f"\n❌ Error: {result['error']}")

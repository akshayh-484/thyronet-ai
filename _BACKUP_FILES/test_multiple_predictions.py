"""
Test multiple predictions to evaluate model performance
"""

from utils.numerical_predictor import NumericalPredictor
import pandas as pd

# Initialize predictor
predictor = NumericalPredictor()

print("="*80)
print("🧪 TESTING MULTIPLE PREDICTIONS")
print("="*80)

# Test cases
test_cases = [
    {
        'name': 'Test 1 - Expected: Benign (Low risk profile)',
        'expected': 'Benign',
        'data': {
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
    },
    {
        'name': 'Test 2 - Expected: Malignant (High risk profile)',
        'expected': 'Malignant',
        'data': {
            'Age': 55,
            'Gender': 'Female',
            'Country': 'USA',
            'Ethnicity': 'Caucasian',
            'Family_History': 'Yes',
            'Radiation_Exposure': 'Yes',
            'Iodine_Deficiency': 'Yes',
            'Smoking': 'Yes',
            'Obesity': 'Yes',
            'Diabetes': 'Yes',
            'TSH_Level': 12.5,
            'T3_Level': 0.8,
            'T4_Level': 4.2,
            'Nodule_Size': 2.5,
            'Thyroid_Cancer_Risk': 'High'
        }
    },
    {
        'name': 'Test 3 - Expected: Benign (Young, healthy)',
        'expected': 'Benign',
        'data': {
            'Age': 25,
            'Gender': 'Female',
            'Country': 'USA',
            'Ethnicity': 'Asian',
            'Family_History': 'No',
            'Radiation_Exposure': 'No',
            'Iodine_Deficiency': 'No',
            'Smoking': 'No',
            'Obesity': 'No',
            'Diabetes': 'No',
            'TSH_Level': 2.5,
            'T3_Level': 1.5,
            'T4_Level': 7.0,
            'Nodule_Size': 0.3,
            'Thyroid_Cancer_Risk': 'Low'
        }
    },
    {
        'name': 'Test 4 - Expected: Malignant (Large nodule)',
        'expected': 'Malignant',
        'data': {
            'Age': 60,
            'Gender': 'Male',
            'Country': 'Japan',
            'Ethnicity': 'Asian',
            'Family_History': 'Yes',
            'Radiation_Exposure': 'Yes',
            'Iodine_Deficiency': 'No',
            'Smoking': 'No',
            'Obesity': 'No',
            'Diabetes': 'No',
            'TSH_Level': 8.5,
            'T3_Level': 1.2,
            'T4_Level': 5.5,
            'Nodule_Size': 3.0,
            'Thyroid_Cancer_Risk': 'High'
        }
    },
    {
        'name': 'Test 5 - Expected: Benign (Normal hormones)',
        'expected': 'Benign',
        'data': {
            'Age': 40,
            'Gender': 'Female',
            'Country': 'UK',
            'Ethnicity': 'Caucasian',
            'Family_History': 'No',
            'Radiation_Exposure': 'No',
            'Iodine_Deficiency': 'No',
            'Smoking': 'No',
            'Obesity': 'No',
            'Diabetes': 'No',
            'TSH_Level': 1.5,
            'T3_Level': 1.8,
            'T4_Level': 8.0,
            'Nodule_Size': 0.5,
            'Thyroid_Cancer_Risk': 'Low'
        }
    }
]

# Run tests
correct = 0
total = len(test_cases)

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"{test['name']}")
    print(f"{'='*80}")
    
    result = predictor.predict(test['data'])
    
    if result['success']:
        prediction = result['prediction']
        confidence = result['confidence']
        
        is_correct = prediction == test['expected']
        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"
        
        print(f"\n{status}")
        print(f"Expected:   {test['expected']}")
        print(f"Predicted:  {prediction}")
        print(f"Confidence: {confidence}%")
        print(f"\nProbabilities:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls}: {prob}%")
    else:
        print(f"\n❌ ERROR: {result['error']}")

# Summary
print(f"\n{'='*80}")
print(f"📊 SUMMARY")
print(f"{'='*80}")
print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"Model Accuracy: 82.06% (on test set)")
print(f"Model: LightGBM")
print(f"{'='*80}")

# Load actual test data and check
print(f"\n{'='*80}")
print(f"📈 CHECKING AGAINST ACTUAL TEST DATA")
print(f"{'='*80}")

try:
    df = pd.read_csv('thyroid_cancer_risk_data.csv')
    print(f"Total dataset: {len(df):,} samples")
    print(f"\nClass distribution:")
    print(df['Diagnosis'].value_counts())
    print(f"\nPercentages:")
    print(df['Diagnosis'].value_counts(normalize=True) * 100)
except Exception as e:
    print(f"Could not load dataset: {e}")

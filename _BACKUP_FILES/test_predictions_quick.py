"""Quick test of both predictors"""

print("="*50)
print("Testing ThyroNet AI Predictors")
print("="*50)

# Test 1: Numerical Predictor
print("\n1. Testing Numerical Predictor...")
try:
    from utils.numerical_predictor import NumericalPredictor
    predictor = NumericalPredictor('models')
    print(f"   ✓ Model loaded: {type(predictor.model).__name__}")
    print(f"   ✓ Features: {len(predictor.feature_names)}")
    
    # Test prediction
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
    
    result = predictor.predict(test_data)
    if result['success']:
        print(f"   ✓ Prediction: {result['prediction']}")
        print(f"   ✓ Confidence: {result['confidence']}%")
        print(f"   ✓ Benign: {result['probabilities']['Benign']}%")
        print(f"   ✓ Malignant: {result['probabilities']['Malignant']}%")
        print("   ✓ NUMERICAL PREDICTION WORKS!")
    else:
        print(f"   ✗ Error: {result['error']}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Image Predictor
print("\n2. Testing Image Predictor...")
try:
    from utils.ensemble_image_predictor import EnsembleImagePredictor
    predictor = EnsembleImagePredictor()
    print(f"   ✓ Ensemble loaded")
    print(f"   ✓ Weights: ResNet({predictor.weights['resnet']}), ResNeXt({predictor.weights['resnext']}), DenseNet({predictor.weights['densenet']})")
    print("   ✓ IMAGE PREDICTOR WORKS!")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n" + "="*50)
print("Test Complete!")
print("="*50)

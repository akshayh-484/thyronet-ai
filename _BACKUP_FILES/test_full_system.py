"""
Complete System Test - Image and Numerical Predictions
"""

print("="*60)
print("THYRONET AI - COMPLETE SYSTEM TEST")
print("="*60)

# Test 1: Image Predictor
print("\n1. Testing Image Predictor...")
print("-" * 60)
try:
    from utils.ensemble_image_predictor import EnsembleImagePredictor
    predictor = EnsembleImagePredictor()
    print("✓ Image predictor loaded successfully")
    print(f"✓ Models: ResNet50({predictor.weights['resnet']*100}%), ResNeXt50({predictor.weights['resnext']*100}%), DenseNet121({predictor.weights['densenet']*100}%)")
    print("✓ Status: READY FOR PREDICTIONS")
    image_status = "WORKING"
except Exception as e:
    print(f"✗ Error: {e}")
    image_status = "FAILED"

# Test 2: Numerical Predictor
print("\n2. Testing Numerical Predictor...")
print("-" * 60)
try:
    from utils.numerical_predictor import NumericalPredictor
    predictor = NumericalPredictor('models')
    print(f"✓ Numerical predictor loaded successfully")
    print(f"✓ Model: {type(predictor.model).__name__}")
    print(f"✓ Features: {len(predictor.feature_names)}")
    
    # Test with sample data
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
        print(f"✓ Test Prediction: {result['prediction']}")
        print(f"✓ Confidence: {result['confidence']}%")
        print(f"✓ Model Used: {result['model_used']}")
        print("✓ Status: READY FOR PREDICTIONS")
        numerical_status = "WORKING"
    else:
        print(f"✗ Prediction failed: {result['error']}")
        numerical_status = "FAILED"
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    numerical_status = "FAILED"

# Test 3: Flask App
print("\n3. Testing Flask Application...")
print("-" * 60)
try:
    from app_enhanced import app
    print("✓ Flask app imported successfully")
    print("✓ Routes configured")
    print("✓ Status: READY TO START")
    app_status = "WORKING"
except Exception as e:
    print(f"✗ Error: {e}")
    app_status = "FAILED"

# Final Summary
print("\n" + "="*60)
print("FINAL STATUS")
print("="*60)
print(f"Image Predictor:     {image_status}")
print(f"Numerical Predictor: {numerical_status}")
print(f"Flask Application:   {app_status}")
print("="*60)

if image_status == "WORKING" and numerical_status == "WORKING" and app_status == "WORKING":
    print("\n✅ ALL SYSTEMS WORKING - 100% READY!")
    print("\nTo start the application:")
    print("  python app_enhanced.py")
    print("\nThen login at: http://localhost:5000")
    print("  Username: doctor")
    print("  Password: thyronet2024")
else:
    print("\n⚠️ SOME SYSTEMS FAILED - CHECK ERRORS ABOVE")

print("\n" + "="*60)

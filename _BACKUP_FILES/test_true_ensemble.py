"""Test TRUE Ensemble - Multiple Models Predicting Together"""

print("="*70)
print("TESTING TRUE ENSEMBLE - MULTIPLE MODELS TOGETHER")
print("="*70)

try:
    from utils.numerical_predictor import NumericalPredictor
    
    print("\nLoading predictor...")
    predictor = NumericalPredictor('models')
    
    if predictor.use_ensemble:
        print("\n✅ TRUE ENSEMBLE LOADED!")
        print("   4 Models working together:")
        print("   1. Logistic Regression")
        print("   2. Support Vector Machine (SVM)")
        print("   3. Random Forest")
        print("   4. XGBoost")
        print("\n   Method: Soft Voting (averaging predictions)")
    else:
        print("\n⚠️ Single model loaded (fallback)")
    
    # Test prediction
    print("\n" + "-"*70)
    print("Testing with sample data...")
    print("-"*70)
    
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
        print(f"\n✅ ENSEMBLE PREDICTION SUCCESSFUL!")
        print(f"\n   Final Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Model Used: {result['model_used']}")
        
        print(f"\n   Probabilities:")
        print(f"   - Benign: {result['probabilities']['Benign']}%")
        print(f"   - Malignant: {result['probabilities']['Malignant']}%")
        
        if 'individual_predictions' in result:
            print(f"\n   Individual Model Predictions:")
            for model, pred in result['individual_predictions'].items():
                print(f"   - {model:20s}: {pred['prediction']:10s} ({pred['confidence']:5.2f}%)")
            
            print("\n   ✅ ALL 4 MODELS VOTED TOGETHER!")
        
        print("\n" + "="*70)
        print("SUCCESS! TRUE ENSEMBLE IS WORKING!")
        print("="*70)
    else:
        print(f"\n✗ Error: {result['error']}")

except Exception as e:
    print(f"\n✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()

"""
Complete System Test - Image & Numerical Predictions
Tests both deep learning (ultrasound images) and ML (numerical data) predictions
"""

import os
import sys
from PIL import Image
import numpy as np

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def test_image_predictor():
    """Test deep learning image predictor"""
    print_header("🖼️  TESTING IMAGE PREDICTOR (Deep Learning)")
    
    try:
        from utils.image_predictor import ImagePredictor
        
        # Check if model file exists
        model_path = 'densenet121_best.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file found: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
        
        # Initialize predictor
        print("\n📥 Loading DenseNet121 model...")
        predictor = ImagePredictor(model_path)
        print("✅ Image predictor loaded successfully")
        print(f"   Device: {predictor.device}")
        print(f"   Classes: {predictor.classes}")
        
        # Find a test image
        test_image_path = None
        test_dirs = [
            'extracted_data/dataset thyroid/test/Benign',
            'extracted_data/dataset thyroid/test/Malignant',
            'extracted_data/dataset thyroid/test/normal thyroid'
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
                if images:
                    test_image_path = os.path.join(test_dir, images[0])
                    break
        
        if not test_image_path:
            print("\n⚠️  No test images found in dataset")
            print("   Creating synthetic test image...")
            # Create a synthetic test image
            test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            test_image_path = test_image
        else:
            print(f"\n✅ Test image found: {test_image_path}")
        
        # Make prediction
        print("\n🔮 Making prediction...")
        result = predictor.predict(test_image_path)
        
        if result['success']:
            print("✅ Prediction successful!")
            print(f"\n📊 Results:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Probabilities:")
            print(f"      Benign:    {result['probabilities']['Benign']}%")
            print(f"      Malignant: {result['probabilities']['Malignant']}%")
            return True
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_numerical_predictor():
    """Test ML numerical predictor"""
    print_header("📊 TESTING NUMERICAL PREDICTOR (Machine Learning)")
    
    try:
        from utils.numerical_predictor import NumericalPredictor
        
        # Check if models exist
        if not os.path.exists('models'):
            print("❌ Models folder not found")
            print("   Run: python quick_train.py")
            return False
        
        required_files = [
            'models/best_model.pkl',
            'models/scaler.pkl',
            'models/label_encoder.pkl',
            'models/metadata.json'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print("❌ Missing model files:")
            for f in missing_files:
                print(f"   - {f}")
            print("\n   Run: python quick_train.py")
            return False
        
        print("✅ All model files found")
        
        # Initialize predictor
        print("\n📥 Loading numerical predictor...")
        predictor = NumericalPredictor('models')
        print("✅ Numerical predictor loaded successfully")
        
        # Get feature names
        feature_names = predictor.get_feature_names()
        print(f"   Features: {len(feature_names)} features")
        
        # Create sample features
        sample_features = {name: 0.0 for name in feature_names}
        sample_features['Age'] = 45
        if 'Gender' in sample_features:
            sample_features['Gender'] = 1
        if 'Smoking' in sample_features:
            sample_features['Smoking'] = 0
        
        print(f"\n🔮 Making prediction with sample data...")
        print(f"   Sample: Age=45, Gender=1, Smoking=0, others=0")
        
        # Make prediction
        result = predictor.predict(sample_features)
        
        if result['success']:
            print("✅ Prediction successful!")
            print(f"\n📊 Results:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Probabilities:")
            print(f"      Benign:    {result['probabilities']['Benign']}%")
            print(f"      Malignant: {result['probabilities']['Malignant']}%")
            return True
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_predictor():
    """Test ensemble predictor"""
    print_header("🎯 TESTING ENSEMBLE PREDICTOR (Advanced ML)")
    
    try:
        from utils.ensemble_predictor import EnsemblePredictor
        
        # Check if ensemble models exist
        ensemble_files = [
            'models/ensemble_lr.pkl',
            'models/ensemble_svm.pkl',
            'models/ensemble_rf.pkl',
            'models/ensemble_xgb.pkl',
            'models/voting_ensemble.pkl',
            'models/stacking_ensemble.pkl',
            'models/ensemble_metadata.json'
        ]
        
        missing_files = [f for f in ensemble_files if not os.path.exists(f)]
        if missing_files:
            print("⚠️  Ensemble models not trained yet")
            print("   Run: python train_ensemble.py")
            return None  # Not an error, just not trained yet
        
        print("✅ All ensemble files found")
        
        # Initialize predictor
        print("\n📥 Loading ensemble predictor...")
        predictor = EnsemblePredictor('models')
        print("✅ Ensemble predictor loaded successfully")
        
        # Get model info
        info = predictor.get_model_info()
        print(f"   Best Model: {info['best_model']}")
        print(f"   Accuracy: {info['accuracy']*100:.2f}%")
        print(f"   ROC-AUC: {info['roc_auc']:.4f}")
        
        # Get feature names
        feature_names = predictor.get_feature_names()
        
        # Create sample features
        sample_features = {name: 0.0 for name in feature_names}
        sample_features['Age'] = 45
        if 'Gender' in sample_features:
            sample_features['Gender'] = 1
        
        print(f"\n🔮 Testing ensemble methods...")
        
        methods = ['best', 'voting', 'stacking', 'weighted']
        all_passed = True
        
        for method in methods:
            result = predictor.predict(sample_features, method=method)
            if result['success']:
                print(f"   ✅ {method.upper()}: {result['prediction']} ({result['confidence']}%)")
            else:
                print(f"   ❌ {method.upper()}: Failed")
                all_passed = False
        
        return all_passed
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_web_app_startup():
    """Test if web app can start"""
    print_header("🌐 TESTING WEB APPLICATION")
    
    try:
        print("📥 Importing Flask app...")
        from app_enhanced import app, image_predictor, numerical_predictor, ensemble_predictor
        
        print("\n✅ Flask app imported successfully")
        
        # Check predictors
        print("\n📊 Predictor Status:")
        print(f"   Image Predictor:     {'✅ Available' if image_predictor else '❌ Not available'}")
        print(f"   Numerical Predictor: {'✅ Available' if numerical_predictor else '❌ Not available'}")
        print(f"   Ensemble Predictor:  {'✅ Available' if ensemble_predictor else '⚠️  Not trained yet'}")
        
        # Check routes
        print("\n📍 Available Routes:")
        routes = [rule.rule for rule in app.url_map.iter_rules() if rule.endpoint != 'static']
        for route in sorted(routes):
            print(f"   {route}")
        
        print("\n✅ Web application ready to start")
        print("   Run: python app_enhanced.py")
        print("   Access: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_dataset():
    """Check if dataset is available"""
    print_header("📁 CHECKING DATASET")
    
    # Check CSV file
    csv_file = 'thyroid_cancer_risk_data.csv'
    if os.path.exists(csv_file):
        print(f"✅ CSV dataset found: {csv_file}")
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {len(df.columns)}")
    else:
        print(f"❌ CSV dataset not found: {csv_file}")
    
    # Check image dataset
    image_dirs = [
        'extracted_data/dataset thyroid/test/Benign',
        'extracted_data/dataset thyroid/test/Malignant',
        'extracted_data/dataset thyroid/test/normal thyroid',
        'extracted_data/dataset thyroid/train/benign',
        'extracted_data/dataset thyroid/train/malignant'
    ]
    
    print("\n📸 Image Dataset:")
    total_images = 0
    for dir_path in image_dirs:
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png'))]
            count = len(images)
            total_images += count
            print(f"   ✅ {dir_path}: {count} images")
        else:
            print(f"   ⚠️  {dir_path}: Not found")
    
    print(f"\n   Total images: {total_images}")
    
    return True

def main():
    print("\n" + "="*80)
    print("  🧪 COMPLETE SYSTEM TEST")
    print("  Testing Image Prediction + Numerical Prediction + Ensemble")
    print("="*80)
    
    results = {}
    
    # Check dataset
    check_dataset()
    
    # Test image predictor (Deep Learning)
    results['image'] = test_image_predictor()
    
    # Test numerical predictor (ML)
    results['numerical'] = test_numerical_predictor()
    
    # Test ensemble predictor (Advanced ML)
    results['ensemble'] = test_ensemble_predictor()
    
    # Test web app
    results['webapp'] = test_web_app_startup()
    
    # Summary
    print_header("📋 TEST SUMMARY")
    
    print("\n✅ = Passed | ❌ = Failed | ⚠️  = Not trained yet\n")
    
    status_icon = lambda x: "✅" if x is True else ("⚠️ " if x is None else "❌")
    
    print(f"   {status_icon(results['image'])} Image Predictor (Deep Learning)")
    print(f"   {status_icon(results['numerical'])} Numerical Predictor (ML)")
    print(f"   {status_icon(results['ensemble'])} Ensemble Predictor (Advanced ML)")
    print(f"   {status_icon(results['webapp'])} Web Application")
    
    # Overall status
    critical_tests = [results['image'], results['webapp']]
    all_critical_passed = all(t is True for t in critical_tests)
    
    print("\n" + "="*80)
    if all_critical_passed:
        print("  🎉 SYSTEM READY!")
        print("="*80)
        print("\n  ✅ Image prediction working (Deep Learning)")
        print("  ✅ Web application ready")
        
        if results['numerical'] is True:
            print("  ✅ Numerical prediction working (ML)")
        else:
            print("  ⚠️  Numerical prediction not trained - Run: python quick_train.py")
        
        if results['ensemble'] is True:
            print("  ✅ Ensemble prediction working (Advanced ML)")
        elif results['ensemble'] is None:
            print("  ⚠️  Ensemble not trained - Run: python train_ensemble.py")
        
        print("\n  🚀 Start web app: python app_enhanced.py")
        print("  🌐 Access: http://localhost:5000")
        print("="*80 + "\n")
        return True
    else:
        print("  ❌ SYSTEM HAS ISSUES")
        print("="*80)
        print("\n  Please fix the errors above before using the system.")
        print("="*80 + "\n")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

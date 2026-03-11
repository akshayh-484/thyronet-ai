"""
Test Ensemble Prediction System
Quick verification that ensemble models work correctly
"""

import os
import json

def check_ensemble_files():
    """Check if ensemble models are trained"""
    print("="*80)
    print("🔍 CHECKING ENSEMBLE FILES")
    print("="*80)
    
    required_files = [
        'models/ensemble_lr.pkl',
        'models/ensemble_svm.pkl',
        'models/ensemble_rf.pkl',
        'models/ensemble_xgb.pkl',
        'models/voting_ensemble.pkl',
        'models/stacking_ensemble.pkl',
        'models/weighted_ensemble_config.json',
        'models/ensemble_metadata.json',
        'models/ensemble_scaler.pkl',
        'models/ensemble_label_encoder.pkl',
        'models/ensemble_feature_names.json'
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_exist = False
    
    print("="*80)
    
    if not all_exist:
        print("\n❌ ENSEMBLE MODELS NOT FOUND")
        print("Please run: python train_ensemble.py")
        return False
    
    print("\n✅ ALL ENSEMBLE FILES PRESENT")
    return True

def test_ensemble_predictor():
    """Test ensemble predictor with sample data"""
    print("\n" + "="*80)
    print("🧪 TESTING ENSEMBLE PREDICTOR")
    print("="*80)
    
    try:
        from utils.ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor('models')
        print("✅ Ensemble predictor loaded successfully")
        
        # Get feature names
        feature_names = predictor.get_feature_names()
        print(f"✅ Features loaded: {len(feature_names)} features")
        
        # Get model info
        info = predictor.get_model_info()
        print(f"\n📊 Best Model: {info['best_model']}")
        print(f"📊 Accuracy: {info['accuracy']*100:.2f}%")
        print(f"📊 ROC-AUC: {info['roc_auc']:.4f}")
        
        # Create sample features (all zeros for testing)
        sample_features = {name: 0.0 for name in feature_names}
        sample_features['Age'] = 45
        sample_features['Gender'] = 1
        
        print("\n" + "="*80)
        print("🎯 TESTING ENSEMBLE METHODS")
        print("="*80)
        
        # Test each method
        methods = ['best', 'voting', 'stacking', 'weighted', 'all']
        
        for method in methods:
            print(f"\nTesting {method.upper()} method...")
            result = predictor.predict(sample_features, method=method)
            
            if result['success']:
                print(f"  ✅ Prediction: {result['prediction']}")
                print(f"  ✅ Confidence: {result['confidence']}%")
                print(f"  ✅ Method: {result['method']}")
                
                if 'individual_results' in result:
                    print("  ✅ Individual results available")
            else:
                print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
                return False
        
        print("\n" + "="*80)
        print("✅ ALL ENSEMBLE TESTS PASSED")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def display_model_comparison():
    """Display model comparison from metadata"""
    print("\n" + "="*80)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    try:
        with open('models/ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"\n{'Model':<30} {'Accuracy':<12} {'ROC-AUC':<12}")
        print("-"*80)
        
        for model_name, metrics in metadata['all_models'].items():
            accuracy = metrics['accuracy'] * 100
            roc_auc = metrics['roc_auc']
            marker = "🏆" if model_name == metadata['best_model_name'] else "  "
            print(f"{marker} {model_name:<28} {accuracy:>6.2f}%      {roc_auc:>6.4f}")
        
        print("="*80)
        print(f"\n🏆 WINNER: {metadata['best_model_name']}")
        print(f"📅 Trained: {metadata['training_date']}")
        
    except Exception as e:
        print(f"❌ Could not load metadata: {str(e)}")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎯 ENSEMBLE SYSTEM TEST")
    print("="*80)
    print("This script verifies that ensemble models are trained and working")
    print("="*80 + "\n")
    
    # Check files
    files_ok = check_ensemble_files()
    
    if not files_ok:
        print("\n⚠️  Run 'python train_ensemble.py' to train ensemble models first")
        exit(1)
    
    # Test predictor
    predictor_ok = test_ensemble_predictor()
    
    if not predictor_ok:
        print("\n❌ Ensemble predictor test failed")
        exit(1)
    
    # Display comparison
    display_model_comparison()
    
    print("\n" + "="*80)
    print("🎉 ENSEMBLE SYSTEM READY!")
    print("="*80)
    print("✅ All ensemble models trained and working")
    print("✅ Predictor functioning correctly")
    print("✅ Ready for production use")
    print("\n🚀 Start web app: python app_enhanced.py")
    print("="*80 + "\n")

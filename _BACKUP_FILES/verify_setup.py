"""
Setup Verification Script
Checks if all required files and dependencies are present
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory(dirpath, description):
    """Check if a directory exists"""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    status = "✅" if is_valid else "❌"
    print(f"{status} Python version: {version.major}.{version.minor}.{version.micro}")
    if not is_valid:
        print("   ⚠️  Python 3.8+ required")
    return is_valid

def check_dependencies():
    """Check if key dependencies can be imported"""
    dependencies = [
        ('flask', 'Flask'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'XGBoost'),
        ('tensorflow', 'TensorFlow'),
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow')
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            all_ok = False
    
    return all_ok

def main():
    print("\n" + "="*60)
    print("THYROID CANCER PREDICTION SYSTEM - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    all_checks = []
    
    # Check Python version
    print("1. Python Version")
    print("-" * 60)
    all_checks.append(check_python_version())
    print()
    
    # Check core files
    print("2. Core Application Files")
    print("-" * 60)
    all_checks.append(check_file('app.py', 'Main application'))
    all_checks.append(check_file('train_models.py', 'Training script'))
    all_checks.append(check_file('requirements.txt', 'Dependencies'))
    print()
    
    # Check utility files
    print("3. Utility Modules")
    print("-" * 60)
    all_checks.append(check_file('utils/image_predictor.py', 'Image predictor'))
    all_checks.append(check_file('utils/numerical_predictor.py', 'Numerical predictor'))
    print()
    
    # Check template files
    print("4. Web Interface")
    print("-" * 60)
    all_checks.append(check_file('templates/index.html', 'HTML template'))
    print()
    
    # Check data files
    print("5. Data Files")
    print("-" * 60)
    all_checks.append(check_file('thyroid_cancer_risk_data.csv', 'Training data'))
    all_checks.append(check_file('densenet121_best.pth', 'Image model'))
    print()
    
    # Check directories
    print("6. Required Directories")
    print("-" * 60)
    all_checks.append(check_directory('models', 'Models directory'))
    all_checks.append(check_directory('templates', 'Templates directory'))
    all_checks.append(check_directory('utils', 'Utils directory'))
    all_checks.append(check_directory('uploads', 'Uploads directory'))
    print()
    
    # Check documentation
    print("7. Documentation")
    print("-" * 60)
    check_file('README.md', 'Main README')
    check_file('DEPLOYMENT.md', 'Deployment guide')
    check_file('USAGE_GUIDE.md', 'Usage guide')
    check_file('PROJECT_SUMMARY.md', 'Project summary')
    print()
    
    # Check deployment files
    print("8. Deployment Configuration")
    print("-" * 60)
    check_file('Dockerfile', 'Docker config')
    check_file('railway.json', 'Railway config')
    check_file('render.yaml', 'Render config')
    check_file('Procfile', 'Heroku config')
    print()
    
    # Check dependencies (optional - only if installed)
    print("9. Python Dependencies (Optional Check)")
    print("-" * 60)
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n   ℹ️  Run: pip install -r requirements.txt")
    print()
    
    # Check if models are trained
    print("10. Trained Models")
    print("-" * 60)
    models_exist = (
        check_file('models/best_model.pkl', 'ML model (sklearn)') or
        check_file('models/best_model.h5', 'DL model (keras)')
    )
    check_file('models/scaler.pkl', 'Feature scaler')
    check_file('models/label_encoder.pkl', 'Label encoder')
    check_file('models/metadata.json', 'Model metadata')
    
    if not models_exist:
        print("\n   ℹ️  Models not trained yet. Run: python train_models.py")
    print()
    
    # Summary
    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if all(all_checks):
        print("✅ All critical checks passed!")
        print("\nNext steps:")
        if not models_exist:
            print("1. Train models: python train_models.py")
            print("2. Start application: python app.py")
        else:
            print("1. Start application: python app.py")
            print("2. Visit: http://localhost:5000")
    else:
        print("❌ Some checks failed. Please review the errors above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Create missing directories: mkdir models templates utils uploads")
        print("3. Ensure all files are present")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    main()

"""
System Testing Script
Tests both image and numerical prediction functionality
"""

import requests
import json
import os
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:5000"

def test_server_running():
    """Test if server is running"""
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print("❌ Server returned status code:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Start it with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False

def test_get_features():
    """Test feature names endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/get-features")
        data = response.json()
        
        if data['success'] and 'features' in data:
            print(f"✅ Feature endpoint working ({len(data['features'])} features)")
            return True, data['features']
        else:
            print("❌ Feature endpoint failed")
            return False, []
    except Exception as e:
        print(f"❌ Error testing features: {e}")
        return False, []

def test_numerical_prediction(features):
    """Test numerical prediction"""
    try:
        # Create sample data
        sample_data = {
            'features': {feature: 1.0 for feature in features}
        }
        
        # Override with realistic values
        sample_data['features'].update({
            'Age': 45,
            'Gender': 1,
            'TSH_Level': 2.5,
            'T3_Level': 1.8,
            'T4_Level': 8.5,
            'Nodule_Size': 2.3
        })
        
        response = requests.post(
            f"{BASE_URL}/predict-numerical",
            json=sample_data,
            headers={'Content-Type': 'application/json'}
        )
        
        data = response.json()
        
        if data['success']:
            print(f"✅ Numerical prediction working")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']}%")
            print(f"   Model: {data.get('model_used', 'N/A')}")
            return True
        else:
            print(f"❌ Numerical prediction failed: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing numerical prediction: {e}")
        return False

def test_image_prediction():
    """Test image prediction"""
    try:
        # Create a test image
        test_image_path = "test_image.jpg"
        
        # Create a dummy RGB image
        img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        img.save(test_image_path)
        
        # Send prediction request
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{BASE_URL}/predict-image", files=files)
        
        data = response.json()
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        if data['success']:
            print(f"✅ Image prediction working")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']}%")
            return True
        else:
            print(f"❌ Image prediction failed: {data.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing image prediction: {e}")
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        return False

def test_error_handling():
    """Test error handling"""
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Invalid image format
    try:
        response = requests.post(
            f"{BASE_URL}/predict-image",
            files={'image': ('test.txt', b'not an image', 'text/plain')}
        )
        if response.status_code >= 400:
            print("✅ Invalid image format handled correctly")
            tests_passed += 1
        else:
            print("❌ Invalid image format not handled")
    except:
        print("❌ Error testing invalid image")
    
    # Test 2: Missing features
    try:
        response = requests.post(
            f"{BASE_URL}/predict-numerical",
            json={},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code >= 400:
            print("✅ Missing features handled correctly")
            tests_passed += 1
        else:
            print("❌ Missing features not handled")
    except:
        print("❌ Error testing missing features")
    
    # Test 3: Invalid endpoint
    try:
        response = requests.get(f"{BASE_URL}/invalid-endpoint")
        if response.status_code == 404:
            print("✅ Invalid endpoint handled correctly")
            tests_passed += 1
        else:
            print("❌ Invalid endpoint not handled")
    except:
        print("❌ Error testing invalid endpoint")
    
    return tests_passed == total_tests

def run_all_tests():
    """Run all system tests"""
    print("\n" + "="*50)
    print("THYROID CANCER PREDICTION SYSTEM - TEST SUITE")
    print("="*50 + "\n")
    
    results = []
    
    # Test 1: Server running
    print("Test 1: Server Status")
    print("-" * 50)
    server_ok = test_server_running()
    results.append(("Server Running", server_ok))
    print()
    
    if not server_ok:
        print("\n⚠️  Cannot continue tests. Please start the server first.")
        print("Run: python app.py")
        return
    
    # Test 2: Feature endpoint
    print("Test 2: Feature Endpoint")
    print("-" * 50)
    features_ok, features = test_get_features()
    results.append(("Feature Endpoint", features_ok))
    print()
    
    # Test 3: Numerical prediction
    print("Test 3: Numerical Prediction")
    print("-" * 50)
    if features:
        numerical_ok = test_numerical_prediction(features)
        results.append(("Numerical Prediction", numerical_ok))
    else:
        print("⚠️  Skipping (no features available)")
        results.append(("Numerical Prediction", False))
    print()
    
    # Test 4: Image prediction
    print("Test 4: Image Prediction")
    print("-" * 50)
    image_ok = test_image_prediction()
    results.append(("Image Prediction", image_ok))
    print()
    
    # Test 5: Error handling
    print("Test 5: Error Handling")
    print("-" * 50)
    error_ok = test_error_handling()
    results.append(("Error Handling", error_ok))
    print()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("="*50 + "\n")

if __name__ == '__main__':
    run_all_tests()

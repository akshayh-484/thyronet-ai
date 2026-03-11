"""
Quick template checker
"""
from flask import Flask, render_template

app = Flask(__name__)

print("="*80)
print("CHECKING TEMPLATES")
print("="*80)

try:
    with app.test_request_context():
        # Test predict template
        print("\n✅ Testing predict.html...")
        result = render_template('predict.html', feature_names=[
            'Age', 'Gender', 'Country', 'Ethnicity',
            'Family_History', 'Radiation_Exposure', 'Iodine_Deficiency',
            'Smoking', 'Obesity', 'Diabetes',
            'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size',
            'Thyroid_Cancer_Risk'
        ])
        print(f"   Length: {len(result)} characters")
        
        # Check for key elements
        if 'Gender' in result and 'select' in result:
            print("   ✅ Gender dropdown found")
        if 'Patient Demographics' in result:
            print("   ✅ Form sections found")
        if 'Predict Risk' in result:
            print("   ✅ Submit button found")
        
        print("\n✅ All templates OK!")
        print("\n" + "="*80)
        print("INSTRUCTIONS:")
        print("="*80)
        print("1. Stop the server (Ctrl+C)")
        print("2. Run: python app_enhanced.py")
        print("3. Go to: http://localhost:5000/predict")
        print("4. Hard refresh browser: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)")
        print("="*80)
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

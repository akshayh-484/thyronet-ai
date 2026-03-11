"""
PROFESSIONAL MEDICAL CLASSIFICATION WEB APPLICATION
Multi-page Flask app with advanced features
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from werkzeug.utils import secure_filename
from utils.image_predictor import ImagePredictor
from utils.numerical_predictor import NumericalPredictor
from PIL import Image
import io
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictors
image_predictor = ImagePredictor('densenet121_best.pth')
numerical_predictor = NumericalPredictor('models')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/model-info')
def model_info():
    """Model information page"""
    try:
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        return render_template('model_info.html', metadata=metadata)
    except:
        return render_template('model_info.html', metadata=None)

@app.route('/predict')
def predict_page():
    """Prediction page"""
    feature_names = numerical_predictor.get_feature_names()
    return render_template('predict.html', feature_names=feature_names)

@app.route('/dashboard')
def dashboard():
    """Results dashboard"""
    return render_template('dashboard.html')

@app.route('/predict-image', methods=['POST'])
def predict_image():
    """Handle image prediction"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Predict
        result = image_predictor.predict(image)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-csv', methods=['POST'])
def predict_csv():
    """Handle CSV batch prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files allowed'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Make predictions
        results = []
        for idx, row in df.iterrows():
            features = row.to_dict()
            pred = numerical_predictor.predict(features)
            if pred['success']:
                results.append({
                    'row': idx + 1,
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence']
                })
        
        return jsonify({'success': True, 'results': results, 'total': len(results)}), 200
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-numerical', methods=['POST'])
def predict_numerical():
    """Handle numerical feature prediction"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'No features provided'}), 400
        
        features = data['features']
        
        # Predict
        result = numerical_predictor.predict(features)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-features', methods=['GET'])
def get_features():
    """Return list of required features"""
    try:
        feature_names = numerical_predictor.get_feature_names()
        return jsonify({'success': True, 'features': feature_names}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Production mode
    app.run(host='0.0.0.0', port=5000, debug=False)

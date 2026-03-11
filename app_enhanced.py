"""
PROFESSIONAL MEDICAL CLASSIFICATION WEB APPLICATION
Multi-page Flask app with advanced features and dashboard
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, flash
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
from utils.ensemble_image_predictor import EnsembleImagePredictor
from utils.numerical_predictor import NumericalPredictor
from utils.ensemble_predictor import EnsemblePredictor
from PIL import Image
import io
import pandas as pd
from functools import wraps

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'thyronet-ai-secret-key-2024'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictors
try:
    image_predictor = EnsembleImagePredictor(
        resnet_path='resnet50_best.pth',
        resnext_path='resnext50_best.pth',
        densenet_path='densenet121_best.pth',
        config_path='ensemble_config.pth'
    )
except Exception as e:
    print(f"Warning: Could not load ensemble image predictor: {e}")
    image_predictor = None

try:
    numerical_predictor = NumericalPredictor('models')
except:
    numerical_predictor = None

try:
    ensemble_predictor = EnsemblePredictor('models')
except:
    ensemble_predictor = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Demo credentials
DEMO_USERS = {
    'doctor': 'thyronet2024',
    'admin': 'admin123'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'logged_in' in session:
        return render_template('home.html', username=session.get('username'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in DEMO_USERS and DEMO_USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful! Welcome to ThyroNet: Thyroid Nodule Analysis.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/predict')
@login_required
def predict_page():
    feature_names = numerical_predictor.get_feature_names() if numerical_predictor else []
    import time
    cache_buster = int(time.time())
    return render_template('predict.html', feature_names=feature_names, v=cache_buster, username=session.get('username'))

@app.route('/about')
@login_required
def about():
    return render_template('about.html', username=session.get('username'))

@app.route('/model-info')
@login_required
def model_info():
    try:
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {}
    return render_template('model_info.html', metadata=metadata, username=session.get('username'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {}
    return render_template('dashboard.html', metadata=metadata, username=session.get('username'))

@app.route('/predict-image', methods=['POST'])
def predict_image():
    # Allow CORS for testing, but check login for web interface
    if not image_predictor:
        return jsonify({'success': False, 'error': 'Image predictor not available'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        result = image_predictor.predict(image)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-numerical', methods=['POST'])
def predict_numerical():
    # Allow CORS for testing, but check login for web interface
    if not numerical_predictor:
        return jsonify({'success': False, 'error': 'Numerical predictor not available - train models first'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'No features provided'}), 400
        
        features = data['features']
        result = numerical_predictor.predict(features)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-numerical-ensemble', methods=['POST'])
def predict_numerical_ensemble():
    # Allow CORS for testing, but check login for web interface
    if not ensemble_predictor:
        return jsonify({'success': False, 'error': 'Ensemble predictor not available - train ensemble models first'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'No features provided'}), 400
        
        features = data['features']
        method = data.get('method', 'best')  # best, voting, stacking, weighted, all
        
        result = ensemble_predictor.predict(features, method=method)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get-features', methods=['GET'])
def get_features():
    if not numerical_predictor:
        return jsonify({'success': False, 'error': 'Predictor not available'}), 500
    
    try:
        feature_names = numerical_predictor.get_feature_names()
        return jsonify({'success': True, 'features': feature_names}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ensemble-status', methods=['GET'])
def ensemble_status():
    """Check if ensemble models are available"""
    try:
        if ensemble_predictor:
            info = ensemble_predictor.get_model_info()
            return jsonify({
                'success': True,
                'available': True,
                'info': info
            }), 200
        else:
            return jsonify({
                'success': True,
                'available': False,
                'message': 'Ensemble models not trained yet. Run: python train_ensemble.py'
            }), 200
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
    app.run(host='0.0.0.0', port=5000, debug=False)

"""
THYRONET: THYROID NODULE ANALYSIS
Clinical Data Prediction using Ensemble Machine Learning
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_cors import CORS
import os
import json
from utils.advanced_image_predictor import AdvancedImagePredictor
from utils.super_ensemble_predictor import SuperEnsemblePredictor
from utils.thyroid_disease_predictor import ThyroidDiseasePredictor
from utils.tirads_calculator import calculate_tirads, UI_OPTIONS
from utils.cancer_risk_predictor import CancerRiskPredictor
from functools import wraps
import time
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'thyronet-ai-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictors

try:
    thyroid_disease_predictor = ThyroidDiseasePredictor('models_thyroid_disease')
except Exception as e:
    print(f"Warning: Could not load thyroid disease predictor: {e}")
    thyroid_disease_predictor = None

try:
    cancer_risk_predictor = CancerRiskPredictor('models_cancer_risk')
except Exception as e:
    print(f"Warning: Could not load cancer risk predictor: {e}")
    cancer_risk_predictor = None

try:
    # Load retrained models (no data leakage, TN5000 dataset)
    print("🚀 Loading Super Ensemble Predictor (models_retrained/)...")
    image_predictor = SuperEnsemblePredictor(models_dir='models_retrained')
    print("✅ Super Ensemble Predictor loaded successfully!")
    predictor_type = 'super_ensemble'
except Exception as e:
    print(f"⚠️  Could not load from models_retrained/: {e}")
    print("Trying legacy modelss/ directory...")
    try:
        image_predictor = SuperEnsemblePredictor(models_dir='modelss')
        print("✅ Loaded from modelss/ (legacy)")
        predictor_type = 'super_ensemble'
    except Exception as e2:
        print(f"❌ Could not load any image predictor: {e2}")
        image_predictor = None
        predictor_type = None

# Demo credentials
DEMO_USERS = {
    'doctor': 'thyronet2024',
    'admin': 'admin123'
}

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/predict-image')
@login_required
def predict_image_page():
    cache_buster = int(time.time())
    return render_template('predict_image.html', v=cache_buster, username=session.get('username'))

@app.route('/about')
@login_required
def about():
    return render_template('about.html', username=session.get('username'))

@app.route('/model-info')
@login_required
def model_info():
    metadata = {}
    return render_template('model_info.html', metadata=metadata, username=session.get('username'))

@app.route('/dashboard')
@login_required
def dashboard():
    metadata = {}
    return render_template('dashboard.html', metadata=metadata, username=session.get('username'))

@app.route('/predict-image-upload', methods=['POST'])
@login_required
def predict_image_upload():
    if not image_predictor:
        return jsonify({'success': False, 'error': 'Image predictor not available. Please ensure model files are in newmod/ directory.'}), 500
    
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
        
        # Open and validate image
        try:
            image = Image.open(file.stream).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'}), 400
        
        # Predict
        result = image_predictor.predict(image)
        
        return jsonify(result), 200 if result['success'] else 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

@app.route('/image-predictor-status', methods=['GET'])
def image_predictor_status():
    try:
        return jsonify({
            'success': True,
            'available': image_predictor is not None,
            'models': {
                'yolo': 'newmod/yolo_model.pt',
                'densenet': 'newmod/densenet_model.h5',
                'mobilenet': 'newmod/mobilenet_model.h5'
            } if image_predictor else None
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ── Thyroid Disease Detection (Hypo/Hyper/Normal) ─────────────────
@app.route('/predict-thyroid-disease-page')
@login_required
def predict_thyroid_disease_page():
    cache_buster = int(time.time())
    info = thyroid_disease_predictor.get_model_info() if thyroid_disease_predictor else {}
    return render_template('predict_thyroid_disease.html',
                           model_info=info,
                           v=cache_buster,
                           username=session.get('username'))

@app.route('/predict-thyroid-disease', methods=['POST'])
@login_required
def predict_thyroid_disease():
    if not thyroid_disease_predictor:
        return jsonify({'success': False, 'error': 'Thyroid disease predictor not available'}), 500
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'No features provided'}), 400
        result = thyroid_disease_predictor.predict(data['features'])
        result['success'] = True
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/thyroid-disease-info', methods=['GET'])
@login_required
def thyroid_disease_info():
    try:
        if thyroid_disease_predictor:
            return jsonify({'success': True, 'available': True,
                            'info': thyroid_disease_predictor.get_model_info()}), 200
        return jsonify({'success': True, 'available': False}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/tirads-page')
@login_required
def tirads_page():
    cache_buster = int(time.time())
    return render_template('tirads.html',
                           ui_options=UI_OPTIONS,
                           v=cache_buster,
                           username=session.get('username'))

@app.route('/calculate-tirads', methods=['POST'])
@login_required
def calculate_tirads_route():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        result = calculate_tirads(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ── Cancer Risk (Benign/Malignant) ────────────────────────────────
@app.route('/cancer-risk-page')
@login_required
def cancer_risk_page():
    cache_buster = int(time.time())
    info = cancer_risk_predictor.get_model_info() if cancer_risk_predictor else {}
    return render_template('cancer_risk.html', model_info=info, ui_options=UI_OPTIONS,
                           v=cache_buster, username=session.get('username'))

@app.route('/predict-cancer-risk', methods=['POST'])
@login_required
def predict_cancer_risk():
    if not cancer_risk_predictor:
        return jsonify({'success': False, 'error': 'Cancer risk predictor not available'}), 500
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'No features provided'}), 400
        result = cancer_risk_predictor.predict(data['features'])
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

"""
Advanced ThyroNet Image Predictor
Supports multiple YOLO and classification models from modelss/ directory
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from torchvision import models as torchvision_models


class AdvancedImagePredictor:
    """
    Advanced image predictor supporting multiple model architectures:
    
    YOLO Models (Detection):
    - yolo11x.pt
    - yolo26n_v2.pt
    - yolov8m_aug_thyroid.pt
    - yolov8m.pt
    - yolov8x.pt
    - yolov9e.pt
    
    Classification Models:
    - convnext_tiny_thyroid.pt
    - densenet121_thyroid.pt
    - efficientnetv2_m_thyroid.pt
    - mobilenetv3_large_thyroid.pt
    - resnet152_thyroid.pt
    - vit_b16_thyroid.pt
    """
    
    def __init__(self, 
                 yolo_model='yolov8m_aug_thyroid.pt',
                 classifier_models=None,
                 models_dir='modelss'):
        """
        Initialize predictor with specified models
        
        Args:
            yolo_model: YOLO model filename for detection
            classifier_models: List of classifier model filenames (if None, uses all)
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load YOLO detection model
        yolo_path = os.path.join(models_dir, yolo_model)
        print(f"Loading YOLO model: {yolo_model}...")
        self.yolo = YOLO(yolo_path)
        
        # Load classification models
        self.classifiers = {}
        self.classifier_weights = {}
        
        if classifier_models is None:
            # Use all available classifiers
            classifier_models = [
                'densenet121_thyroid.pt',
                'resnet152_thyroid.pt',
                'efficientnetv2_m_thyroid.pt',
                'mobilenetv3_large_thyroid.pt',
                'convnext_tiny_thyroid.pt',
                'vit_b16_thyroid.pt'
            ]
        
        # Load each classifier
        for model_name in classifier_models:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                try:
                    print(f"Loading classifier: {model_name}...")
                    model = self._load_classifier(model_path, model_name)
                    if model is not None:
                        self.classifiers[model_name] = model
                        # Equal weights for now (can be optimized)
                        self.classifier_weights[model_name] = 1.0 / len(classifier_models)
                except Exception as e:
                    print(f"Warning: Could not load {model_name}: {e}")
        
        if not self.classifiers:
            raise ValueError("No classifiers loaded successfully!")
        
        print(f"✅ Loaded {len(self.classifiers)} classifiers")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_classifier(self, model_path, model_name):
        """Load a PyTorch classification model"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine architecture from name
            if 'densenet121' in model_name:
                model = torchvision_models.densenet121(pretrained=False)
                model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            elif 'resnet152' in model_name:
                model = torchvision_models.resnet152(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
            elif 'efficientnetv2' in model_name:
                model = torchvision_models.efficientnet_v2_m(pretrained=False)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
            elif 'mobilenetv3' in model_name:
                model = torchvision_models.mobilenet_v3_large(pretrained=False)
                model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
            elif 'convnext' in model_name:
                model = torchvision_models.convnext_tiny(pretrained=False)
                model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
            elif 'vit' in model_name:
                model = torchvision_models.vit_b_16(pretrained=False)
                model.heads.head = torch.nn.Linear(model.heads.head.in_features, 2)
            else:
                print(f"Unknown architecture for {model_name}")
                return None
            
            # Load weights
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    
    def _validate_image(self, img_pil):
        """Validate that image is a proper ultrasound"""
        img_array = np.array(img_pil)
        
        # Check if grayscale or color
        if len(img_array.shape) == 2:
            gray_ch = img_array
        else:
            r = img_array[:,:,0].astype(float)
            g = img_array[:,:,1].astype(float)
            b = img_array[:,:,2].astype(float)
            avg_color_diff = (np.mean(np.abs(r-g)) + np.mean(np.abs(r-b)) + np.mean(np.abs(g-b))) / 3
            gray_ch = np.mean(img_array, axis=2)
            
            # Reject clearly colored images
            if avg_color_diff > 15:
                return False, '❌ Invalid image. Please upload a grayscale thyroid ultrasound image.'
        
        white_ratio = np.sum(gray_ch > 220) / gray_ch.size
        dark_ratio = np.sum(gray_ch < 30) / gray_ch.size
        std_brightness = np.std(gray_ch)
        
        # Reject images with too much white (documents, charts)
        if white_ratio > 0.35:
            return False, '❌ Invalid image. This looks like a document or screenshot.'
        
        # Reject blank images
        if std_brightness < 10:
            return False, '❌ Image appears blank. Please upload a real thyroid ultrasound.'
        
        # Ultrasounds have significant dark regions
        if dark_ratio < 0.05:
            return False, '❌ Invalid image. Thyroid ultrasounds typically have dark regions.'
        
        return True, None
    
    def predict(self, image_input):
        """
        Predict thyroid nodule classification
        
        Args:
            image_input: PIL Image or path to image file
            
        Returns:
            dict with prediction results
        """
        try:
            # Load image
            if isinstance(image_input, str):
                img_pil = Image.open(image_input).convert('RGB')
            else:
                img_pil = image_input.convert('RGB')
            
            # Validate image
            is_valid, error_msg = self._validate_image(img_pil)
            if not is_valid:
                return {'success': False, 'error': error_msg}
            
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            orig_image = image.copy()
            orig_h, orig_w = image.shape[:2]
            
            # Remove black borders
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)
                if bw * bh > orig_w * orig_h * 0.3:
                    image_for_yolo = image[by:by+bh, bx:bx+bw]
                    border_offset = (bx, by)
                else:
                    image_for_yolo = image
                    border_offset = (0, 0)
            else:
                image_for_yolo = image
                border_offset = (0, 0)
            
            # YOLO detection
            results = self.yolo(image_for_yolo, verbose=False)[0]
            
            if len(results.boxes) == 0:
                return {
                    'success': False,
                    'error': '❌ No thyroid nodule detected. Please upload a clear ultrasound showing a nodule.'
                }
            
            # Get best detection
            boxes = results.boxes
            best_idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
            x1 += border_offset[0]; y1 += border_offset[1]
            x2 += border_offset[0]; y2 += border_offset[1]
            detection_conf = float(boxes.conf[best_idx])
            
            # Crop nodule with padding
            pad = 10
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(orig_w, x2 + pad)
            y2c = min(orig_h, y2 + pad)
            crop = image[y1c:y2c, x1c:x2c]
            
            if crop.size == 0:
                return {'success': False, 'error': 'Could not crop nodule region'}
            
            # Preprocess with CLAHE
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            crop_eq = clahe.apply(crop_gray)
            crop_rgb = cv2.cvtColor(crop_eq, cv2.COLOR_GRAY2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            # Transform for models
            crop_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            
            # Get predictions from all classifiers
            individual_predictions = {}
            weighted_sum = 0.0
            
            with torch.no_grad():
                for model_name, model in self.classifiers.items():
                    output = model(crop_tensor)
                    probs = torch.softmax(output, dim=1)
                    malignant_prob = float(probs[0][1])
                    
                    weight = self.classifier_weights[model_name]
                    weighted_sum += malignant_prob * weight
                    
                    pred_class = 'Malignant' if malignant_prob > 0.5 else 'Benign'
                    confidence = malignant_prob if malignant_prob > 0.5 else (1 - malignant_prob)
                    
                    individual_predictions[model_name.replace('_thyroid.pt', '')] = {
                        'prediction': pred_class,
                        'confidence': round(confidence * 100, 2),
                        'malignant_prob': round(malignant_prob * 100, 2),
                        'weight': weight
                    }
            
            # Ensemble prediction
            final_score = weighted_sum
            predicted_class = 'Malignant' if final_score > 0.51 else 'Benign'
            
            # Calculate confidence (50-99%)
            if predicted_class == 'Malignant':
                confidence = round(50 + (final_score - 0.51) / (1.0 - 0.51) * 49, 2)
            else:
                confidence = round(50 + (0.51 - final_score) / 0.51 * 49, 2)
            confidence = float(min(99.0, max(50.0, confidence)))
            
            # Draw detection box
            segmentation_image = self._draw_detection(
                orig_image, x1, y1, x2, y2, predicted_class, confidence
            )
            
            return {
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    'Benign': round((1 - final_score) * 100, 2),
                    'Malignant': round(final_score * 100, 2)
                },
                'model_used': f'YOLO + {len(self.classifiers)}-Model Ensemble',
                'detection_confidence': round(detection_conf * 100, 2),
                'individual_predictions': individual_predictions,
                'segmentation_image': segmentation_image,
                'ensemble_details': {
                    'num_models': len(self.classifiers),
                    'models_used': list(self.classifiers.keys())
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': f'Prediction error: {str(e)}'}
    
    def _draw_detection(self, image, x1, y1, x2, y2, prediction, confidence):
        """Draw detection box on image"""
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        color = (255, 50, 50) if prediction == 'Malignant' else (50, 220, 50)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        
        label = f"{prediction} {confidence:.1f}%"
        font_scale = max(0.5, image.shape[1] / 700)
        thickness = max(1, int(image.shape[1] / 400))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(th + 10, y1 - 8)
        
        cv2.rectangle(overlay, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 4), color, -1)
        cv2.putText(overlay, label, (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        result_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        return f"data:image/png;base64,{img_b64}"

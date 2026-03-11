"""
Hybrid Deep Learning Ensemble Image Predictor
Combines ResNet50, ResNeXt50, and DenseNet121 for thyroid cancer detection
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class EnsembleImagePredictor:
    def __init__(self, 
                 resnet_path='resnet50_best.pth',
                 resnext_path='resnext50_best.pth', 
                 densenet_path='densenet121_best.pth',
                 config_path='ensemble_config.pth'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ensemble configuration
        self.config = torch.load(config_path, map_location=self.device)
        self.weights = self.config['weights']
        self.threshold = self.config['threshold']
        
        # Load all three models
        self.resnet = self.load_resnet(resnet_path)
        self.resnext = self.load_resnext(resnext_path)
        self.densenet = self.load_densenet(densenet_path)
        
        self.transform = self.get_transforms()
        self.classes = ['Benign', 'Malignant']
        
    def load_resnet(self, model_path):
        """Load ResNet50 model"""
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_resnext(self, model_path):
        """Load ResNeXt50 model"""
        model = models.resnext50_32x4d(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def load_densenet(self, model_path):
        """Load DenseNet121 model"""
        model = models.densenet121(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def validate_image(self, image):
        """
        Basic validation to check if image might be a medical ultrasound
        Returns: (is_valid, warning_message)
        """
        import numpy as np
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Check if image is too small
        if image.size[0] < 100 or image.size[1] < 100:
            return False, "Image too small. Please upload a clear ultrasound image (minimum 100x100 pixels)."
        
        # Check if image is grayscale or has low color variance (ultrasounds are typically grayscale)
        if len(img_array.shape) == 3:
            # Calculate color variance
            r_var = np.var(img_array[:,:,0])
            g_var = np.var(img_array[:,:,1])
            b_var = np.var(img_array[:,:,2])
            
            # If all channels are very similar, it's likely grayscale (good for ultrasound)
            channel_diff = max(abs(r_var - g_var), abs(g_var - b_var), abs(r_var - b_var))
            
            # Very high color variance suggests it's not a medical image
            if channel_diff > 1000:
                return True, "Warning: This image appears to be a color photo. Thyroid ultrasounds are typically grayscale. Results may not be accurate."
        
        return True, None
    
    def predict(self, image_path):
        """
        Predict using hybrid ensemble (ResNet + ResNeXt + DenseNet)
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            dict with prediction, confidence, probabilities, and individual model results
        """
        try:
            # Load and preprocess image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            # Validate image
            is_valid, warning = self.validate_image(image)
            if not is_valid:
                return {
                    'success': False,
                    'error': warning
                }
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions from all three models
            with torch.no_grad():
                # ResNet50 prediction
                resnet_outputs = self.resnet(image_tensor)
                resnet_probs = torch.softmax(resnet_outputs, dim=1)[0].cpu().numpy()
                
                # ResNeXt50 prediction
                resnext_outputs = self.resnext(image_tensor)
                resnext_probs = torch.softmax(resnext_outputs, dim=1)[0].cpu().numpy()
                
                # DenseNet121 prediction
                densenet_outputs = self.densenet(image_tensor)
                densenet_probs = torch.softmax(densenet_outputs, dim=1)[0].cpu().numpy()
            
            # Weighted ensemble (soft voting)
            ensemble_probs = (
                self.weights['resnet'] * resnet_probs +
                self.weights['resnext'] * resnext_probs +
                self.weights['densenet'] * densenet_probs
            )
            
            # Get final prediction
            predicted_class = np.argmax(ensemble_probs)
            pred_class = self.classes[predicted_class]
            confidence = ensemble_probs[predicted_class] * 100
            
            result = {
                'prediction': pred_class,
                'confidence': round(float(confidence), 2),
                'probabilities': {
                    'Benign': round(float(ensemble_probs[0]) * 100, 2),
                    'Malignant': round(float(ensemble_probs[1]) * 100, 2)
                },
                'model_used': 'Hybrid Ensemble (ResNet50 + ResNeXt50 + DenseNet121)',
                'individual_predictions': {
                    'ResNet50': {
                        'prediction': self.classes[np.argmax(resnet_probs)],
                        'confidence': round(float(np.max(resnet_probs)) * 100, 2),
                        'weight': self.weights['resnet']
                    },
                    'ResNeXt50': {
                        'prediction': self.classes[np.argmax(resnext_probs)],
                        'confidence': round(float(np.max(resnext_probs)) * 100, 2),
                        'weight': self.weights['resnext']
                    },
                    'DenseNet121': {
                        'prediction': self.classes[np.argmax(densenet_probs)],
                        'confidence': round(float(np.max(densenet_probs)) * 100, 2),
                        'weight': self.weights['densenet']
                    }
                },
                'ensemble_weights': self.weights,
                'success': True
            }
            
            # Add warning if image validation raised concerns
            if warning:
                result['warning'] = warning
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

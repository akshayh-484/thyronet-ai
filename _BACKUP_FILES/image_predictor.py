"""
Image Prediction Module using Ensemble Deep Learning Model
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ImagePredictor:
    def __init__(self, model_path='densenet121_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = self.get_transforms()
        self.classes = ['Benign', 'Malignant']
        
    def load_model(self, model_path):
        """Load pre-trained Ensemble model (DenseNet121 architecture)"""
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Create DenseNet121 architecture
        model = models.densenet121(pretrained=False)
        
        # Modify classifier for binary classification
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        # Load weights
        model.load_state_dict(state_dict)
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
    
    def predict(self, image_path):
        """
        Predict malignant/benign from image
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        try:
            # Load and preprocess image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path.convert('RGB')
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # Get results
            pred_class = self.classes[predicted.item()]
            conf_percent = confidence.item() * 100
            probs = probabilities[0].cpu().numpy()
            
            return {
                'prediction': pred_class,
                'confidence': round(conf_percent, 2),
                'probabilities': {
                    'Benign': round(float(probs[0]) * 100, 2),
                    'Malignant': round(float(probs[1]) * 100, 2)
                },
                'model_used': 'Ensemble Model',
                'success': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

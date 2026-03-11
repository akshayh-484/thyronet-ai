"""
Ensemble Predictor for Numerical Data
Combines multiple models for superior predictions
"""

import pickle
import json
import numpy as np

class EnsemblePredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load all ensemble models and artifacts"""
        # Load scaler
        with open(f'{self.models_dir}/ensemble_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load label encoder
        with open(f'{self.models_dir}/ensemble_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load feature names
        with open(f'{self.models_dir}/ensemble_feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load metadata
        with open(f'{self.models_dir}/ensemble_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load individual models
        with open(f'{self.models_dir}/ensemble_lr.pkl', 'rb') as f:
            self.lr_model = pickle.load(f)
        
        with open(f'{self.models_dir}/ensemble_svm.pkl', 'rb') as f:
            self.svm_model = pickle.load(f)
        
        with open(f'{self.models_dir}/ensemble_rf.pkl', 'rb') as f:
            self.rf_model = pickle.load(f)
        
        with open(f'{self.models_dir}/ensemble_xgb.pkl', 'rb') as f:
            self.xgb_model = pickle.load(f)
        
        # Load ensemble models
        with open(f'{self.models_dir}/voting_ensemble.pkl', 'rb') as f:
            self.voting_ensemble = pickle.load(f)
        
        with open(f'{self.models_dir}/stacking_ensemble.pkl', 'rb') as f:
            self.stacking_ensemble = pickle.load(f)
        
        # Load weighted ensemble config
        with open(f'{self.models_dir}/weighted_ensemble_config.json', 'r') as f:
            self.weighted_config = json.load(f)
        
        self.classes = self.metadata['classes']
        self.best_model_name = self.metadata['best_model_name']
    
    def predict(self, features, method='best'):
        """
        Predict using ensemble methods
        
        Args:
            features: dict or list of feature values
            method: 'best', 'voting', 'stacking', 'weighted', or 'all'
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        try:
            # Encode categorical features
            encoded_features = {}
            
            # Mapping for categorical values
            gender_map = {'Male': 1, 'Female': 0}
            yes_no_map = {'Yes': 1, 'No': 0}
            ethnicity_map = {'Caucasian': 0, 'Asian': 1, 'African': 2, 'Hispanic': 3, 'Other': 4}
            risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
            
            for name in self.feature_names:
                value = features.get(name, 0) if isinstance(features, dict) else 0
                
                # Convert categorical to numeric
                if name == 'Gender' and isinstance(value, str):
                    value = gender_map.get(value, 1)
                elif name in ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 
                             'Smoking', 'Obesity', 'Diabetes'] and isinstance(value, str):
                    value = yes_no_map.get(value, 0)
                elif name == 'Ethnicity' and isinstance(value, str):
                    value = ethnicity_map.get(value, 0)
                elif name == 'Thyroid_Cancer_Risk' and isinstance(value, str):
                    value = risk_map.get(value, 0)
                elif name == 'Country' and isinstance(value, str):
                    value = hash(value) % 100
                
                try:
                    encoded_features[name] = float(value)
                except:
                    encoded_features[name] = 0.0
            
            # Convert dict to array
            feature_array = np.array([encoded_features.get(name, 0) for name in self.feature_names])
            
            # Reshape and scale
            feature_array = feature_array.reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_array)
            
            if method == 'voting':
                return self._predict_voting(feature_scaled)
            elif method == 'stacking':
                return self._predict_stacking(feature_scaled)
            elif method == 'weighted':
                return self._predict_weighted(feature_scaled)
            elif method == 'all':
                return self._predict_all(feature_scaled)
            else:  # best
                return self._predict_best(feature_scaled)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _predict_voting(self, feature_scaled):
        """Predict using voting ensemble"""
        probabilities = self.voting_ensemble.predict_proba(feature_scaled)[0]
        pred_idx = np.argmax(probabilities)
        prediction = self.classes[pred_idx]
        confidence = probabilities[pred_idx] * 100
        
        return {
            'prediction': prediction,
            'confidence': round(float(confidence), 2),
            'probabilities': {
                self.classes[0]: round(float(probabilities[0]) * 100, 2),
                self.classes[1]: round(float(probabilities[1]) * 100, 2)
            },
            'method': 'Voting Ensemble',
            'success': True
        }
    
    def _predict_stacking(self, feature_scaled):
        """Predict using stacking ensemble"""
        probabilities = self.stacking_ensemble.predict_proba(feature_scaled)[0]
        pred_idx = np.argmax(probabilities)
        prediction = self.classes[pred_idx]
        confidence = probabilities[pred_idx] * 100
        
        return {
            'prediction': prediction,
            'confidence': round(float(confidence), 2),
            'probabilities': {
                self.classes[0]: round(float(probabilities[0]) * 100, 2),
                self.classes[1]: round(float(probabilities[1]) * 100, 2)
            },
            'method': 'Stacking Ensemble',
            'success': True
        }
    
    def _predict_weighted(self, feature_scaled):
        """Predict using weighted ensemble"""
        weights = self.weighted_config['weights']
        
        # Get predictions from all models
        models = {
            'Logistic Regression': self.lr_model,
            'SVM': self.svm_model,
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model
        }
        
        # Weighted average of probabilities
        weighted_proba = np.zeros(2)
        for name, model in models.items():
            if name in weights:
                proba = model.predict_proba(feature_scaled)[0]
                weighted_proba += proba * weights[name]
        
        pred_idx = np.argmax(weighted_proba)
        prediction = self.classes[pred_idx]
        confidence = weighted_proba[pred_idx] * 100
        
        return {
            'prediction': prediction,
            'confidence': round(float(confidence), 2),
            'probabilities': {
                self.classes[0]: round(float(weighted_proba[0]) * 100, 2),
                self.classes[1]: round(float(weighted_proba[1]) * 100, 2)
            },
            'method': 'Weighted Ensemble',
            'weights': weights,
            'success': True
        }
    
    def _predict_best(self, feature_scaled):
        """Predict using best performing ensemble"""
        if self.best_model_name == 'Voting Ensemble (Soft)':
            return self._predict_voting(feature_scaled)
        elif self.best_model_name == 'Stacking Ensemble':
            return self._predict_stacking(feature_scaled)
        elif self.best_model_name == 'Weighted Ensemble':
            return self._predict_weighted(feature_scaled)
        else:
            # Fallback to weighted
            return self._predict_weighted(feature_scaled)
    
    def _predict_all(self, feature_scaled):
        """Get predictions from all ensemble methods"""
        voting_result = self._predict_voting(feature_scaled)
        stacking_result = self._predict_stacking(feature_scaled)
        weighted_result = self._predict_weighted(feature_scaled)
        
        # Aggregate results
        all_predictions = [
            voting_result['prediction'],
            stacking_result['prediction'],
            weighted_result['prediction']
        ]
        
        # Majority vote
        from collections import Counter
        vote_counts = Counter(all_predictions)
        final_prediction = vote_counts.most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = (
            voting_result['confidence'] +
            stacking_result['confidence'] +
            weighted_result['confidence']
        ) / 3
        
        return {
            'prediction': final_prediction,
            'confidence': round(avg_confidence, 2),
            'individual_results': {
                'voting': voting_result,
                'stacking': stacking_result,
                'weighted': weighted_result
            },
            'method': 'All Ensembles (Majority Vote)',
            'success': True
        }
    
    def get_feature_names(self):
        """Return list of required feature names"""
        return self.feature_names
    
    def get_model_info(self):
        """Return ensemble model information"""
        return {
            'best_model': self.best_model_name,
            'accuracy': self.metadata['best_model_accuracy'],
            'roc_auc': self.metadata['best_model_roc_auc'],
            'all_models': self.metadata['all_models'],
            'training_date': self.metadata['training_date']
        }

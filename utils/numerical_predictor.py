"""
Numerical Data Prediction Module
"""

import pickle
import json
import numpy as np

class NumericalPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load model, scaler, and metadata"""
        # Load metadata
        with open(f'{self.models_dir}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load scaler - use ensemble scaler for ensemble models
        try:
            with open(f'{self.models_dir}/ensemble_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Loaded ensemble scaler (15 features)")
            self.use_engineered_features = False
        except:
            with open(f'{self.models_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Loaded regular scaler (54 features)")
            self.use_engineered_features = True
        
        # Load label encoder
        with open(f'{self.models_dir}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load categorical encoders if available
        try:
            with open(f'{self.models_dir}/categorical_encoders.pkl', 'rb') as f:
                self.categorical_encoders = pickle.load(f)
        except:
            self.categorical_encoders = None
        
        # Load MULTIPLE models for TRUE ensemble prediction
        try:
            # Load all 4 individual models
            with open(f'{self.models_dir}/ensemble_lr.pkl', 'rb') as f:
                self.lr_model = pickle.load(f)
            with open(f'{self.models_dir}/ensemble_svm.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            with open(f'{self.models_dir}/ensemble_rf.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            with open(f'{self.models_dir}/ensemble_xgb.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            
            self.use_ensemble = True
            self.model = None  # We'll use multiple models
            print("Loaded 4 models for TRUE ENSEMBLE: LR + SVM + RF + XGBoost")
        except Exception as e:
            # Fallback to single XGBoost if ensemble files don't exist
            with open(f'{self.models_dir}/ensemble_xgb.pkl', 'rb') as f:
                self.model = pickle.load(f)
            self.use_ensemble = False
            print(f"Loaded single XGBoost model (fallback): {e}")
        
        self.feature_names = self.metadata['feature_names']
        self.classes = self.metadata['classes']
        
        # If using ensemble model with 15 features, use only original features
        if not self.use_engineered_features:
            self.feature_names = [
                'Age', 'Gender', 'Country', 'Ethnicity', 'Family_History',
                'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity',
                'Diabetes', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size',
                'Thyroid_Cancer_Risk'
            ]
            print(f"Using {len(self.feature_names)} original features")
        else:
            print(f"Using {len(self.feature_names)} features (with engineered)")
        
        # Get encoding mappings from metadata
        self.encoding_maps = self.metadata.get('categorical_encoders', {})
        
    def predict(self, features):
        """
        Predict from numerical features
        
        Args:
            features: dict or list of feature values
            
        Returns:
            dict with prediction, confidence, and probabilities
        """
        try:
            # Encode categorical features using saved mappings
            encoded_features = {}
            
            # Only process features we need
            for name in self.feature_names:
                # Skip engineered features if not using them
                if not self.use_engineered_features and name not in [
                    'Age', 'Gender', 'Country', 'Ethnicity', 'Family_History',
                    'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity',
                    'Diabetes', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size',
                    'Thyroid_Cancer_Risk'
                ]:
                    continue
                
                # Skip engineered features - we'll calculate them later if needed
                if self.use_engineered_features and name in ['TSH_T3_ratio', 'TSH_T4_ratio', 'T3_T4_ratio', 'Nodule_TSH_interaction', 'Age_group']:
                    continue
                    
                value = features.get(name, 0)
                
                # Use saved encoding mappings if available
                if name in self.encoding_maps and isinstance(value, str):
                    value = self.encoding_maps[name].get(value, 0)
                
                # Convert to float
                try:
                    encoded_features[name] = float(value)
                except:
                    encoded_features[name] = 0.0
            
            # Only calculate engineered features if using them
            if self.use_engineered_features:
                # Calculate ALL engineered features (matching save_best_model.py)
                tsh = encoded_features.get('TSH_Level', 0)
                t3 = encoded_features.get('T3_Level', 0)
                t4 = encoded_features.get('T4_Level', 0)
                nodule = encoded_features.get('Nodule_Size', 0)
                age = encoded_features.get('Age', 0)
            
            # Hormone ratios
            if 'TSH_T3_ratio' in self.feature_names:
                encoded_features['TSH_T3_ratio'] = tsh / (t3 + 0.001)
            if 'TSH_T4_ratio' in self.feature_names:
                encoded_features['TSH_T4_ratio'] = tsh / (t4 + 0.001)
            if 'T3_T4_ratio' in self.feature_names:
                encoded_features['T3_T4_ratio'] = t3 / (t4 + 0.001)
            if 'T3_T4_product' in self.feature_names:
                encoded_features['T3_T4_product'] = t3 * t4
            if 'Hormone_sum' in self.feature_names:
                encoded_features['Hormone_sum'] = tsh + t3 + t4
            if 'Hormone_balance' in self.feature_names:
                encoded_features['Hormone_balance'] = (tsh + t3 + t4) / 3
            if 'Hormone_variance' in self.feature_names:
                hb = (tsh + t3 + t4) / 3
                encoded_features['Hormone_variance'] = ((tsh - hb)**2 + (t3 - hb)**2 + (t4 - hb)**2) / 3
            
            # Polynomial features
            if 'TSH_squared' in self.feature_names:
                encoded_features['TSH_squared'] = tsh ** 2
            if 'T3_squared' in self.feature_names:
                encoded_features['T3_squared'] = t3 ** 2
            if 'T4_squared' in self.feature_names:
                encoded_features['T4_squared'] = t4 ** 2
            if 'TSH_cubed' in self.feature_names:
                encoded_features['TSH_cubed'] = tsh ** 3
            if 'T3_cubed' in self.feature_names:
                encoded_features['T3_cubed'] = t3 ** 3
            if 'T4_cubed' in self.feature_names:
                encoded_features['T4_cubed'] = t4 ** 3
            
            # Nodule features
            if 'Nodule_TSH_interaction' in self.feature_names:
                encoded_features['Nodule_TSH_interaction'] = nodule * tsh
            if 'Nodule_T3_interaction' in self.feature_names:
                encoded_features['Nodule_T3_interaction'] = nodule * t3
            if 'Nodule_T4_interaction' in self.feature_names:
                encoded_features['Nodule_T4_interaction'] = nodule * t4
            if 'Nodule_squared' in self.feature_names:
                encoded_features['Nodule_squared'] = nodule ** 2
            if 'Nodule_log' in self.feature_names:
                encoded_features['Nodule_log'] = np.log1p(nodule)
            if 'Nodule_cubed' in self.feature_names:
                encoded_features['Nodule_cubed'] = nodule ** 3
            if 'Nodule_sqrt' in self.feature_names:
                encoded_features['Nodule_sqrt'] = np.sqrt(nodule)
            
            # Age features
            if 'Age_squared' in self.feature_names:
                encoded_features['Age_squared'] = age ** 2
            if 'Age_log' in self.feature_names:
                encoded_features['Age_log'] = np.log1p(age)
            if 'Age_cubed' in self.feature_names:
                encoded_features['Age_cubed'] = age ** 3
            if 'Age_sqrt' in self.feature_names:
                encoded_features['Age_sqrt'] = np.sqrt(age)
            if 'Age_group' in self.feature_names:
                if age <= 30:
                    encoded_features['Age_group'] = 0
                elif age <= 50:
                    encoded_features['Age_group'] = 1
                elif age <= 70:
                    encoded_features['Age_group'] = 2
                else:
                    encoded_features['Age_group'] = 3
            
            # Age-hormone interactions
            if 'Age_TSH_interaction' in self.feature_names:
                encoded_features['Age_TSH_interaction'] = age * tsh
            if 'Age_T3_interaction' in self.feature_names:
                encoded_features['Age_T3_interaction'] = age * t3
            if 'Age_T4_interaction' in self.feature_names:
                encoded_features['Age_T4_interaction'] = age * t4
            if 'Age_Nodule_interaction' in self.feature_names:
                encoded_features['Age_Nodule_interaction'] = age * nodule
            
            # Critical thresholds
            if 'TSH_high' in self.feature_names:
                encoded_features['TSH_high'] = 1 if tsh > 4.5 else 0
            if 'TSH_low' in self.feature_names:
                encoded_features['TSH_low'] = 1 if tsh < 0.5 else 0
            if 'TSH_very_high' in self.feature_names:
                encoded_features['TSH_very_high'] = 1 if tsh > 10.0 else 0
            if 'TSH_normal' in self.feature_names:
                encoded_features['TSH_normal'] = 1 if (tsh >= 0.5 and tsh <= 4.5) else 0
            if 'Nodule_large' in self.feature_names:
                encoded_features['Nodule_large'] = 1 if nodule > 1.0 else 0
            if 'Nodule_very_large' in self.feature_names:
                encoded_features['Nodule_very_large'] = 1 if nodule > 2.0 else 0
            if 'Nodule_small' in self.feature_names:
                encoded_features['Nodule_small'] = 1 if nodule < 0.5 else 0
            
            # Complex interactions
            if 'TSH_T3_T4_product' in self.feature_names:
                encoded_features['TSH_T3_T4_product'] = tsh * t3 * t4
            if 'Nodule_Hormone_sum' in self.feature_names:
                encoded_features['Nodule_Hormone_sum'] = nodule * (tsh + t3 + t4)
            
            # Risk score
            if 'Risk_score' in self.feature_names:
                risk_score = 0
                for col in ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity', 'Diabetes']:
                    risk_score += encoded_features.get(col, 0)
                encoded_features['Risk_score'] = risk_score
            
            # Convert dict to array
            feature_array = np.array([encoded_features.get(name, 0) for name in self.feature_names])
            
            # Reshape and scale
            feature_array = feature_array.reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_array)
            
            # Predict using ENSEMBLE (multiple models voting together!)
            if self.use_ensemble:
                # Get predictions from ALL 4 models
                lr_proba = self.lr_model.predict_proba(feature_scaled)[0]
                svm_proba = self.svm_model.predict_proba(feature_scaled)[0]
                rf_proba = self.rf_model.predict_proba(feature_scaled)[0]
                xgb_proba = self.xgb_model.predict_proba(feature_scaled)[0]
                
                # ENSEMBLE: Average all predictions (soft voting)
                probabilities = (lr_proba + svm_proba + rf_proba + xgb_proba) / 4
                
                # Store individual predictions for display
                individual_preds = {
                    'Logistic Regression': {
                        'prediction': self.classes[np.argmax(lr_proba)],
                        'confidence': round(float(np.max(lr_proba)) * 100, 2)
                    },
                    'SVM': {
                        'prediction': self.classes[np.argmax(svm_proba)],
                        'confidence': round(float(np.max(svm_proba)) * 100, 2)
                    },
                    'Random Forest': {
                        'prediction': self.classes[np.argmax(rf_proba)],
                        'confidence': round(float(np.max(rf_proba)) * 100, 2)
                    },
                    'XGBoost': {
                        'prediction': self.classes[np.argmax(xgb_proba)],
                        'confidence': round(float(np.max(xgb_proba)) * 100, 2)
                    }
                }
            else:
                # Single model prediction (fallback)
                probabilities = self.model.predict_proba(feature_scaled)[0]
                individual_preds = None
            
            # Get prediction
            pred_idx = np.argmax(probabilities)
            prediction = self.classes[pred_idx]
            confidence = probabilities[pred_idx] * 100
            
            result = {
                'prediction': prediction,
                'confidence': round(float(confidence), 2),
                'probabilities': {
                    self.classes[0]: round(float(probabilities[0]) * 100, 2),
                    self.classes[1]: round(float(probabilities[1]) * 100, 2)
                },
                'model_used': 'Ensemble (LR + SVM + RF + XGBoost)' if self.use_ensemble else 'XGBoost',
                'success': True
            }
            
            # Add individual predictions if ensemble
            if individual_preds:
                result['individual_predictions'] = individual_preds
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_feature_names(self):
        """Return list of required feature names"""
        return self.feature_names

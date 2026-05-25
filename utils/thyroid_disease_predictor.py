"""
Thyroid Disease Predictor — UCI dataset (9,172 patients)
Predicts: Hyperthyroid / Hypothyroid / Negative
Features: age, sex, TT4, T3, T4U, FTI, TSH, pregnant
"""

import os
import pickle
import json
import numpy as np


class ThyroidDiseasePredictor:
    def __init__(self, model_dir='models_thyroid_disease'):
        self.model_dir = model_dir
        self.models = {}
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = []
        self.classes = []
        self._load()

    def _load(self):
        # Load metadata
        meta_path = os.path.join(self.model_dir, 'thyroid_disease_metadata.json')
        with open(meta_path) as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']
        self.classes       = meta['classes']
        self.meta          = meta

        # Load scaler, imputer, encoder
        with open(os.path.join(self.model_dir, 'thyroid_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(self.model_dir, 'thyroid_imputer.pkl'), 'rb') as f:
            self.imputer = pickle.load(f)
        with open(os.path.join(self.model_dir, 'thyroid_label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load individual models
        model_files = {
            'KNN':               'thyroid_knn.pkl',
            'Logistic Regression': 'thyroid_lr.pkl',
            'Random Forest':     'thyroid_rf.pkl',
            'XGBoost':           'thyroid_xgb.pkl',
            'Gradient Boosting': 'thyroid_gb.pkl',
            'Extra Trees':       'thyroid_et.pkl',
            'Voting Ensemble':   'thyroid_voting.pkl',
            'Stacking Ensemble': 'thyroid_stacking.pkl',
        }
        for name, fname in model_files.items():
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)

        print(f'ThyroidDiseasePredictor loaded: {len(self.models)} models')

    def _preprocess(self, data: dict) -> np.ndarray:
        """Convert input dict → scaled numpy array."""
        # sex encoding
        sex_val = data.get('sex', data.get('Sex', 'F'))
        sex_enc = 1 if str(sex_val).upper() in ('M', 'MALE', '1') else 0

        # pregnant encoding
        preg_val = data.get('pregnant', data.get('Pregnant', 'f'))
        preg_enc = 1 if str(preg_val).lower() in ('t', 'true', 'yes', '1') else 0

        def _float(key, default=np.nan):
            v = data.get(key, data.get(key.upper(), default))
            try:
                return float(v) if v not in (None, '', 'nan', 'NaN') else np.nan
            except (ValueError, TypeError):
                return np.nan

        row = np.array([[
            _float('age'),
            sex_enc,
            _float('TT4'),
            _float('T3'),
            _float('T4U'),
            _float('FTI'),
            _float('TSH'),
            preg_enc,
        ]], dtype=float)

        row = self.imputer.transform(row)
        row = self.scaler.transform(row)
        return row

    def predict(self, data: dict) -> dict:
        """
        Predict thyroid disease class.
        Returns dict with prediction, confidence, probabilities, individual model votes.
        """
        X = self._preprocess(data)

        # Use voting ensemble as primary
        primary = self.models.get('Voting Ensemble', self.models.get('XGBoost'))
        proba   = primary.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_class = self.classes[pred_idx]
        confidence = round(float(proba[pred_idx]) * 100, 2)

        # Probabilities for all classes
        probabilities = {cls: round(float(p) * 100, 2)
                         for cls, p in zip(self.classes, proba)}

        # Risk level
        if pred_class == 'Negative':
            risk_level = 'Low'
        elif pred_class == 'Hypothyroid':
            risk_level = 'Medium' if confidence < 85 else 'High'
        else:  # Hyperthyroid
            risk_level = 'Medium' if confidence < 85 else 'High'

        # Individual model votes
        individual = {}
        for name, clf in self.models.items():
            if name in ('Voting Ensemble', 'Stacking Ensemble'):
                continue
            p = clf.predict_proba(X)[0]
            idx = int(np.argmax(p))
            individual[name] = {
                'prediction': self.classes[idx],
                'confidence': round(float(p[idx]) * 100, 2),
            }

        # Model agreement
        votes = [v['prediction'] for v in individual.values()]
        agreement = round(votes.count(pred_class) / len(votes) * 100, 1) if votes else 100.0

        return {
            'prediction':  pred_class,
            'confidence':  confidence,
            'risk_level':  risk_level,
            'probabilities': probabilities,
            'individual_predictions': individual,
            'model_agreement': agreement,
        }

    def get_model_info(self) -> dict:
        return {
            'best_model':        self.meta.get('best_model'),
            'best_accuracy':     round(self.meta.get('best_accuracy', 0), 2),
            'voting_accuracy':   round(self.meta.get('voting_accuracy', 0), 2),
            'voting_auc':        round(self.meta.get('voting_auc', 0), 2),
            'stacking_accuracy': round(self.meta.get('stacking_accuracy', 0), 2),
            'n_models':          len(self.models),
            'dataset_size':      self.meta.get('n_samples'),
            'features':          self.feature_names,
            'classes':           self.classes,
        }

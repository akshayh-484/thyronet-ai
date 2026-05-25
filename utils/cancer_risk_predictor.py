"""
Cancer Risk Predictor — Benign vs Malignant thyroid nodule classification
Trained on clinically realistic synthetic dataset (50,000 patients)
Based on ATA 2016 guidelines and ACR TI-RADS risk factors
"""

import os, pickle, json
import numpy as np


class CancerRiskPredictor:
    def __init__(self, model_dir='models_cancer_risk'):
        self.model_dir = model_dir
        self.models    = {}
        self.scaler    = None
        self.meta      = {}
        self._load()

    def _load(self):
        meta_path = os.path.join(self.model_dir, 'cancer_risk_metadata.json')
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.feature_names = self.meta['feature_names']
        self.classes       = self.meta['classes']          # ['Benign','Malignant']
        self.threshold     = self.meta.get('optimal_threshold', 0.43)

        with open(os.path.join(self.model_dir, 'cancer_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        model_files = {
            'Logistic Regression': 'cancer_lr.pkl',
            'SVM':                 'cancer_svm.pkl',
            'Random Forest':       'cancer_rf.pkl',
            'XGBoost':             'cancer_xgb.pkl',
            'Gradient Boosting':   'cancer_gb.pkl',
            'Extra Trees':         'cancer_et.pkl',
            'Voting Ensemble':     'cancer_voting.pkl',
            'Stacking Ensemble':   'cancer_stacking.pkl',
        }
        for name, fname in model_files.items():
            path = os.path.join(self.model_dir, fname)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)

        print(f'CancerRiskPredictor loaded: {len(self.models)} models')

    # ── Encoding helpers ─────────────────────────────────────────
    COMPOSITION_MAP = {
        'cystic_or_almost_cystic': 0, 'spongiform': 0,
        'mixed_cystic_solid': 1,
        'solid_or_almost_solid': 2,
        '0': 0, '1': 1, '2': 2,
    }
    ECHOGENICITY_MAP = {
        'anechoic': 0,
        'hyperechoic_or_isoechoic': 1,
        'hypoechoic': 2,
        'very_hypoechoic': 3,
        '0': 0, '1': 1, '2': 2, '3': 3,
    }
    SHAPE_MAP = {
        'wider_than_tall': 0, 'horizontal': 0,
        'taller_than_wide': 1, 'vertical': 1,
        '0': 0, '1': 1,
    }
    MARGIN_MAP = {
        'smooth': 0,
        'ill_defined': 1,
        'lobulated_or_irregular': 2, 'irregular': 2,
        'extra_thyroidal_extension': 3,
        '0': 0, '1': 1, '2': 2, '3': 3,
    }

    def _encode(self, data: dict) -> np.ndarray:
        def _f(key, default=0.0):
            v = data.get(key, default)
            try:    return float(v) if v not in (None, '', 'nan') else default
            except: return default

        sex = 1 if str(data.get('Sex', data.get('sex', 'F'))).upper() in ('M','MALE','1') else 0
        fh  = 1 if str(data.get('Family_History', 'No')).lower() in ('yes','1','true') else 0
        rad = 1 if str(data.get('Radiation_Exposure', 'No')).lower() in ('yes','1','true') else 0
        iod = 1 if str(data.get('Iodine_Deficiency', 'No')).lower() in ('yes','1','true') else 0
        smk = 1 if str(data.get('Smoking', 'No')).lower() in ('yes','1','true') else 0
        obs = 1 if str(data.get('Obesity', 'No')).lower() in ('yes','1','true') else 0
        dia = 1 if str(data.get('Diabetes', 'No')).lower() in ('yes','1','true') else 0
        mic = 1 if str(data.get('Microcalcifications', 'No')).lower() in ('yes','1','true') else 0

        comp = self.COMPOSITION_MAP.get(str(data.get('Composition', '1')).lower(), 1)
        echo = self.ECHOGENICITY_MAP.get(str(data.get('Echogenicity', '1')).lower(), 1)
        shp  = self.SHAPE_MAP.get(str(data.get('Shape', '0')).lower(), 0)
        marg = self.MARGIN_MAP.get(str(data.get('Margin', '0')).lower(), 0)

        row = np.array([[
            _f('Age', 45),
            sex, fh, rad, iod, smk, obs, dia,
            _f('TSH_Level', 2.0),
            _f('T3_Level', 2.0),
            _f('T4_Level', 100.0),
            _f('Nodule_Size_cm', 1.0),
            comp, echo, shp, marg, mic,
        ]], dtype=float)
        return self.scaler.transform(row)

    def predict(self, data: dict) -> dict:
        X = self._encode(data)

        # Primary: voting ensemble
        primary = self.models.get('Voting Ensemble', self.models.get('XGBoost'))
        proba   = primary.predict_proba(X)[0]
        mal_prob = float(proba[1])

        # Apply tuned threshold
        pred_idx   = 1 if mal_prob >= self.threshold else 0
        pred_class = self.classes[pred_idx]
        confidence = round(mal_prob * 100 if pred_idx == 1 else (1 - mal_prob) * 100, 2)

        # Risk level
        if mal_prob < 0.20:
            risk_level = 'Low'
        elif mal_prob < 0.50:
            risk_level = 'Intermediate'
        else:
            risk_level = 'High'

        # Individual model votes
        individual = {}
        for name, clf in self.models.items():
            if name in ('Voting Ensemble', 'Stacking Ensemble'):
                continue
            p   = clf.predict_proba(X)[0]
            idx = 1 if float(p[1]) >= self.threshold else 0
            individual[name] = {
                'prediction': self.classes[idx],
                'malignant_prob': round(float(p[1]) * 100, 2),
            }

        votes = [v['prediction'] for v in individual.values()]
        agreement = round(votes.count(pred_class) / len(votes) * 100, 1) if votes else 100.0

        return {
            'success':           True,
            'prediction':        pred_class,
            'confidence':        confidence,
            'malignant_prob':    round(mal_prob * 100, 2),
            'benign_prob':       round((1 - mal_prob) * 100, 2),
            'risk_level':        risk_level,
            'model_agreement':   agreement,
            'individual_predictions': individual,
            'threshold_used':    round(self.threshold, 2),
        }

    def get_model_info(self) -> dict:
        return {
            'best_model':      self.meta.get('best_model'),
            'best_accuracy':   round(self.meta.get('best_accuracy', 0), 2),
            'best_auc':        round(self.meta.get('best_auc', 0), 2),
            'voting_auc':      round(self.meta.get('voting_auc', 0), 2),
            'sensitivity':     round(self.meta.get('sensitivity', 0), 2),
            'specificity':     round(self.meta.get('specificity', 0), 2),
            'n_models':        len(self.models),
            'dataset_size':    self.meta.get('n_samples'),
            'features':        self.feature_names,
            'classes':         self.classes,
            'threshold':       self.threshold,
        }

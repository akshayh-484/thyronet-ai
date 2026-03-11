"""
Advanced ML/DL Training Pipeline with Class Imbalance Handling
Trains multiple models and selects the best performer
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, csv_path='thyroid_cancer_risk_data.csv'):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
        
    def load_and_preprocess(self):
        """Load CSV and preprocess data"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        
        # Detect target column (binary classification)
        target_col = 'Diagnosis'
        print(f"✅ Target column detected: {target_col}")
        
        # Separate features and target
        X = self.df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
        y = self.df[target_col]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"📈 Dataset shape: {X.shape}")
        print(f"🎯 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Check class imbalance
        class_counts = np.bincount(y)
        imbalance_ratio = max(class_counts) / min(class_counts)
        print(f"⚖️  Imbalance ratio: {imbalance_ratio:.2f}")
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE if imbalanced
        if imbalance_ratio > 1.5:
            print("🔄 Applying SMOTE to balance dataset...")
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            print(f"✅ After SMOTE: {dict(zip(*np.unique(self.y_train, return_counts=True)))}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✅ Preprocessing complete!\n")
        
    def train_ml_models(self):
        """Train multiple ML models"""
        print("🤖 Training Machine Learning Models...\n")
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}\n")
            
            # Track best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
                self.best_model_name = name
    
    def train_deep_learning_model(self):
        """Train Dense Neural Network"""
        print("🧠 Training Deep Learning Model...\n")
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=(self.X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(self.X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.results['Dense Neural Network'] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': accuracy,
            'cv_std': 0.0
        }
        
        print(f"Training Dense Neural Network...")
        print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
        
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model
            self.best_model_name = 'Dense Neural Network'
    
    def generate_reports(self):
        """Generate evaluation reports"""
        print("📊 Generating Reports...\n")
        
        # Best model predictions
        if self.best_model_name == 'Dense Neural Network':
            y_pred_proba = self.best_model.predict(self.X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                    target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Confusion matrix saved: confusion_matrix.png")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.best_model_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✅ ROC curve saved: roc_curve.png\n")
        
    def save_models(self):
        """Save best model and preprocessing objects"""
        print("💾 Saving models...\n")
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save feature names
        with open('models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save best model
        if self.best_model_name == 'Dense Neural Network':
            self.best_model.save('models/best_model.h5')
            model_type = 'keras'
        else:
            with open('models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
            model_type = 'sklearn'
        
        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'model_type': model_type,
            'accuracy': self.best_score,
            'feature_names': self.feature_names,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Best Model: {self.best_model_name}")
        print(f"✅ Accuracy: {self.best_score:.4f}")
        print(f"✅ Model saved to: models/")
        
        # Print all results
        print("\n📊 All Model Results:")
        print("-" * 80)
        for name, metrics in self.results.items():
            print(f"{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        print("-" * 80)
    
    def run(self):
        """Execute full training pipeline"""
        self.load_and_preprocess()
        self.train_ml_models()
        self.train_deep_learning_model()
        self.generate_reports()
        self.save_models()

if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)
    
    trainer = ModelTrainer()
    trainer.run()

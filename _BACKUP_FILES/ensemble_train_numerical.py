"""
ENSEMBLE LEARNING SYSTEM FOR NUMERICAL DATA
Combines all 4 models for superior performance
Uses Voting, Stacking, and Weighted Ensemble methods
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class EnsembleNumericalTrainer:
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
        
        # Individual models
        self.lr_model = None
        self.svm_model = None
        self.rf_model = None
        self.xgb_model = None
        
        # Ensemble models
        self.voting_ensemble = None
        self.stacking_ensemble = None
        self.weighted_ensemble = None
        
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
        print("="*80)
        print("🎯 ENSEMBLE LEARNING SYSTEM - NUMERICAL DATA")
        print("="*80)
        print("Combining 4 ML models for superior performance")
        print("Methods: Voting, Stacking, Weighted Ensemble")
        print("="*80 + "\n")

    def load_and_preprocess(self):
        """Load and preprocess data"""
        print("📊 STEP 1: DATA LOADING & PREPROCESSING")
        print("-"*80)
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Target column
        target_col = 'Diagnosis'
        
        # Separate features and target
        X = self.df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
        y = self.df[target_col]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Handle imbalance
        unique, counts = np.unique(self.y_train, return_counts=True)
        imbalance_ratio = counts.max() / counts.min()
        
        if imbalance_ratio > 1.5:
            print(f"🔄 Applying SMOTETomek (imbalance ratio: {imbalance_ratio:.2f})")
            smote_tomek = SMOTETomek(random_state=42)
            self.X_train, self.y_train = smote_tomek.fit_resample(self.X_train, self.y_train)
            print(f"✅ Dataset balanced")
        
        # Scale features
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✅ Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        print("\n" + "="*80 + "\n")
    
    def train_base_models(self):
        """Train individual base models with optimized parameters"""
        print("🤖 STEP 2: TRAINING BASE MODELS")
        print("-"*80)
        
        # Logistic Regression
        print("Training Logistic Regression...")
        self.lr_model = LogisticRegression(
            C=1.0, penalty='l2', solver='liblinear',
            max_iter=1000, random_state=42, class_weight='balanced'
        )
        self.lr_model.fit(self.X_train, self.y_train)
        self._evaluate_model('Logistic Regression', self.lr_model)
        
        # SVM
        print("\nTraining SVM...")
        self.svm_model = SVC(
            C=10, kernel='rbf', gamma='scale', probability=True,
            random_state=42, class_weight='balanced'
        )
        self.svm_model.fit(self.X_train, self.y_train)
        self._evaluate_model('SVM', self.svm_model)
        
        # Random Forest
        print("\nTraining Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=2,
            min_samples_leaf=1, max_features='sqrt',
            random_state=42, class_weight='balanced', n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train)
        self._evaluate_model('Random Forest', self.rf_model)
        
        # XGBoost
        print("\nTraining XGBoost...")
        unique, counts = np.unique(self.y_train, return_counts=True)
        scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1
        
        self.xgb_model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
        self.xgb_model.fit(self.X_train, self.y_train)
        self._evaluate_model('XGBoost', self.xgb_model)
        
        print("\n" + "="*80 + "\n")

    def create_voting_ensemble(self):
        """Create Voting Classifier (Hard & Soft voting)"""
        print("🎯 STEP 3: CREATING VOTING ENSEMBLE")
        print("-"*80)
        
        # Soft Voting (uses predicted probabilities)
        print("Creating Soft Voting Ensemble...")
        self.voting_ensemble = VotingClassifier(
            estimators=[
                ('lr', self.lr_model),
                ('svm', self.svm_model),
                ('rf', self.rf_model),
                ('xgb', self.xgb_model)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        self.voting_ensemble.fit(self.X_train, self.y_train)
        self._evaluate_model('Voting Ensemble (Soft)', self.voting_ensemble)
        
        print("\n" + "="*80 + "\n")
    
    def create_stacking_ensemble(self):
        """Create Stacking Classifier with meta-learner"""
        print("🎯 STEP 4: CREATING STACKING ENSEMBLE")
        print("-"*80)
        
        print("Creating Stacking Ensemble with Logistic Regression meta-learner...")
        
        # Base estimators
        base_estimators = [
            ('lr', self.lr_model),
            ('svm', self.svm_model),
            ('rf', self.rf_model),
            ('xgb', self.xgb_model)
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        self.stacking_ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        self.stacking_ensemble.fit(self.X_train, self.y_train)
        self._evaluate_model('Stacking Ensemble', self.stacking_ensemble)
        
        print("\n" + "="*80 + "\n")
    
    def create_weighted_ensemble(self):
        """Create custom weighted ensemble based on individual model performance"""
        print("🎯 STEP 5: CREATING WEIGHTED ENSEMBLE")
        print("-"*80)
        
        # Calculate weights based on ROC-AUC scores
        weights = {}
        total_score = 0
        
        for name in ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost']:
            if name in self.results:
                score = self.results[name]['roc_auc']
                weights[name] = score
                total_score += score
        
        # Normalize weights
        for name in weights:
            weights[name] = weights[name] / total_score
        
        print("Model Weights (based on ROC-AUC):")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Create weighted predictions
        y_pred_proba_weighted = np.zeros((len(self.X_test), 2))
        
        models = {
            'Logistic Regression': self.lr_model,
            'SVM': self.svm_model,
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model
        }
        
        for name, model in models.items():
            if name in weights:
                proba = model.predict_proba(self.X_test)
                y_pred_proba_weighted += proba * weights[name]
        
        y_pred_weighted = np.argmax(y_pred_proba_weighted, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred_weighted)
        precision = precision_score(self.y_test, y_pred_weighted, zero_division=0)
        recall = recall_score(self.y_test, y_pred_weighted, zero_division=0)
        f1 = f1_score(self.y_test, y_pred_weighted, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba_weighted[:, 1])
        
        self.results['Weighted Ensemble'] = {
            'model': 'weighted',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred_weighted,
            'y_pred_proba': y_pred_proba_weighted[:, 1],
            'weights': weights
        }
        
        print(f"\n📊 Weighted Ensemble Results:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        if roc_auc > self.best_score:
            self.best_score = roc_auc
            self.best_model = 'weighted'
            self.best_model_name = 'Weighted Ensemble'
            print(f"   🏆 NEW BEST MODEL!")
        
        print("\n" + "="*80 + "\n")

    def _evaluate_model(self, name, model):
        """Evaluate individual model"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        if roc_auc > self.best_score:
            self.best_score = roc_auc
            self.best_model = model
            self.best_model_name = name
            print(f"   🏆 NEW BEST MODEL!")
    
    def compare_all_models(self):
        """Compare all models including ensembles"""
        print("📊 STEP 6: COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        df_comparison.to_csv('ensemble_comparison.csv', index=False)
        
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"{'='*80}")
        best_metrics = self.results[self.best_model_name]
        print(f"   Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall:    {best_metrics['recall']:.4f}")
        print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")
        print(f"{'='*80}\n")
    
    def generate_visualizations(self):
        """Generate ensemble comparison visualizations"""
        print("📈 STEP 7: GENERATING VISUALIZATIONS")
        print("-"*80)
        
        # 1. Model Comparison Bar Chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            model_names = list(self.results.keys())
            values = [self.results[name][metric] for name in model_names]
            
            colors = ['#2ecc71' if name == self.best_model_name else '#3498db' for name in model_names]
            
            bars = axes[idx].bar(range(len(model_names)), values, color=colors, alpha=0.8, edgecolor='black')
            axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].set_xticks(range(len(model_names)))
            axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
            axes[idx].grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide last subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig('ensemble_comparison_metrics.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: ensemble_comparison_metrics.png")
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(12, 8))
        
        for name, metrics in self.results.items():
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            
            linestyle = '-' if name == self.best_model_name else '--'
            linewidth = 3 if name == self.best_model_name else 2
            
            plt.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
                    label=f'{name} (AUC = {metrics["roc_auc"]:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models + Ensembles', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(alpha=0.3)
        plt.savefig('ensemble_roc_curves.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: ensemble_roc_curves.png")
        plt.close()
        
        print("\n" + "="*80 + "\n")

    def save_ensemble_models(self):
        """Save all models and ensemble configurations"""
        print("💾 STEP 8: SAVING ENSEMBLE MODELS")
        print("-"*80)
        
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save individual models
        with open('models/ensemble_lr.pkl', 'wb') as f:
            pickle.dump(self.lr_model, f)
        print("✅ Saved: models/ensemble_lr.pkl")
        
        with open('models/ensemble_svm.pkl', 'wb') as f:
            pickle.dump(self.svm_model, f)
        print("✅ Saved: models/ensemble_svm.pkl")
        
        with open('models/ensemble_rf.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        print("✅ Saved: models/ensemble_rf.pkl")
        
        with open('models/ensemble_xgb.pkl', 'wb') as f:
            pickle.dump(self.xgb_model, f)
        print("✅ Saved: models/ensemble_xgb.pkl")
        
        # Save ensemble models
        with open('models/voting_ensemble.pkl', 'wb') as f:
            pickle.dump(self.voting_ensemble, f)
        print("✅ Saved: models/voting_ensemble.pkl")
        
        with open('models/stacking_ensemble.pkl', 'wb') as f:
            pickle.dump(self.stacking_ensemble, f)
        print("✅ Saved: models/stacking_ensemble.pkl")
        
        # Save weighted ensemble configuration
        if 'Weighted Ensemble' in self.results:
            weighted_config = {
                'weights': self.results['Weighted Ensemble']['weights'],
                'model_names': list(self.results['Weighted Ensemble']['weights'].keys())
            }
            with open('models/weighted_ensemble_config.json', 'w') as f:
                json.dump(weighted_config, f, indent=2)
            print("✅ Saved: models/weighted_ensemble_config.json")
        
        # Save scaler and encoder
        with open('models/ensemble_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Saved: models/ensemble_scaler.pkl")
        
        with open('models/ensemble_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✅ Saved: models/ensemble_label_encoder.pkl")
        
        # Save feature names
        with open('models/ensemble_feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print("✅ Saved: models/ensemble_feature_names.json")
        
        # Save comprehensive metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_model_accuracy': float(self.results[self.best_model_name]['accuracy']),
            'best_model_roc_auc': float(self.results[self.best_model_name]['roc_auc']),
            'ensemble_type': 'voting_stacking_weighted',
            'feature_names': self.feature_names,
            'classes': self.label_encoder.classes_.tolist(),
            'n_features': len(self.feature_names),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'all_models': {
                name: {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall']),
                    'f1_score': float(metrics['f1_score']),
                    'roc_auc': float(metrics['roc_auc'])
                }
                for name, metrics in self.results.items()
            }
        }
        
        with open('models/ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Saved: models/ensemble_metadata.json")
        
        print("\n" + "="*80 + "\n")
    
    def run_complete_pipeline(self):
        """Execute complete ensemble training pipeline"""
        start_time = datetime.now()
        
        try:
            self.load_and_preprocess()
            self.train_base_models()
            self.create_voting_ensemble()
            self.create_stacking_ensemble()
            self.create_weighted_ensemble()
            self.compare_all_models()
            self.generate_visualizations()
            self.save_ensemble_models()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("="*80)
            print("🎉 ENSEMBLE TRAINING COMPLETE!")
            print("="*80)
            print(f"⏱️  Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"🏆 Best Model: {self.best_model_name}")
            print(f"📊 Best Accuracy: {self.results[self.best_model_name]['accuracy']*100:.2f}%")
            print(f"📊 Best ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f}")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    trainer = EnsembleNumericalTrainer('thyroid_cancer_risk_data.csv')
    trainer.run_complete_pipeline()

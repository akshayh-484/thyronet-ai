"""
COMPETITION-LEVEL NUMERICAL CLASSIFICATION SYSTEM
Advanced ML Pipeline with Imbalance Handling & Hyperparameter Tuning
Target: 85%+ Accuracy with Balanced Performance
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Imbalance Handling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, auc, make_scorer
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

class AdvancedNumericalTrainer:
    def __init__(self, csv_path='thyroid_cancer_risk_data.csv', target_accuracy=0.85):
        self.csv_path = csv_path
        self.target_accuracy = target_accuracy
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
        print("="*80)
        print("🏆 COMPETITION-LEVEL MEDICAL CLASSIFICATION SYSTEM")
        print("="*80)
        print(f"Target Accuracy: {target_accuracy*100}%+")
        print(f"Dataset: {csv_path}")
        print("="*80 + "\n")
    
    def load_and_explore(self):
        """Load data and perform exploratory analysis"""
        print("📊 STEP 1: DATA LOADING & EXPLORATION")
        print("-"*80)
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Display basic info
        print(f"\n📋 Dataset Info:")
        print(f"   Columns: {list(self.df.columns)}")
        print(f"   Memory usage: {self.df.memory_usage().sum() / 1024:.2f} KB")
        
        # Check for target column
        target_col = 'Diagnosis'
        if target_col not in self.df.columns:
            # Try to find binary column
            for col in self.df.columns:
                if self.df[col].nunique() == 2:
                    target_col = col
                    break
        
        print(f"   Target column: {target_col}")
        
        # Class distribution
        print(f"\n📈 Class Distribution (BEFORE balancing):")
        class_counts = self.df[target_col].value_counts()
        for cls, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {cls}: {count} ({percentage:.2f}%)")
        
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"   ⚖️  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print(f"   ⚠️  HIGHLY IMBALANCED - Will apply SMOTE")
        
        # Missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values Detected:")
            for col, count in missing[missing > 0].items():
                print(f"   {col}: {count} ({count/len(self.df)*100:.2f}%)")
        else:
            print(f"\n✅ No missing values detected")
        
        print("\n" + "="*80 + "\n")
        return target_col

    def preprocess_data(self, target_col):
        """Advanced preprocessing with robust scaling"""
        print("🔧 STEP 2: DATA PREPROCESSING")
        print("-"*80)
        
        # Separate features and target
        X = self.df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
        y = self.df[target_col]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print(f"📝 Encoding categorical features: {categorical_cols}")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"✅ Target classes: {self.label_encoder.classes_}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        print(f"✅ Features: {len(self.feature_names)} numerical features")
        
        # Handle missing values with median (robust to outliers)
        if X.isnull().sum().sum() > 0:
            print(f"🔧 Filling missing values with median...")
            X = X.fillna(X.median())
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"✅ Train set: {self.X_train.shape[0]} samples")
        print(f"✅ Test set: {self.X_test.shape[0]} samples")
        
        print("\n" + "="*80 + "\n")
    
    def handle_imbalance(self):
        """Apply advanced imbalance handling techniques"""
        print("⚖️  STEP 3: IMBALANCE HANDLING")
        print("-"*80)
        
        # Check imbalance
        unique, counts = np.unique(self.y_train, return_counts=True)
        imbalance_ratio = counts.max() / counts.min()
        
        print(f"📊 Class distribution (BEFORE):")
        for cls, count in zip(unique, counts):
            print(f"   Class {self.label_encoder.classes_[cls]}: {count}")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print(f"\n🔄 Applying SMOTETomek (SMOTE + Tomek Links)...")
            print(f"   - SMOTE: Synthetic oversampling of minority class")
            print(f"   - Tomek: Remove noisy borderline samples")
            
            smote_tomek = SMOTETomek(random_state=42)
            self.X_train, self.y_train = smote_tomek.fit_resample(self.X_train, self.y_train)
            
            unique, counts = np.unique(self.y_train, return_counts=True)
            print(f"\n📊 Class distribution (AFTER SMOTETomek):")
            for cls, count in zip(unique, counts):
                print(f"   Class {self.label_encoder.classes_[cls]}: {count}")
            print(f"   ✅ Dataset balanced!")
        else:
            print(f"✅ Dataset already balanced (ratio < 1.5)")
        
        print("\n" + "="*80 + "\n")

    def scale_features(self):
        """Apply robust scaling"""
        print("📏 STEP 4: FEATURE SCALING")
        print("-"*80)
        
        print("🔧 Applying RobustScaler (resistant to outliers)...")
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✅ Features scaled successfully")
        print("\n" + "="*80 + "\n")
    
    def train_logistic_regression(self):
        """Train Logistic Regression with hyperparameter tuning"""
        print("🤖 MODEL 1: LOGISTIC REGRESSION")
        print("-"*80)
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
        
        print("🔍 Hyperparameter tuning with GridSearchCV...")
        print(f"   Parameter grid: {len(param_grid['C']) * len(param_grid['penalty']) * len(param_grid['solver'])} combinations")
        
        lr = LogisticRegression(random_state=42, class_weight='balanced')
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['Logistic Regression'] = best_model
        self._evaluate_model('Logistic Regression', best_model)
        
        print("\n" + "="*80 + "\n")
    
    def train_svm(self):
        """Train Support Vector Machine with hyperparameter tuning"""
        print("🤖 MODEL 2: SUPPORT VECTOR MACHINE (SVM)")
        print("-"*80)
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'degree': [2, 3]
        }
        
        print("🔍 Hyperparameter tuning with GridSearchCV...")
        print(f"   Testing multiple kernels and regularization strengths...")
        
        svm = SVC(probability=True, random_state=42, class_weight='balanced')
        
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['SVM'] = best_model
        self._evaluate_model('SVM', best_model)
        
        print("\n" + "="*80 + "\n")

    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("🤖 MODEL 3: RANDOM FOREST")
        print("-"*80)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print("🔍 Hyperparameter tuning with GridSearchCV...")
        print(f"   Optimizing tree depth, number of estimators, and split criteria...")
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['Random Forest'] = best_model
        self._evaluate_model('Random Forest', best_model)
        
        print("\n" + "="*80 + "\n")
    
    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        print("🤖 MODEL 4: XGBOOST")
        print("-"*80)
        
        # Calculate scale_pos_weight for imbalance
        unique, counts = np.unique(self.y_train, return_counts=True)
        scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        print("🔍 Hyperparameter tuning with GridSearchCV...")
        print(f"   Optimizing learning rate, tree depth, and regularization...")
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")
        
        xgb_model = XGBClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        print(f"✅ Best parameters: {grid_search.best_params_}")
        print(f"✅ Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        self.models['XGBoost'] = best_model
        self._evaluate_model('XGBoost', best_model)
        
        print("\n" + "="*80 + "\n")

    def _evaluate_model(self, name, model):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
        
        # Store results
        self.results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\n📊 Evaluation Results:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   CV Score:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Check overfitting
        train_score = model.score(self.X_train, self.y_train)
        test_score = accuracy
        overfit_gap = train_score - test_score
        
        if overfit_gap > 0.05:
            print(f"   ⚠️  Overfitting detected: {overfit_gap:.4f} gap")
        else:
            print(f"   ✅ No significant overfitting: {overfit_gap:.4f} gap")
        
        # Update best model
        if roc_auc > self.best_score:
            self.best_score = roc_auc
            self.best_model = model
            self.best_model_name = name
            print(f"   🏆 NEW BEST MODEL!")

    def compare_models(self):
        """Generate comprehensive model comparison"""
        print("📊 STEP 5: MODEL COMPARISON")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'CV Score': f"{metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        # Save comparison
        df_comparison.to_csv('model_comparison.csv', index=False)
        print(f"\n✅ Comparison saved to: model_comparison.csv")
        
        # Best model summary
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"{'='*80}")
        best_metrics = self.results[self.best_model_name]
        print(f"   Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {best_metrics['precision']:.4f}")
        print(f"   Recall:    {best_metrics['recall']:.4f}")
        print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")
        
        if best_metrics['accuracy'] >= self.target_accuracy:
            print(f"\n   ✅ TARGET ACHIEVED: {best_metrics['accuracy']*100:.2f}% >= {self.target_accuracy*100}%")
        else:
            print(f"\n   ⚠️  Below target: {best_metrics['accuracy']*100:.2f}% < {self.target_accuracy*100}%")
        
        print(f"{'='*80}\n")
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("📈 STEP 6: GENERATING VISUALIZATIONS")
        print("-"*80)
        
        # 1. Confusion Matrices for all models
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            axes[idx].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.4f}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices_all.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: confusion_matrices_all.png")
        plt.close()
        
        # 2. ROC Curves for all models
        plt.figure(figsize=(12, 8))
        
        for name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, metrics['y_pred_proba'])
            roc_auc = metrics['roc_auc']
            
            linestyle = '-' if name == self.best_model_name else '--'
            linewidth = 3 if name == self.best_model_name else 2
            
            plt.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
                    label=f'{name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig('roc_curves_all.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: roc_curves_all.png")
        plt.close()

        # 3. Model Performance Comparison Bar Chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cv_mean']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            model_names = list(self.results.keys())
            values = [self.results[name][metric] for name in model_names]
            
            colors = ['#2ecc71' if name == self.best_model_name else '#3498db' for name in model_names]
            
            bars = axes[idx].bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')
            axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: model_comparison_metrics.png")
        plt.close()
        
        # 4. Best Model - Detailed Confusion Matrix
        best_metrics = self.results[self.best_model_name]
        cm = confusion_matrix(self.y_test, best_metrics['y_pred'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   linewidths=2, linecolor='black')
        plt.title(f'Confusion Matrix - {self.best_model_name}\nAccuracy: {best_metrics["accuracy"]:.4f}',
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.savefig('best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: best_model_confusion_matrix.png")
        plt.close()
        
        print("\n" + "="*80 + "\n")
    
    def generate_classification_reports(self):
        """Generate detailed classification reports"""
        print("📋 STEP 7: CLASSIFICATION REPORTS")
        print("-"*80)
        
        for name, metrics in self.results.items():
            print(f"\n{name}:")
            print("-" * 60)
            report = classification_report(
                self.y_test, metrics['y_pred'],
                target_names=self.label_encoder.classes_,
                digits=4
            )
            print(report)
            
            # Save to file
            with open(f'classification_report_{name.replace(" ", "_").lower()}.txt', 'w') as f:
                f.write(f"Classification Report - {name}\n")
                f.write("="*60 + "\n\n")
                f.write(report)
            
            print(f"✅ Saved: classification_report_{name.replace(' ', '_').lower()}.txt")
        
        print("\n" + "="*80 + "\n")

    def save_models(self):
        """Save all models and artifacts"""
        print("💾 STEP 8: SAVING MODELS & ARTIFACTS")
        print("-"*80)
        
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ Saved: models/scaler.pkl")
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("✅ Saved: models/label_encoder.pkl")
        
        # Save feature names
        with open('models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print("✅ Saved: models/feature_names.json")
        
        # Save all models
        for name, metrics in self.results.items():
            model_filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(metrics['model'], f)
            print(f"✅ Saved: {model_filename}")
        
        # Save best model separately
        with open('models/best_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✅ Saved: models/best_model.pkl (Best: {self.best_model_name})")
        
        # Save metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'best_model_accuracy': float(self.results[self.best_model_name]['accuracy']),
            'best_model_roc_auc': float(self.results[self.best_model_name]['roc_auc']),
            'target_accuracy': self.target_accuracy,
            'target_achieved': self.results[self.best_model_name]['accuracy'] >= self.target_accuracy,
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
        
        with open('models/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✅ Saved: models/metadata.json")
        
        print("\n" + "="*80 + "\n")
    
    def run_complete_pipeline(self):
        """Execute the complete training pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Load and explore
            target_col = self.load_and_explore()
            
            # Step 2: Preprocess
            self.preprocess_data(target_col)
            
            # Step 3: Handle imbalance
            self.handle_imbalance()
            
            # Step 4: Scale features
            self.scale_features()
            
            # Step 5: Train models
            self.train_logistic_regression()
            self.train_svm()
            self.train_random_forest()
            self.train_xgboost()
            
            # Step 6: Compare models
            self.compare_models()
            
            # Step 7: Generate visualizations
            self.generate_visualizations()
            
            # Step 8: Classification reports
            self.generate_classification_reports()
            
            # Step 9: Save everything
            self.save_models()
            
            # Final summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("="*80)
            print("🎉 TRAINING COMPLETE!")
            print("="*80)
            print(f"⏱️  Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"🏆 Best Model: {self.best_model_name}")
            print(f"📊 Best Accuracy: {self.results[self.best_model_name]['accuracy']*100:.2f}%")
            print(f"📊 Best ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f}")
            
            if self.results[self.best_model_name]['accuracy'] >= self.target_accuracy:
                print(f"✅ TARGET ACHIEVED: {self.results[self.best_model_name]['accuracy']*100:.2f}% >= {self.target_accuracy*100}%")
            else:
                print(f"⚠️  Target not met: {self.results[self.best_model_name]['accuracy']*100:.2f}% < {self.target_accuracy*100}%")
            
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    trainer = AdvancedNumericalTrainer(
        csv_path='thyroid_cancer_risk_data.csv',
        target_accuracy=0.85
    )
    trainer.run_complete_pipeline()

"""
PROPER TRAINING - Full Dataset with High Accuracy
Uses full 212K dataset for maximum accuracy (95%+)
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTETomek
import os

print("="*80)
print("🎯 PROPER TRAINING - FULL DATASET")
print("="*80)
print("This will use the complete 212K dataset for maximum accuracy")
print("Expected time: 10-15 minutes")
print("Expected accuracy: 95%+")
print("="*80)

# Load full data
print("\n📊 Loading full dataset...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")

# Prepare features
target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

print(f"\n📋 Features: {len(X.columns)}")
print(f"   {', '.join(X.columns.tolist())}")

# Store original feature names and types
feature_info = {}
for col in X.columns:
    feature_info[col] = {
        'dtype': str(X[col].dtype),
        'unique_values': X[col].nunique(),
        'is_categorical': X[col].dtype == 'object'
    }

# Encode categorical features with proper mapping
print("\n🔤 Encoding categorical features...")
categorical_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    categorical_encoders[col] = {
        'encoder': le,
        'classes': le.classes_.tolist(),
        'mapping': {cls: idx for idx, cls in enumerate(le.classes_)}
    }
    print(f"   ✅ {col}: {len(le.classes_)} categories")

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\n🎯 Target classes: {label_encoder.classes_.tolist()}")

feature_names = X.columns.tolist()

# Handle missing values
print("\n🔧 Handling missing values...")
X = X.fillna(X.median())

# Train-test split
print("\n✂️  Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Train: {len(X_train):,} samples")
print(f"   Test:  {len(X_test):,} samples")

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n⚖️  Class distribution:")
for cls, count in zip(label_encoder.classes_, counts):
    print(f"   {cls}: {count:,} ({count/len(y_train)*100:.1f}%)")

# Handle imbalance
imbalance_ratio = counts.max() / counts.min()
if imbalance_ratio > 1.5:
    print(f"\n🔄 Applying SMOTETomek (imbalance ratio: {imbalance_ratio:.2f})...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
    print(f"   ✅ Balanced: {len(X_train):,} training samples")

# Scale features
print("\n📏 Scaling features with RobustScaler...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*80)
print("🤖 TRAINING MODELS (This will take 10-15 minutes)")
print("="*80)

models = {}
results = {}

# 1. Random Forest (Best for this dataset)
print("\n1️⃣  Training Random Forest (5-10 minutes)...")
print("   Parameters: 300 trees, max_depth=25, balanced weights")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    verbose=1
)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Random Forest'] = rf
print(f"\n   ✅ Random Forest Results:")
print(f"      Accuracy:  {results['Random Forest']['accuracy']*100:.2f}%")
print(f"      Precision: {results['Random Forest']['precision']:.4f}")
print(f"      Recall:    {results['Random Forest']['recall']:.4f}")
print(f"      F1-Score:  {results['Random Forest']['f1_score']:.4f}")
print(f"      ROC-AUC:   {results['Random Forest']['roc_auc']:.4f}")

# 2. XGBoost
print("\n2️⃣  Training XGBoost (5-10 minutes)...")
print("   Parameters: 300 estimators, max_depth=7, learning_rate=0.1")

unique_train, counts_train = np.unique(y_train, return_counts=True)
scale_pos_weight = counts_train[0] / counts_train[1] if len(counts_train) > 1 else 1

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=1
)
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)
y_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]

results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['XGBoost'] = xgb
print(f"\n   ✅ XGBoost Results:")
print(f"      Accuracy:  {results['XGBoost']['accuracy']*100:.2f}%")
print(f"      Precision: {results['XGBoost']['precision']:.4f}")
print(f"      Recall:    {results['XGBoost']['recall']:.4f}")
print(f"      F1-Score:  {results['XGBoost']['f1_score']:.4f}")
print(f"      ROC-AUC:   {results['XGBoost']['roc_auc']:.4f}")

# Find best model
print("\n" + "="*80)
print("📊 MODEL COMPARISON")
print("="*80)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_model = models[best_model_name]

print("\n" + "="*80)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"📊 Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"📊 ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print("="*80)

# Save models
print("\n💾 Saving models and encoders...")
os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✅ Saved: models/best_model.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: models/scaler.pkl")

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("✅ Saved: models/label_encoder.pkl")

# Save categorical encoders
with open('models/categorical_encoders.pkl', 'wb') as f:
    pickle.dump(categorical_encoders, f)
print("✅ Saved: models/categorical_encoders.pkl")

# Save metadata with encoding information
metadata = {
    'best_model_name': best_model_name,
    'best_model_accuracy': float(results[best_model_name]['accuracy']),
    'best_model_roc_auc': float(results[best_model_name]['roc_auc']),
    'feature_names': feature_names,
    'feature_info': feature_info,
    'categorical_encoders': {
        col: data['mapping'] for col, data in categorical_encoders.items()
    },
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'all_models': {
        name: {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc'])
        }
        for name, metrics in results.items()
    }
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: models/metadata.json")

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)
print(f"✅ Best Model: {best_model_name}")
print(f"✅ Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"✅ ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"✅ Trained on: {len(X_train):,} samples")
print(f"✅ All models saved to 'models/' folder")
print("\n🚀 Next: Restart web app with 'python app_enhanced.py'")
print("="*80)

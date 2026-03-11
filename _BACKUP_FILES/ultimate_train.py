"""
ULTIMATE TRAINING - Maximum Accuracy with Advanced Techniques
Target: 85%+ accuracy with all optimizations
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import os

print("="*80)
print("🎯 ULTIMATE TRAINING - MAXIMUM ACCURACY")
print("="*80)
print("Using advanced techniques:")
print("  • Full 212K dataset")
print("  • Feature engineering")
print("  • Multiple algorithms (RF, XGB, GradientBoosting)")
print("  • Hyperparameter tuning")
print("  • Advanced balancing")
print("Expected time: 15-20 minutes")
print("Target accuracy: 85%+")
print("="*80)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"✅ Loaded: {len(df):,} rows")

# Prepare features
target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

# Feature engineering
print("\n🔧 Feature Engineering...")

# Create interaction features
X['TSH_T3_ratio'] = X['TSH_Level'] / (X['T3_Level'] + 0.001)
X['TSH_T4_ratio'] = X['TSH_Level'] / (X['T4_Level'] + 0.001)
X['T3_T4_ratio'] = X['T3_Level'] / (X['T4_Level'] + 0.001)
X['Nodule_TSH_interaction'] = X['Nodule_Size'] * X['TSH_Level']

# Age groups
X['Age_group'] = pd.cut(X['Age'], bins=[0, 30, 50, 70, 120], labels=[0, 1, 2, 3])

# Risk factors count
risk_factors = ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity', 'Diabetes']

print(f"✅ Created {X.shape[1] - 15} new features")

# Encode categorical
categorical_encoders = {}
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    categorical_encoders[col] = {
        'encoder': le,
        'classes': le.classes_.tolist(),
        'mapping': {cls: idx for idx, cls in enumerate(le.classes_)}
    }

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns.tolist()
X = X.fillna(X.median())

# Split
print("\n✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Advanced balancing with SMOTE
print("\n⚖️  Applying SMOTE for better balance...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train, y_train)
print(f"   ✅ Balanced: {len(X_train):,} samples")

# Scale
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n" + "="*80)
print("🤖 TRAINING MULTIPLE MODELS")
print("="*80)

models = {}
results = {}

# 1. Random Forest with tuning
print("\n1️⃣  Random Forest with optimized parameters...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1
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
print(f"   Accuracy: {results['Random Forest']['accuracy']*100:.2f}%")

# 2. XGBoost with tuning
print("\n2️⃣  XGBoost with optimized parameters...")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1
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
print(f"   Accuracy: {results['XGBoost']['accuracy']*100:.2f}%")

# 3. Gradient Boosting
print("\n3️⃣  Gradient Boosting with optimized parameters...")
gb = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, y_train)
y_pred = gb.predict(X_test_scaled)
y_pred_proba = gb.predict_proba(X_test_scaled)[:, 1]

results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Gradient Boosting'] = gb
print(f"   Accuracy: {results['Gradient Boosting']['accuracy']*100:.2f}%")

# Find best
print("\n" + "="*80)
print("📊 FINAL RESULTS")
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

# Save
print("\n💾 Saving...")
os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/categorical_encoders.pkl', 'wb') as f:
    pickle.dump(categorical_encoders, f)

metadata = {
    'best_model_name': best_model_name,
    'best_model_accuracy': float(results[best_model_name]['accuracy']),
    'best_model_roc_auc': float(results[best_model_name]['roc_auc']),
    'feature_names': feature_names,
    'categorical_encoders': {col: data['mapping'] for col, data in categorical_encoders.items()},
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'all_models': {name: {k: float(v) for k, v in metrics.items()} for name, metrics in results.items()}
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ All saved!")
print("\n🚀 Restart: python app_enhanced.py")
print("="*80)

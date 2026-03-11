"""
Fast Training Script - Trains models quickly without extensive hyperparameter tuning
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTETomek
import os

print("="*80)
print("🚀 FAST TRAINING - NUMERICAL MODELS")
print("="*80)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

# Sample for speed (use 10k rows for very fast training)
if len(df) > 10000:
    from sklearn.model_selection import train_test_split
    df, _ = train_test_split(df, train_size=10000, random_state=42, stratify=df['Diagnosis'])
    print(f"✅ Sampled to {len(df)} rows for faster training")

# Prepare features
target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns.tolist()

# Handle missing values
X = X.fillna(X.median())

# Train-test split
print("\n✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Handle imbalance
print("\n⚖️  Balancing dataset with SMOTETomek...")
smote_tomek = SMOTETomek(random_state=42)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
print(f"✅ Balanced: {len(X_train)} training samples")

# Scale features
print("\n📏 Scaling features...")
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
print("\n" + "="*80)
print("🤖 TRAINING MODELS")
print("="*80)

models = {}
results = {}

# 1. Logistic Regression
print("\n1️⃣  Training Logistic Regression...")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:, 1]
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Logistic Regression'] = lr
print(f"   ✅ Accuracy: {results['Logistic Regression']['accuracy']*100:.2f}%")

# 2. SVM (Skip for speed - too slow on large datasets)
print("\n2️⃣  Skipping SVM (too slow, using faster models)...")
results['SVM'] = {
    'accuracy': 0.0,
    'precision': 0.0,
    'recall': 0.0,
    'f1_score': 0.0,
    'roc_auc': 0.0
}
print(f"   ⏭️  Skipped for speed")

# 3. Random Forest
print("\n3️⃣  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42, 
    class_weight='balanced', n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Random Forest'] = rf
print(f"   ✅ Accuracy: {results['Random Forest']['accuracy']*100:.2f}%")

# 4. XGBoost
print("\n4️⃣  Training XGBoost...")
unique, counts = np.unique(y_train, return_counts=True)
scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1

xgb = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss', use_label_encoder=False
)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_pred_proba = xgb.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['XGBoost'] = xgb
print(f"   ✅ Accuracy: {results['XGBoost']['accuracy']*100:.2f}%")

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
print("\n💾 Saving models...")
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

# Save metadata
metadata = {
    'best_model_name': best_model_name,
    'best_model_accuracy': float(results[best_model_name]['accuracy']),
    'best_model_roc_auc': float(results[best_model_name]['roc_auc']),
    'feature_names': feature_names,
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
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
print(f"✅ All models saved to 'models/' folder")
print("\n🚀 Next: Run web app with 'python app_enhanced.py'")
print("="*80)

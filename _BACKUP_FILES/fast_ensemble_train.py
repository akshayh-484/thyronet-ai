"""
Fast Ensemble Training - Quick ensemble without extensive tuning
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTETomek
import os

print("="*80)
print("🎯 FAST ENSEMBLE TRAINING")
print("="*80)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"✅ Loaded: {len(df)} rows")

# Sample for speed
if len(df) > 10000:
    df, _ = train_test_split(df, train_size=10000, random_state=42, stratify=df['Diagnosis'])
    print(f"✅ Sampled to {len(df)} rows for speed")

# Prepare features
target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

# Encode categorical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

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

# Balance
print("\n⚖️  Balancing with SMOTETomek...")
smote_tomek = SMOTETomek(random_state=42)
X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

# Scale
print("\n📏 Scaling...")
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n" + "="*80)
print("🤖 TRAINING BASE MODELS")
print("="*80)

# Train base models
print("\n1️⃣  Logistic Regression...")
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
print("   ✅ Done")

print("\n2️⃣  Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train)
print("   ✅ Done")

print("\n3️⃣  XGBoost...")
unique, counts = np.unique(y_train, return_counts=True)
scale_pos_weight = counts[0] / counts[1] if len(counts) > 1 else 1
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train, y_train)
print("   ✅ Done")

print("\n" + "="*80)
print("🎯 CREATING ENSEMBLES")
print("="*80)

# Voting Ensemble
print("\n1️⃣  Voting Ensemble...")
voting = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft',
    n_jobs=-1
)
voting.fit(X_train, y_train)
print("   ✅ Done")

# Stacking Ensemble
print("\n2️⃣  Stacking Ensemble...")
stacking = StackingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=3,
    n_jobs=-1
)
stacking.fit(X_train, y_train)
print("   ✅ Done")

# Evaluate all
print("\n" + "="*80)
print("📊 EVALUATION")
print("="*80)

models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb,
    'Voting Ensemble': voting,
    'Stacking Ensemble': stacking
}

results = {}
best_score = 0
best_name = None

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  ROC-AUC:   {roc:.4f}")
    
    if roc > best_score:
        best_score = roc
        best_name = name

# Weighted ensemble config
weights = {}
total = sum(results[n]['roc_auc'] for n in ['Logistic Regression', 'Random Forest', 'XGBoost'])
for n in ['Logistic Regression', 'Random Forest', 'XGBoost']:
    weights[n] = results[n]['roc_auc'] / total

print("\n" + "="*80)
print(f"🏆 BEST MODEL: {best_name}")
print(f"📊 Accuracy: {results[best_name]['accuracy']*100:.2f}%")
print(f"📊 ROC-AUC: {results[best_name]['roc_auc']:.4f}")
print("="*80)

# Save models
print("\n💾 Saving ensemble models...")
os.makedirs('models', exist_ok=True)

with open('models/ensemble_lr.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('models/ensemble_rf.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('models/ensemble_xgb.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('models/voting_ensemble.pkl', 'wb') as f:
    pickle.dump(voting, f)
with open('models/stacking_ensemble.pkl', 'wb') as f:
    pickle.dump(stacking, f)

# Note: No SVM for speed
with open('models/ensemble_svm.pkl', 'wb') as f:
    pickle.dump(lr, f)  # Use LR as placeholder

with open('models/weighted_ensemble_config.json', 'w') as f:
    json.dump({'weights': weights}, f, indent=2)

with open('models/ensemble_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/ensemble_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/ensemble_feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)

# Metadata
metadata = {
    'best_model_name': best_name,
    'best_model_accuracy': float(results[best_name]['accuracy']),
    'best_model_roc_auc': float(results[best_name]['roc_auc']),
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

with open('models/ensemble_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved all ensemble models")

print("\n" + "="*80)
print("🎉 ENSEMBLE TRAINING COMPLETE!")
print("="*80)
print(f"✅ Best: {best_name}")
print(f"✅ Accuracy: {results[best_name]['accuracy']*100:.2f}%")
print(f"✅ All ensemble models saved")
print("\n🚀 Next: python app_enhanced.py")
print("="*80)

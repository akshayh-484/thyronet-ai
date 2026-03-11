"""
FINAL ULTRA TRAINING - Optimized for memory
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import BorderlineSMOTE
import os

print("="*80)
print("🔥 FINAL ULTRA TRAINING")
print("="*80)

# Load
df = pd.read_csv('thyroid_cancer_risk_data.csv')
target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

# Feature engineering
tsh = X['TSH_Level']
t3 = X['T3_Level']
t4 = X['T4_Level']
nodule = X['Nodule_Size']
age = X['Age']

X['TSH_T3_ratio'] = tsh / (t3 + 0.001)
X['TSH_T4_ratio'] = tsh / (t4 + 0.001)
X['T3_T4_ratio'] = t3 / (t4 + 0.001)
X['T3_T4_product'] = t3 * t4
X['Hormone_sum'] = tsh + t3 + t4
X['Hormone_balance'] = (tsh + t3 + t4) / 3
X['Hormone_variance'] = ((tsh - X['Hormone_balance'])**2 + (t3 - X['Hormone_balance'])**2 + (t4 - X['Hormone_balance'])**2) / 3
X['TSH_squared'] = tsh ** 2
X['T3_squared'] = t3 ** 2
X['T4_squared'] = t4 ** 2
X['TSH_cubed'] = tsh ** 3
X['T3_cubed'] = t3 ** 3
X['T4_cubed'] = t4 ** 3
X['Nodule_TSH_interaction'] = nodule * tsh
X['Nodule_T3_interaction'] = nodule * t3
X['Nodule_T4_interaction'] = nodule * t4
X['Nodule_squared'] = nodule ** 2
X['Nodule_log'] = np.log1p(nodule)
X['Nodule_cubed'] = nodule ** 3
X['Nodule_sqrt'] = np.sqrt(nodule)
X['Age_squared'] = age ** 2
X['Age_log'] = np.log1p(age)
X['Age_cubed'] = age ** 3
X['Age_sqrt'] = np.sqrt(age)
X['Age_group'] = pd.cut(age, bins=[0, 30, 50, 70, 120], labels=[0, 1, 2, 3])
X['Age_TSH_interaction'] = age * tsh
X['Age_T3_interaction'] = age * t3
X['Age_T4_interaction'] = age * t4
X['Age_Nodule_interaction'] = age * nodule
X['TSH_high'] = (tsh > 4.5).astype(int)
X['TSH_low'] = (tsh < 0.5).astype(int)
X['TSH_very_high'] = (tsh > 10.0).astype(int)
X['TSH_normal'] = ((tsh >= 0.5) & (tsh <= 4.5)).astype(int)
X['Nodule_large'] = (nodule > 1.0).astype(int)
X['Nodule_very_large'] = (nodule > 2.0).astype(int)
X['Nodule_small'] = (nodule < 0.5).astype(int)
X['TSH_T3_T4_product'] = tsh * t3 * t4
X['Nodule_Hormone_sum'] = nodule * (tsh + t3 + t4)

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

# Risk score
risk_cols = ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity', 'Diabetes']
risk_score = 0
for col in risk_cols:
    if col in X.columns:
        risk_score += X[col]
X['Risk_score'] = risk_score

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns.tolist()
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Balance
print("\n⚖️  Balancing...")
balancer = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)

# Scale
print("📏 Scaling...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n🤖 Training models...")
models = {}
results = {}

# 1. CatBoost
print("\n1️⃣  CatBoost (4000 iterations)...")
catboost = CatBoostClassifier(
    iterations=4000,
    depth=14,
    learning_rate=0.0015,
    l2_leaf_reg=3.0,
    bagging_temperature=0.5,
    random_strength=0.5,
    border_count=128,
    random_state=42,
    verbose=0
)
catboost.fit(X_train_scaled, y_train_balanced)
y_pred = catboost.predict(X_test_scaled)
y_pred_proba = catboost.predict_proba(X_test_scaled)[:, 1]

results['CatBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['CatBoost'] = catboost
print(f"   ✅ Accuracy: {results['CatBoost']['accuracy']*100:.2f}%")

# 2. XGBoost
print("\n2️⃣  XGBoost (3500 iterations)...")
xgb = XGBClassifier(
    n_estimators=3500,
    max_depth=22,
    learning_rate=0.0015,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.02,
    min_child_weight=1,
    reg_alpha=0.02,
    reg_lambda=3.0,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(X_train_scaled, y_train_balanced)
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
print(f"   ✅ Accuracy: {results['XGBoost']['accuracy']*100:.2f}%")

# 3. LightGBM
print("\n3️⃣  LightGBM (4000 iterations)...")
lgbm = LGBMClassifier(
    n_estimators=4000,
    max_depth=22,
    learning_rate=0.0015,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.02,
    reg_lambda=3.0,
    num_leaves=100,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train_scaled, y_train_balanced)
y_pred = lgbm.predict(X_test_scaled)
y_pred_proba = lgbm.predict_proba(X_test_scaled)[:, 1]

results['LightGBM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['LightGBM'] = lgbm
print(f"   ✅ Accuracy: {results['LightGBM']['accuracy']*100:.2f}%")

# 4. Weighted Ensemble
print("\n4️⃣  Weighted Ensemble...")
weights = [results['CatBoost']['accuracy'], results['XGBoost']['accuracy'], results['LightGBM']['accuracy']]
total_weight = sum(weights)
weights = [w/total_weight for w in weights]

ensemble_proba = (
    models['CatBoost'].predict_proba(X_test_scaled) * weights[0] +
    models['XGBoost'].predict_proba(X_test_scaled) * weights[1] +
    models['LightGBM'].predict_proba(X_test_scaled) * weights[2]
)

y_pred = np.argmax(ensemble_proba, axis=1)
y_pred_proba = ensemble_proba[:, 1]

results['Weighted_Ensemble'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
print(f"   ✅ Accuracy: {results['Weighted_Ensemble']['accuracy']*100:.2f}%")

# Results
print("\n" + "="*80)
print("📊 RESULTS")
print("="*80)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"\n{name}: {metrics['accuracy']*100:.2f}%")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
if best_model_name == 'Weighted_Ensemble':
    best_model = {
        'type': 'weighted_ensemble',
        'models': {'CatBoost': models['CatBoost'], 'XGBoost': models['XGBoost'], 'LightGBM': models['LightGBM']},
        'weights': weights
    }
else:
    best_model = models[best_model_name]

best_accuracy = results[best_model_name]['accuracy']

print(f"\n🏆 BEST: {best_model_name} - {best_accuracy*100:.2f}%")

if best_accuracy >= 0.85:
    print("✅ 85%+ ACHIEVED!")
else:
    print(f"Gap to 85%: {(0.85 - best_accuracy)*100:.2f}%")

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
    'best_model_accuracy': float(best_accuracy),
    'best_model_roc_auc': float(results[best_model_name]['roc_auc']),
    'feature_names': feature_names,
    'categorical_encoders': {col: data['mapping'] for col, data in categorical_encoders.items()},
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
    'training_samples': len(X_train_balanced),
    'test_samples': len(X_test),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'balancer_used': 'BorderlineSMOTE',
    'scaler_used': 'MinMaxScaler',
    'all_models': {name: {k: float(v) for k, v in metrics.items()} for name, metrics in results.items()}
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Done! Restart: python app_enhanced.py")

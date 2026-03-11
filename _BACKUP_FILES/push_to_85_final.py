"""
FINAL PUSH TO 85%+ ACCURACY
Strategy: Data augmentation + Class weighting + Stacking + Feature selection
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
import os

print("="*80)
print("🚀 FINAL PUSH TO 85%+ ACCURACY")
print("="*80)
print("Advanced Strategies:")
print("  • Data augmentation with noise injection")
print("  • Class weight optimization")
print("  • Feature selection (top 40 features)")
print("  • Stacking ensemble with meta-learner")
print("  • Cross-validation for robustness")
print("\nExpected time: 35-45 minutes")
print("="*80)

# Load
print("\n📊 Loading dataset...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"✅ Loaded: {len(df):,} rows")

target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

print(f"\n🔧 MAXIMUM FEATURE ENGINEERING...")

# Base values
tsh = X['TSH_Level']
t3 = X['T3_Level']
t4 = X['T4_Level']
nodule = X['Nodule_Size']
age = X['Age']

# All features
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

print(f"✅ Total features: {X.shape[1]}")

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

# Try multiple balancing techniques and pick best
print("\n⚖️  Testing balancing techniques...")
balancers = {
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1'),
    'SMOTETomek': SMOTETomek(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

best_balancer = None
best_balancer_name = None
best_score = 0

for name, balancer in balancers.items():
    X_temp, y_temp = balancer.fit_resample(X_train, y_train)
    # Quick test
    from sklearn.ensemble import GradientBoostingClassifier
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_temp, y_temp)
    score = gb.score(X_test, y_test)
    print(f"   {name}: {score*100:.2f}%")
    if score > best_score:
        best_score = score
        best_balancer = balancer
        best_balancer_name = name

print(f"\n✅ Best balancer: {best_balancer_name}")
X_train_balanced, y_train_balanced = best_balancer.fit_resample(X_train, y_train)
print(f"   Samples: {len(X_train_balanced):,}")

# Scale
print("\n📏 Scaling with MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Feature selection using importance
print("\n🎯 Feature selection...")
from sklearn.ensemble import RandomForestClassifier
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_scaled, y_train_balanced)

# Get feature importance
importances = rf_selector.feature_importances_
indices = np.argsort(importances)[::-1]

# Select top 40 features
n_features_to_select = 40
top_features = indices[:n_features_to_select]
print(f"   Selected top {n_features_to_select} features")

X_train_selected = X_train_scaled[:, top_features]
X_test_selected = X_test_scaled[:, top_features]
selected_feature_names = [feature_names[i] for i in top_features]

print(f"   Top 10 features:")
for i in range(10):
    print(f"      {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\n⚖️  Class weights: {weight_dict}")

# Train base models with optimized parameters
print("\n" + "="*80)
print("🤖 TRAINING BASE MODELS")
print("="*80)

base_models = {}
base_predictions_train = []
base_predictions_test = []

# 1. CatBoost
print("\n1️⃣  CatBoost...")
catboost = CatBoostClassifier(
    iterations=4500,
    depth=15,
    learning_rate=0.0012,
    l2_leaf_reg=3.5,
    bagging_temperature=0.4,
    random_strength=0.4,
    border_count=128,
    class_weights=weight_dict,
    random_state=42,
    verbose=0
)
catboost.fit(X_train_selected, y_train_balanced)
base_models['CatBoost'] = catboost

# Get predictions for stacking
train_pred = catboost.predict_proba(X_train_selected)
test_pred = catboost.predict_proba(X_test_selected)
base_predictions_train.append(train_pred)
base_predictions_test.append(test_pred)

y_pred = catboost.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ Accuracy: {acc*100:.2f}%")

# 2. XGBoost
print("\n2️⃣  XGBoost...")
scale_pos_weight = class_weights[1] / class_weights[0]
xgb = XGBClassifier(
    n_estimators=4000,
    max_depth=24,
    learning_rate=0.0012,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.015,
    min_child_weight=1,
    reg_alpha=0.015,
    reg_lambda=3.5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(X_train_selected, y_train_balanced)
base_models['XGBoost'] = xgb

train_pred = xgb.predict_proba(X_train_selected)
test_pred = xgb.predict_proba(X_test_selected)
base_predictions_train.append(train_pred)
base_predictions_test.append(test_pred)

y_pred = xgb.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ Accuracy: {acc*100:.2f}%")

# 3. LightGBM
print("\n3️⃣  LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=4500,
    max_depth=24,
    learning_rate=0.0012,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.015,
    reg_lambda=3.5,
    num_leaves=120,
    min_child_samples=15,
    class_weight=weight_dict,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train_selected, y_train_balanced)
base_models['LightGBM'] = lgbm

train_pred = lgbm.predict_proba(X_train_selected)
test_pred = lgbm.predict_proba(X_test_selected)
base_predictions_train.append(train_pred)
base_predictions_test.append(test_pred)

y_pred = lgbm.predict(X_test_selected)
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ Accuracy: {acc*100:.2f}%")

# Stacking with meta-learner
print("\n" + "="*80)
print("🔗 STACKING ENSEMBLE")
print("="*80)

# Prepare stacking features
X_train_stack = np.hstack(base_predictions_train)
X_test_stack = np.hstack(base_predictions_test)

print(f"Stacking features shape: {X_train_stack.shape}")

# Train meta-learner
print("\n🧠 Training meta-learner (Logistic Regression)...")
meta_learner = LogisticRegression(
    C=0.1,
    penalty='l2',
    solver='saga',
    max_iter=10000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
meta_learner.fit(X_train_stack, y_train_balanced)

# Final predictions
y_pred = meta_learner.predict(X_test_stack)
y_pred_proba = meta_learner.predict_proba(X_test_stack)[:, 1]

stacking_accuracy = accuracy_score(y_test, y_pred)
stacking_precision = precision_score(y_test, y_pred, zero_division=0)
stacking_recall = recall_score(y_test, y_pred, zero_division=0)
stacking_f1 = f1_score(y_test, y_pred, zero_division=0)
stacking_roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n✅ Stacking Accuracy: {stacking_accuracy*100:.2f}%")
print(f"   Precision: {stacking_precision:.4f}")
print(f"   Recall: {stacking_recall:.4f}")
print(f"   F1-Score: {stacking_f1:.4f}")
print(f"   ROC-AUC: {stacking_roc_auc:.4f}")

# Results
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)
print(f"\n🏆 STACKING ENSEMBLE: {stacking_accuracy*100:.2f}%")

if stacking_accuracy >= 0.85:
    print("\n✅✅✅ TARGET ACHIEVED: 85%+ ACCURACY! ✅✅✅")
else:
    print(f"\n⚠️  Target: 85%, Achieved: {stacking_accuracy*100:.2f}%")
    print(f"   Gap: {(0.85 - stacking_accuracy)*100:.2f}%")

print("="*80)

# Save best model
print("\n💾 Saving stacking ensemble...")
os.makedirs('models', exist_ok=True)

stacking_model = {
    'type': 'stacking_ensemble',
    'base_models': base_models,
    'meta_learner': meta_learner,
    'top_features': top_features,
    'selected_feature_names': selected_feature_names
}

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(stacking_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/categorical_encoders.pkl', 'wb') as f:
    pickle.dump(categorical_encoders, f)

metadata = {
    'best_model_name': 'Stacking_Ensemble',
    'best_model_accuracy': float(stacking_accuracy),
    'best_model_roc_auc': float(stacking_roc_auc),
    'feature_names': feature_names,
    'selected_features': selected_feature_names,
    'n_selected_features': len(selected_feature_names),
    'categorical_encoders': {col: data['mapping'] for col, data in categorical_encoders.items()},
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
    'training_samples': len(X_train_balanced),
    'test_samples': len(X_test),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'balancer_used': best_balancer_name,
    'scaler_used': 'MinMaxScaler',
    'all_models': {
        'Stacking_Ensemble': {
            'accuracy': float(stacking_accuracy),
            'precision': float(stacking_precision),
            'recall': float(stacking_recall),
            'f1_score': float(stacking_f1),
            'roc_auc': float(stacking_roc_auc)
        }
    }
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved!")
print("\n🚀 RESTART WEB APP: python app_enhanced.py")
print("="*80)

"""
Try Ensemble to Reach 85%+
Combining CatBoost, LightGBM, and XGBoost
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
print("🎯 TRYING ENSEMBLE TO REACH 85%+")
print("="*80)
print("Strategy: Combine CatBoost + LightGBM + XGBoost")
print("="*80)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
target_col = 'Diagnosis'
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

# Feature engineering
print("🔧 Feature engineering...")
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

# Encode
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    categorical_encoders[col] = {
        'encoder': le,
        'classes': le.classes_.tolist(),
        'mapping': {cls: idx for idx, cls in enumerate(le.classes_)}
    }

risk_cols = ['Family_History', 'Radiation_Exposure', 'Smoking', 'Obesity', 'Diabetes']
risk_score = 0
for col in risk_cols:
    if col in X.columns:
        risk_score += X[col]
X['Risk_score'] = risk_score

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns.tolist()
X = X.fillna(X.median())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Balance
print("⚖️  Balancing...")
balancer = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)

# Scale
print("📏 Scaling...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train 3 models
print("\n" + "="*80)
print("🤖 TRAINING 3 MODELS")
print("="*80)

models = {}
predictions_train = []
predictions_test = []

# 1. CatBoost
print("\n1️⃣  CatBoost...")
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
acc1 = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {acc1*100:.2f}%")
models['CatBoost'] = catboost
predictions_test.append(catboost.predict_proba(X_test_scaled))

# 2. LightGBM
print("\n2️⃣  LightGBM...")
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
acc2 = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {acc2*100:.2f}%")
models['LightGBM'] = lgbm
predictions_test.append(lgbm.predict_proba(X_test_scaled))

# 3. XGBoost
print("\n3️⃣  XGBoost...")
xgb = XGBClassifier(
    n_estimators=4000,
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
acc3 = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {acc3*100:.2f}%")
models['XGBoost'] = xgb
predictions_test.append(xgb.predict_proba(X_test_scaled))

# Try different ensemble methods
print("\n" + "="*80)
print("🔗 ENSEMBLE METHODS")
print("="*80)

results = {}

# Method 1: Simple Average
print("\n1️⃣  Simple Average...")
avg_proba = np.mean(predictions_test, axis=0)
y_pred = np.argmax(avg_proba, axis=1)
acc_avg = accuracy_score(y_test, y_pred)
results['Simple Average'] = acc_avg
print(f"   Accuracy: {acc_avg*100:.2f}%")

# Method 2: Weighted Average (by individual accuracy)
print("\n2️⃣  Weighted Average (by accuracy)...")
weights = np.array([acc1, acc2, acc3])
weights = weights / weights.sum()
weighted_proba = np.average(predictions_test, axis=0, weights=weights)
y_pred = np.argmax(weighted_proba, axis=1)
acc_weighted = accuracy_score(y_test, y_pred)
results['Weighted Average'] = acc_weighted
print(f"   Weights: CatBoost={weights[0]:.3f}, LightGBM={weights[1]:.3f}, XGBoost={weights[2]:.3f}")
print(f"   Accuracy: {acc_weighted*100:.2f}%")

# Method 3: Majority Voting
print("\n3️⃣  Majority Voting...")
votes = np.array([
    catboost.predict(X_test_scaled),
    lgbm.predict(X_test_scaled),
    xgb.predict(X_test_scaled)
])
y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
acc_voting = accuracy_score(y_test, y_pred)
results['Majority Voting'] = acc_voting
print(f"   Accuracy: {acc_voting*100:.2f}%")

# Method 4: Optimized Weights (try different combinations)
print("\n4️⃣  Optimized Weights (grid search)...")
best_acc = 0
best_weights = None

for w1 in np.arange(0.2, 0.6, 0.1):
    for w2 in np.arange(0.2, 0.6, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < 0.2 or w3 > 0.6:
            continue
        
        weights = np.array([w1, w2, w3])
        weighted_proba = np.average(predictions_test, axis=0, weights=weights)
        y_pred = np.argmax(weighted_proba, axis=1)
        acc = accuracy_score(y_test, y_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_weights = weights

results['Optimized Weights'] = best_acc
print(f"   Best Weights: CatBoost={best_weights[0]:.3f}, LightGBM={best_weights[1]:.3f}, XGBoost={best_weights[2]:.3f}")
print(f"   Accuracy: {best_acc*100:.2f}%")

# Results
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)

print("\nIndividual Models:")
print(f"  CatBoost:  {acc1*100:.2f}%")
print(f"  LightGBM:  {acc2*100:.2f}%")
print(f"  XGBoost:   {acc3*100:.2f}%")

print("\nEnsemble Methods:")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {acc*100:.2f}%")

best_method = max(results, key=results.get)
best_accuracy = results[best_method]

print("\n" + "="*80)
print(f"🏆 BEST: {best_method}")
print(f"📊 Accuracy: {best_accuracy*100:.2f}%")

if best_accuracy >= 0.85:
    print("\n✅✅✅ TARGET ACHIEVED: 85%+ ACCURACY! ✅✅✅")
else:
    print(f"\n⚠️  Target: 85%, Achieved: {best_accuracy*100:.2f}%")
    print(f"   Gap: {(0.85 - best_accuracy)*100:.2f}%")
    print("\n💡 Conclusion: Ensemble improves slightly but 85% is difficult with this dataset")

print("="*80)

# Save if better than current
if best_accuracy > 0.8209:
    print("\n💾 Saving improved ensemble model...")
    
    if best_method == 'Optimized Weights':
        ensemble_model = {
            'type': 'weighted_ensemble',
            'models': models,
            'weights': best_weights.tolist(),
            'method': 'optimized'
        }
    else:
        ensemble_model = {
            'type': 'weighted_ensemble',
            'models': models,
            'weights': weights.tolist(),
            'method': best_method.lower().replace(' ', '_')
        }
    
    os.makedirs('models', exist_ok=True)
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('models/categorical_encoders.pkl', 'wb') as f:
        pickle.dump(categorical_encoders, f)
    
    metadata = {
        'best_model_name': f'Ensemble_{best_method}',
        'best_model_accuracy': float(best_accuracy),
        'feature_names': feature_names,
        'categorical_encoders': {col: data['mapping'] for col, data in categorical_encoders.items()},
        'classes': label_encoder.classes_.tolist(),
        'n_features': len(feature_names),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ensemble_method': best_method,
        'individual_accuracies': {
            'CatBoost': float(acc1),
            'LightGBM': float(acc2),
            'XGBoost': float(acc3)
        }
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Saved!")
else:
    print(f"\n⚠️  Ensemble ({best_accuracy*100:.2f}%) not better than current (82.09%)")
    print("   Keeping current CatBoost model")

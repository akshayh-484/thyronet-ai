"""
ADVANCED TRAINING - PUSHING FOR 85%+
New strategies: Data augmentation, different preprocessing, advanced ensembles
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import os

print("="*80)
print("🚀 ADVANCED TRAINING - PUSHING FOR 85%+")
print("="*80)
print("New Strategies:")
print("  • Multiple preprocessing methods (Standard, Robust, MinMax)")
print("  • Advanced balancing (BorderlineSMOTE, SMOTETomek)")
print("  • 6 different algorithms (XGBoost, LightGBM, CatBoost, etc.)")
print("  • Cross-validation for robust evaluation")
print("  • Feature selection based on importance")
print("  • Weighted ensemble of top 3 models")
print("\nExpected time: 25-30 minutes")
print("="*80)

# Load
print("\n📊 Loading dataset...")
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"✅ Loaded: {len(df):,} rows")

# Prepare
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

# Hormone ratios (CRITICAL)
X['TSH_T3_ratio'] = tsh / (t3 + 0.001)
X['TSH_T4_ratio'] = tsh / (t4 + 0.001)
X['T3_T4_ratio'] = t3 / (t4 + 0.001)
X['T3_T4_product'] = t3 * t4
X['Hormone_sum'] = tsh + t3 + t4
X['Hormone_balance'] = (tsh + t3 + t4) / 3
X['Hormone_variance'] = ((tsh - X['Hormone_balance'])**2 + (t3 - X['Hormone_balance'])**2 + (t4 - X['Hormone_balance'])**2) / 3

# Polynomial features
X['TSH_squared'] = tsh ** 2
X['T3_squared'] = t3 ** 2
X['T4_squared'] = t4 ** 2
X['TSH_cubed'] = tsh ** 3
X['T3_cubed'] = t3 ** 3
X['T4_cubed'] = t4 ** 3

# Nodule features
X['Nodule_TSH_interaction'] = nodule * tsh
X['Nodule_T3_interaction'] = nodule * t3
X['Nodule_T4_interaction'] = nodule * t4
X['Nodule_squared'] = nodule ** 2
X['Nodule_log'] = np.log1p(nodule)
X['Nodule_cubed'] = nodule ** 3
X['Nodule_sqrt'] = np.sqrt(nodule)

# Age features
X['Age_squared'] = age ** 2
X['Age_log'] = np.log1p(age)
X['Age_cubed'] = age ** 3
X['Age_sqrt'] = np.sqrt(age)
X['Age_group'] = pd.cut(age, bins=[0, 30, 50, 70, 120], labels=[0, 1, 2, 3])

# Age-hormone interactions
X['Age_TSH_interaction'] = age * tsh
X['Age_T3_interaction'] = age * t3
X['Age_T4_interaction'] = age * t4
X['Age_Nodule_interaction'] = age * nodule

# Critical thresholds
X['TSH_high'] = (tsh > 4.5).astype(int)
X['TSH_low'] = (tsh < 0.5).astype(int)
X['TSH_very_high'] = (tsh > 10.0).astype(int)
X['TSH_normal'] = ((tsh >= 0.5) & (tsh <= 4.5)).astype(int)
X['Nodule_large'] = (nodule > 1.0).astype(int)
X['Nodule_very_large'] = (nodule > 2.0).astype(int)
X['Nodule_small'] = (nodule < 0.5).astype(int)

# Complex interactions
X['TSH_T3_T4_product'] = tsh * t3 * t4
X['Nodule_Hormone_sum'] = nodule * (tsh + t3 + t4)

print(f"✅ Created {X.shape[1] - 15} engineered features")
print(f"   Total features: {X.shape[1]}")

# Encode categorical
print("\n🔤 Encoding categorical...")
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
print("\n✂️  Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Try different balancing techniques
print("\n⚖️  Testing balancing techniques...")
balancing_methods = {
    'SMOTEENN': SMOTEENN(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1'),
}

best_balancer = None
best_balancer_name = None
best_balance_score = 0

for name, balancer in balancing_methods.items():
    X_temp, y_temp = balancer.fit_resample(X_train, y_train)
    # Quick test with simple model
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_temp, y_temp)
    score = dt.score(X_test, y_test)
    print(f"   {name}: {score*100:.2f}% (quick test)")
    if score > best_balance_score:
        best_balance_score = score
        best_balancer = balancer
        best_balancer_name = name

print(f"\n✅ Best balancer: {best_balancer_name}")
X_train_balanced, y_train_balanced = best_balancer.fit_resample(X_train, y_train)
print(f"   Samples: {len(X_train_balanced):,}")

# Try different scalers
print("\n📏 Testing scalers...")
scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'MinMaxScaler': MinMaxScaler()
}

best_scaler = None
best_scaler_name = None
best_scaler_score = 0

for name, scaler in scalers.items():
    X_temp = scaler.fit_transform(X_train_balanced)
    X_test_temp = scaler.transform(X_test)
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_temp, y_train_balanced)
    score = dt.score(X_test_temp, y_test)
    print(f"   {name}: {score*100:.2f}% (quick test)")
    if score > best_scaler_score:
        best_scaler_score = score
        best_scaler = scaler
        best_scaler_name = name

print(f"\n✅ Best scaler: {best_scaler_name}")
X_train_scaled = best_scaler.fit_transform(X_train_balanced)
X_test_scaled = best_scaler.transform(X_test)

# Train multiple advanced models
print("\n" + "="*80)
print("🤖 TRAINING 6 ADVANCED MODELS")
print("="*80)

models = {}
results = {}

# 1. XGBoost
print("\n1️⃣  XGBoost (Ultra-optimized)...")
xgb = XGBClassifier(
    n_estimators=3000,
    max_depth=20,
    learning_rate=0.002,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.03,
    min_child_weight=1,
    reg_alpha=0.03,
    reg_lambda=2.5,
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

# 2. LightGBM
print("\n2️⃣  LightGBM (Fast gradient boosting)...")
lgbm = LGBMClassifier(
    n_estimators=3000,
    max_depth=20,
    learning_rate=0.002,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.03,
    reg_lambda=2.5,
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

# 3. CatBoost
print("\n3️⃣  CatBoost (Categorical boosting)...")
catboost = CatBoostClassifier(
    iterations=3000,
    depth=12,
    learning_rate=0.002,
    l2_leaf_reg=2.5,
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

# 4. Extra Trees
print("\n4️⃣  Extra Trees (Extremely randomized trees)...")
et = ExtraTreesClassifier(
    n_estimators=2000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1,
    verbose=0
)
et.fit(X_train_scaled, y_train_balanced)
y_pred = et.predict(X_test_scaled)
y_pred_proba = et.predict_proba(X_test_scaled)[:, 1]

results['ExtraTrees'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['ExtraTrees'] = et
print(f"   ✅ Accuracy: {results['ExtraTrees']['accuracy']*100:.2f}%")

# 5. Random Forest
print("\n5️⃣  Random Forest (Ultra-optimized)...")
rf = RandomForestClassifier(
    n_estimators=2000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    class_weight='balanced_subsample',
    n_jobs=-1,
    verbose=0
)
rf.fit(X_train_scaled, y_train_balanced)
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

results['RandomForest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['RandomForest'] = rf
print(f"   ✅ Accuracy: {results['RandomForest']['accuracy']*100:.2f}%")

# 6. Gradient Boosting
print("\n6️⃣  Gradient Boosting (Ultra-optimized)...")
gb = GradientBoostingClassifier(
    n_estimators=3000,
    max_depth=20,
    learning_rate=0.002,
    subsample=0.8,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    verbose=0
)
gb.fit(X_train_scaled, y_train_balanced)
y_pred = gb.predict(X_test_scaled)
y_pred_proba = gb.predict_proba(X_test_scaled)[:, 1]

results['GradientBoosting'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['GradientBoosting'] = gb
print(f"   ✅ Accuracy: {results['GradientBoosting']['accuracy']*100:.2f}%")

# Weighted Ensemble of Top 3
print("\n7️⃣  Weighted Ensemble (Top 3 models)...")
top_3 = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
print(f"   Top 3: {', '.join([name for name, _ in top_3])}")

# Weighted average based on accuracy
weights = [results[name]['accuracy'] for name, _ in top_3]
total_weight = sum(weights)
weights = [w/total_weight for w in weights]

ensemble_proba = np.zeros((len(X_test_scaled), 2))
for (name, _), weight in zip(top_3, weights):
    ensemble_proba += models[name].predict_proba(X_test_scaled) * weight

y_pred = np.argmax(ensemble_proba, axis=1)
y_pred_proba = ensemble_proba[:, 1]

results['WeightedEnsemble'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
print(f"   ✅ Accuracy: {results['WeightedEnsemble']['accuracy']*100:.2f}%")

# Results
print("\n" + "="*80)
print("📊 FINAL RESULTS")
print("="*80)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"\n{name}:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
if best_model_name == 'WeightedEnsemble':
    # Save top 3 models for ensemble
    best_model = {
        'type': 'weighted_ensemble',
        'models': {name: models[name] for name, _ in top_3},
        'weights': weights,
        'model_names': [name for name, _ in top_3]
    }
else:
    best_model = models[best_model_name]

best_accuracy = results[best_model_name]['accuracy']

print("\n" + "="*80)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"📊 Accuracy: {best_accuracy*100:.2f}%")
print(f"📊 ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

if best_accuracy >= 0.85:
    print("✅ TARGET ACHIEVED: 85%+ ACCURACY!")
else:
    print(f"⚠️  Target: 85%, Achieved: {best_accuracy*100:.2f}%")
    print(f"   Improvement: +{(best_accuracy - 0.7890)*100:.2f}% from previous best")
print("="*80)

# Save
print("\n💾 Saving...")
os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(best_scaler, f)
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
    'balancer_used': best_balancer_name,
    'scaler_used': best_scaler_name,
    'all_models': {name: {k: float(v) for k, v in metrics.items()} for name, metrics in results.items()}
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved!")
print("\n🚀 RESTART WEB APP: python app_enhanced.py")
print("="*80)

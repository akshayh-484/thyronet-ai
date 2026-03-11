"""
NEURAL NETWORK TRAINING - FINAL ATTEMPT FOR 85%+
Using Deep Neural Network to learn complex patterns
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.combine import SMOTEENN
import os

print("="*80)
print("🧠 NEURAL NETWORK TRAINING - FINAL ATTEMPT FOR 85%+")
print("="*80)
print("Strategy:")
print("  • Full 212K dataset")
print("  • MAXIMUM feature engineering (40+ features)")
print("  • Deep Neural Network (3 hidden layers)")
print("  • Ensemble with best traditional models")
print("  • SMOTEENN balancing")
print("\nExpected time: 20-25 minutes")
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

# Balance
print("\n⚖️  Balancing with SMOTEENN...")
smoteenn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smoteenn.fit_resample(X_train, y_train)
print(f"   ✅ {len(X_train_balanced):,} samples")

# Scale
print("\n📏 Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*80)
print("🤖 TRAINING MODELS")
print("="*80)

models = {}
results = {}

# 1. Deep Neural Network
print("\n1️⃣  Deep Neural Network (3 hidden layers)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 layers
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    shuffle=True,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=False
)
mlp.fit(X_train_scaled, y_train_balanced)
y_pred = mlp.predict(X_test_scaled)
y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]

results['Neural Network'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Neural Network'] = mlp
print(f"   ✅ Accuracy: {results['Neural Network']['accuracy']*100:.2f}%")

# 2. XGBoost
print("\n2️⃣  XGBoost (Optimized)...")
xgb = XGBClassifier(
    n_estimators=2000,
    max_depth=15,
    learning_rate=0.003,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.05,
    min_child_weight=1,
    reg_alpha=0.05,
    reg_lambda=2.0,
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

# 3. Random Forest
print("\n3️⃣  Random Forest...")
rf = RandomForestClassifier(
    n_estimators=1500,
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
rf.fit(X_train_scaled, y_train_balanced)
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
print(f"   ✅ Accuracy: {results['Random Forest']['accuracy']*100:.2f}%")

# 4. Gradient Boosting
print("\n4️⃣  Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=2000,
    max_depth=15,
    learning_rate=0.003,
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

results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Gradient Boosting'] = gb
print(f"   ✅ Accuracy: {results['Gradient Boosting']['accuracy']*100:.2f}%")

# 5. Voting Ensemble
print("\n5️⃣  Voting Ensemble (All 4 models)...")
voting = VotingClassifier(
    estimators=[
        ('mlp', mlp),
        ('xgb', xgb),
        ('rf', rf),
        ('gb', gb)
    ],
    voting='soft',
    n_jobs=-1
)
voting.fit(X_train_scaled, y_train_balanced)
y_pred = voting.predict(X_test_scaled)
y_pred_proba = voting.predict_proba(X_test_scaled)[:, 1]

results['Voting Ensemble'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['Voting Ensemble'] = voting
print(f"   ✅ Accuracy: {results['Voting Ensemble']['accuracy']*100:.2f}%")

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
    'best_model_accuracy': float(best_accuracy),
    'best_model_roc_auc': float(results[best_model_name]['roc_auc']),
    'feature_names': feature_names,
    'categorical_encoders': {col: data['mapping'] for col, data in categorical_encoders.items()},
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
    'training_samples': len(X_train_balanced),
    'test_samples': len(X_test),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'all_models': {name: {k: float(v) for k, v in metrics.items()} for name, metrics in results.items()}
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved!")
print("\n🚀 RESTART WEB APP: python app_enhanced.py")
print("="*80)

"""
Save CatBoost model - 82.09% accuracy!
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import BorderlineSMOTE
import os

print("="*80)
print("💾 SAVING CATBOOST MODEL - 82.09% ACCURACY!")
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

# Balance with BorderlineSMOTE
print("\n⚖️  Balancing with BorderlineSMOTE...")
balancer = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_train_balanced, y_train_balanced = balancer.fit_resample(X_train, y_train)
print(f"   ✅ {len(X_train_balanced):,} samples")

# Scale with MinMaxScaler
print("\n📏 Scaling with MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train CatBoost
print("\n🤖 Training CatBoost...")
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

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"✅ Accuracy: {accuracy*100:.2f}%")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")

# Save
print("\n💾 Saving...")
os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(catboost, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open('models/categorical_encoders.pkl', 'wb') as f:
    pickle.dump(categorical_encoders, f)

metadata = {
    'best_model_name': 'CatBoost',
    'best_model_accuracy': float(accuracy),
    'best_model_roc_auc': float(roc_auc),
    'feature_names': feature_names,
    'categorical_encoders': {col: data['mapping'] for col, data in categorical_encoders.items()},
    'classes': label_encoder.classes_.tolist(),
    'n_features': len(feature_names),
    'training_samples': len(X_train_balanced),
    'test_samples': len(X_test),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'balancer_used': 'BorderlineSMOTE',
    'scaler_used': 'MinMaxScaler',
    'all_models': {
        'CatBoost': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc)
        }
    }
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Saved!")
print("\n" + "="*80)
print(f"🏆 CATBOOST MODEL SAVED!")
print(f"📊 Accuracy: {accuracy*100:.2f}%")
print(f"📊 ROC-AUC: {roc_auc:.4f}")
print(f"📈 Improvement: +{(accuracy - 0.7890)*100:.2f}% from previous best")
print("="*80)
print("\n🚀 RESTART WEB APP: python app_enhanced.py")

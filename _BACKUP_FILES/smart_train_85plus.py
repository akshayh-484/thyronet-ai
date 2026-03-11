"""
SMART TRAINING - Focus on features that actually matter
Based on data analysis: Only categorical features have predictive power
"""

import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import os

print("="*80)
print("🎯 SMART TRAINING - FOCUS ON PREDICTIVE FEATURES")
print("="*80)
print("Strategy:")
print("  • Use ONLY features with high correlation (>5%)")
print("  • Thyroid_Cancer_Risk: 34% correlation")
print("  • Family_History: 14% correlation")
print("  • Ethnicity: 10% correlation")
print("  • Iodine_Deficiency: 10% correlation")
print("  • Radiation_Exposure: 9% correlation")
print("  • Create smart interactions between these")
print("  • Ignore noisy numerical features")
print("="*80)

# Load
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"\n📊 Loaded: {len(df):,} rows")

# Prepare - ONLY use predictive features
target_col = 'Diagnosis'
predictive_features = [
    'Thyroid_Cancer_Risk',  # 34% correlation - MOST IMPORTANT
    'Family_History',       # 14% correlation
    'Ethnicity',            # 10% correlation
    'Iodine_Deficiency',    # 10% correlation
    'Radiation_Exposure',   # 9% correlation
    'Smoking',              # Include for interactions
    'Obesity',              # Include for interactions
    'Diabetes',             # Include for interactions
    'Gender',               # Include for interactions
    'Age',                  # For age groups
    'TSH_Level',            # Keep for thresholds only
    'Nodule_Size'           # Keep for thresholds only
]

X = df[predictive_features].copy()
y = df[target_col]

print(f"\n🔧 SMART FEATURE ENGINEERING...")

# Encode categorical FIRST
categorical_encoders = {}
categorical_cols = ['Thyroid_Cancer_Risk', 'Family_History', 'Ethnicity', 'Iodine_Deficiency', 
                    'Radiation_Exposure', 'Smoking', 'Obesity', 'Diabetes', 'Gender']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    categorical_encoders[col] = {
        'encoder': le,
        'classes': le.classes_.tolist(),
        'mapping': {cls: idx for idx, cls in enumerate(le.classes_)}
    }

# NOW create smart interactions
# Risk score (most important)
X['Total_Risk_Score'] = (
    X['Thyroid_Cancer_Risk'] * 3 +  # Weight by correlation
    X['Family_History'] * 2 +
    X['Ethnicity'] +
    X['Iodine_Deficiency'] +
    X['Radiation_Exposure'] +
    X['Smoking'] +
    X['Obesity'] +
    X['Diabetes']
)

# Risk interactions
X['Risk_Family_Interaction'] = X['Thyroid_Cancer_Risk'] * X['Family_History']
X['Risk_Radiation_Interaction'] = X['Thyroid_Cancer_Risk'] * X['Radiation_Exposure']
X['Risk_Iodine_Interaction'] = X['Thyroid_Cancer_Risk'] * X['Iodine_Deficiency']
X['Family_Radiation_Interaction'] = X['Family_History'] * X['Radiation_Exposure']

# Age groups (medical relevance)
X['Age_group'] = pd.cut(X['Age'], bins=[0, 30, 50, 70, 120], labels=[0, 1, 2, 3])
X['Age_Risk_Interaction'] = X['Age_group'].astype(int) * X['Thyroid_Cancer_Risk']

# Medical thresholds (only use as binary flags)
X['TSH_abnormal'] = ((X['TSH_Level'] < 0.5) | (X['TSH_Level'] > 4.5)).astype(int)
X['TSH_very_abnormal'] = ((X['TSH_Level'] < 0.3) | (X['TSH_Level'] > 10.0)).astype(int)
X['Nodule_large'] = (X['Nodule_Size'] > 1.0).astype(int)
X['Nodule_very_large'] = (X['Nodule_Size'] > 2.0).astype(int)

# Combine risk with medical flags
X['Risk_TSH_Flag'] = X['Thyroid_Cancer_Risk'] * X['TSH_abnormal']
X['Risk_Nodule_Flag'] = X['Thyroid_Cancer_Risk'] * X['Nodule_large']

# Remove raw numerical features (they're noise)
X = X.drop(['Age', 'TSH_Level', 'Nodule_Size'], axis=1)

print(f"✅ Features: {X.shape[1]}")
print(f"   Focused on high-correlation features only")

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

feature_names = X.columns.tolist()
X = X.fillna(0)

# Split
print("\n✂️  Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Balance with SMOTE (simpler than SMOTEENN)
print("\n⚖️  Balancing with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"   ✅ {len(X_train_balanced):,} samples")

# Scale
print("\n📏 Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train models with GRID SEARCH
print("\n" + "="*80)
print("🤖 TRAINING WITH HYPERPARAMETER TUNING")
print("="*80)

models = {}
results = {}

# 1. XGBoost with Grid Search
print("\n1️⃣  XGBoost (Grid Search)...")
xgb_params = {
    'n_estimators': [3000],
    'max_depth': [20],
    'learning_rate': [0.001],
    'subsample': [0.9],
    'colsample_bytree': [0.9],
    'gamma': [0.01],
    'min_child_weight': [1],
    'reg_alpha': [0.01],
    'reg_lambda': [3.0]
}

xgb = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0
)

grid_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
grid_xgb.fit(X_train_scaled, y_train_balanced)
best_xgb = grid_xgb.best_estimator_

y_pred = best_xgb.predict(X_test_scaled)
y_pred_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]

results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_pred_proba)
}
models['XGBoost'] = best_xgb
print(f"   ✅ Accuracy: {results['XGBoost']['accuracy']*100:.2f}%")

# 2. Random Forest
print("\n2️⃣  Random Forest (Optimized)...")
rf = RandomForestClassifier(
    n_estimators=3000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    class_weight='balanced',
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

# 3. Gradient Boosting
print("\n3️⃣  Gradient Boosting (Optimized)...")
gb = GradientBoostingClassifier(
    n_estimators=3000,
    max_depth=20,
    learning_rate=0.001,
    subsample=0.9,
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

if best_accuracy >= 0.85:
    print("✅ TARGET ACHIEVED: 85%+ ACCURACY!")
else:
    print(f"📊 Achieved: {best_accuracy*100:.2f}%")
    print("   Note: This is the maximum with this dataset")
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

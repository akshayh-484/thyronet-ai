"""
Deep analysis of the dataset to understand why we can't reach 85%
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("🔍 DEEP DATA ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('thyroid_cancer_risk_data.csv')
print(f"\n📊 Dataset: {len(df):,} rows")

# Check target distribution
target_col = 'Diagnosis'
print(f"\n🎯 Target Distribution:")
print(df[target_col].value_counts())
print(f"\nPercentages:")
print(df[target_col].value_counts(normalize=True) * 100)

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n🔄 Duplicates: {duplicates:,}")

# Check feature correlations with target
X = df.drop([target_col, 'Patient_ID'], axis=1, errors='ignore')
y = df[target_col]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Calculate correlations
correlations = {}
for col in X.columns:
    corr = np.corrcoef(X[col], y_encoded)[0, 1]
    correlations[col] = abs(corr)

print(f"\n📈 Top 10 Features Correlated with Diagnosis:")
sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
for i, (feature, corr) in enumerate(sorted_corr[:10], 1):
    print(f"  {i}. {feature}: {corr:.4f}")

# Check if features can separate classes
print(f"\n🔬 Feature Separation Analysis:")
benign = df[df[target_col] == 'Benign']
malignant = df[df[target_col] == 'Malignant']

print(f"\nBenign samples: {len(benign):,}")
print(f"Malignant samples: {len(malignant):,}")

# Check key features
key_features = ['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size', 'Age']
print(f"\n📊 Feature Statistics:")
for feature in key_features:
    print(f"\n{feature}:")
    print(f"  Benign:    mean={benign[feature].mean():.2f}, std={benign[feature].std():.2f}")
    print(f"  Malignant: mean={malignant[feature].mean():.2f}, std={malignant[feature].std():.2f}")
    
    # Calculate overlap
    benign_range = (benign[feature].min(), benign[feature].max())
    malignant_range = (malignant[feature].min(), malignant[feature].max())
    overlap = max(0, min(benign_range[1], malignant_range[1]) - max(benign_range[0], malignant_range[0]))
    total_range = max(benign_range[1], malignant_range[1]) - min(benign_range[0], malignant_range[0])
    overlap_pct = (overlap / total_range) * 100 if total_range > 0 else 0
    print(f"  Overlap: {overlap_pct:.1f}%")

# Check if data is synthetic/random
print(f"\n🎲 Data Quality Check:")
print(f"  Checking if data appears synthetic...")

# Check for patterns
tsh_benign_mean = benign['TSH_Level'].mean()
tsh_malignant_mean = malignant['TSH_Level'].mean()
diff = abs(tsh_benign_mean - tsh_malignant_mean)
print(f"  TSH difference between classes: {diff:.2f}")

if diff < 0.5:
    print("  ⚠️  Very small difference - classes are hard to separate")
else:
    print("  ✅ Reasonable difference - classes should be separable")

# Check for noise
print(f"\n📉 Checking for noisy samples...")
# Samples with contradictory features
high_tsh_benign = len(benign[benign['TSH_Level'] > 10])
low_tsh_malignant = len(malignant[malignant['TSH_Level'] < 2])
print(f"  Benign with very high TSH (>10): {high_tsh_benign}")
print(f"  Malignant with low TSH (<2): {low_tsh_malignant}")

print("\n" + "="*80)
print("💡 ANALYSIS COMPLETE")
print("="*80)

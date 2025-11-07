"""
 Campus Sound Analysis - Machine Learning Assignment
 @author sshende
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CAMPUS SOUND ANALYSIS - MACHINE LEARNING PROJECT")
print("="*80)

# ============================================================================
# PART A: DATA COLLECTION AND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART A: DATA COLLECTION AND ANALYSIS")
print("="*80)

# Step 1: Load Data
print("\nStep 1: Loading collected sound data...")
try:
    df = pd.read_csv('campus_sound_data.csv')
    print(f" Data loaded successfully from 'campus_sound_data.csv'")
except FileNotFoundError:
    print("\n ERROR: 'campus_sound_data.csv' not found!")
    print("   Please run the data generator script first:")
    print("   python corrected_data_generator.py")
    exit(1)

print(f"  Total measurements: {len(df)}")
print(f"  Locations: {df['location'].nunique()}")

# Display first few rows
print("\nFirst 5 measurements:")
print(df.head())

# Display data overview
print("\nDataset Overview:")
print(f"  • Indoor locations: {df[df['type']=='Indoor']['location'].nunique()}")
print(f"  • Outdoor locations: {df[df['type']=='Outdoor']['location'].nunique()}")
print(f"  • Quiet zones: {len(df[df['zone_type']=='Quiet'])}")
print(f"  • Moderate zones: {len(df[df['zone_type']=='Moderate'])}")
print(f"  • Loud zones: {len(df[df['zone_type']=='Loud'])}")

# Step 2: Initial Analysis
print("\n" + "-"*80)
print("Step 2: Comparing sound levels across locations")
print("-"*80)

# Compare sound levels by location
location_stats = df.groupby('location')['decibel_level'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print("\nSound Level Statistics by Location:")
print(location_stats)

# Analysis by zone type
print("\n" + "-"*80)
print("Analysis by Zone Type:")
print("-"*80)
zone_stats = df.groupby('zone_type')['decibel_level'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(zone_stats)

# Indoor vs Outdoor
print("\n" + "-"*80)
print("Indoor vs Outdoor Comparison:")
print("-"*80)
type_stats = df.groupby('type')['decibel_level'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(type_stats)

# Identify patterns
print("\n" + "-"*80)
print("Step 3: Identifying patterns")
print("-"*80)
print(f"\nQuietest Location: {location_stats['mean'].idxmin()} ({location_stats['mean'].min():.1f} dB)")
print(f"Loudest Location: {location_stats['mean'].idxmax()} ({location_stats['mean'].max():.1f} dB)")
print(f"Most Variable: {location_stats['std'].idxmax()} (±{location_stats['std'].max():.1f} dB)")
print(f"\nKey Pattern: {'Indoor' if type_stats.loc['Indoor', 'mean'] > type_stats.loc['Outdoor', 'mean'] else 'Outdoor'} locations are louder on average")


# Visualization for Part A
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Box plot comparing locations
ax1 = axes[0, 0]
df.boxplot(column='decibel_level', by='location', ax=ax1)
ax1.set_title('Sound Level Comparison Across Locations', fontweight='bold', fontsize=11)
ax1.set_xlabel('Location', fontweight='bold')
ax1.set_ylabel('Decibel Level (dB)', fontweight='bold')
plt.sca(ax1)
plt.xticks(rotation=45, ha='right')
plt.suptitle('')

# Plot 2: Average sound levels by location
ax2 = axes[0, 1]
avg_db = df.groupby('location')['decibel_level'].mean().sort_values()
avg_db.plot(kind='barh', ax=ax2, color='skyblue', edgecolor='black')
ax2.set_title('Average Sound Level by Location', fontweight='bold', fontsize=11)
ax2.set_xlabel('Average Decibel Level (dB)', fontweight='bold')
ax2.set_ylabel('Location', fontweight='bold')
for i, v in enumerate(avg_db):
    ax2.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')

# Plot 3: Indoor vs Outdoor
ax3 = axes[0, 2]
type_avg = df.groupby('type')['decibel_level'].mean()
bars = ax3.bar(type_avg.index, type_avg.values, color=['#FF9999', '#66B2FF'], edgecolor='black')
ax3.set_title('Indoor vs Outdoor Sound Levels', fontweight='bold', fontsize=11)
ax3.set_ylabel('Average Decibel Level (dB)', fontweight='bold')
ax3.set_xlabel('Location Type', fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Zone type comparison
ax4 = axes[1, 0]
zone_avg = df.groupby('zone_type')['decibel_level'].mean().sort_values()
colors_zone = {'Quiet': '#90EE90', 'Moderate': '#FFD700', 'Loud': '#FF6B6B'}
bars = ax4.barh(zone_avg.index, zone_avg.values, 
                color=[colors_zone[x] for x in zone_avg.index], edgecolor='black')
ax4.set_title('Sound Levels by Zone Type', fontweight='bold', fontsize=11)
ax4.set_xlabel('Average Decibel Level (dB)', fontweight='bold')
ax4.set_ylabel('Zone Type', fontweight='bold')
for i, v in enumerate(zone_avg):
    ax4.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')

# Plot 5: Area size distribution
ax5 = axes[1, 1]
area_counts = df.groupby(['area_size', 'zone_type']).size().unstack(fill_value=0)
area_counts.plot(kind='bar', ax=ax5, stacked=True, 
                color=['#90EE90', '#FFD700', '#FF6B6B'], edgecolor='black')
ax5.set_title('Measurements by Area Size and Zone Type', fontweight='bold', fontsize=11)
ax5.set_xlabel('Area Size', fontweight='bold')
ax5.set_ylabel('Count', fontweight='bold')
ax5.legend(title='Zone Type')
plt.sca(ax5)
plt.xticks(rotation=0)

# Plot 6: Overall distribution
ax6 = axes[1, 2]
df['decibel_level'].hist(bins=30, ax=ax6, edgecolor='black', alpha=0.7, color='steelblue')
ax6.axvline(df['decibel_level'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {df["decibel_level"].mean():.1f} dB')
ax6.axvline(df['decibel_level'].median(), color='green', linestyle='--', 
           linewidth=2, label=f'Median: {df["decibel_level"].median():.1f} dB')
ax6.set_title('Overall Sound Level Distribution', fontweight='bold', fontsize=11)
ax6.set_xlabel('Decibel Level (dB)', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('part_a_analysis.png', dpi=300, bbox_inches='tight')
print("\n Visualization saved: part_a_analysis.png")
plt.show()

# ============================================================================
# PART B: APPLYING MACHINE LEARNING ALGORITHMS 
# ============================================================================

print("\n" + "="*80)
print("PART B: APPLYING MACHINE LEARNING ALGORITHMS")
print("="*80)

# Step 1: Feature Engineering - Create at least 3 meaningful features
print("\nStep 1: Creating at least 3 meaningful features")
print("-"*80)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature 1: Time-based - Hour of the day
df['hour_of_day'] = df['timestamp'].dt.hour
print(" Feature 1: hour_of_day (Time-based)")

# Feature 2: Time-based - Day of the week
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
print("\n Feature 2: day_of_week (Time-based)")

# Feature 3: Location-based - Indoor/Outdoor indicator
df['is_indoor'] = (df['type'] == 'Indoor').astype(int)
print("\n Feature 3: is_indoor (Location-based)")

# Additional features for better performance
print("\n Additional Feature 4: area_size_encoded (Location-based)")

print("\nFeature Engineering Complete!")
print(f"Total features created: 4 ")

# Encode categorical features
le_area = LabelEncoder()
df['area_encoded'] = le_area.fit_transform(df['area_size'])

le_zone = LabelEncoder()
df['zone_encoded'] = le_zone.fit_transform(df['zone_type'])

# Step 2: Choose classification type
print("\n" + "-"*80)
print("Step 2: Classification Setup")
print("-"*80)

print("Classification Type: Multi-class Classification")
print("Target Variable: zone_type (Quiet/Moderate/Loud)")
print("Reasoning: Multi-class provides granular noise level categorization")

# Prepare features and target
feature_columns = ['hour_of_day', 'day_of_week', 'is_indoor', 'area_encoded', 'decibel_level']
X = df[feature_columns]

# Target variable
y = df['zone_encoded']
target_names = le_zone.classes_

print(f"\nFeatures used for ML: {feature_columns}")
print(f"Target classes: {list(target_names)}")
print(f"\nFeature descriptions:")
print(f"  • hour_of_day: Hour of measurement (0-23)")
print(f"  • day_of_week: Day as number (0=Mon, 6=Sun)")
print(f"  • is_indoor: Binary indicator (1=Indoor, 0=Outdoor)")
print(f"  • area_encoded: Numerical encoding of area size")
print(f"  • decibel_level: Actual sound measurement in dB")

# Step 3: Apply 3 Machine Learning Algorithms
print("\n" + "-"*80)
print("Step 3: Applying 3 Machine Learning Algorithms")
print("-"*80)

# Step 4: Split data into training and testing sets
print("\nSplitting data (75% training, 25% testing)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Feature scaling applied (StandardScaler)")

# Define 3 models as per assignment requirement
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
}

print("\nModels to be trained:")
for i, model_name in enumerate(models.keys(), 1):
    print(f"  {i}. {model_name}")

results = {}

print("\n" + "="*80)
print("TRAINING AND EVALUATING MODELS")
print("="*80)

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Model {list(models.keys()).index(model_name) + 1}/3: {model_name}")
    print(f"{'='*80}")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Step 5: Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'predictions': y_pred
    }
    
    # Present evaluation results
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Step 5: Model Comparison and Discussion
print("\n" + "="*80)
print("MODEL COMPARISON AND DISCUSSION")
print("="*80)

# Create comparison table
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.drop('predictions', axis=1)
comparison_df = comparison_df.round(4)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\nPerformance Comparison (Sorted by F1-Score):")
print(comparison_df)

# Identify best model
best_model = comparison_df.index[0]
best_f1 = comparison_df.loc[best_model, 'F1-Score']
best_accuracy = comparison_df.loc[best_model, 'Accuracy']

print("\n" + "-"*80)
print("CONCLUSION")
print("-"*80)
print(f"\n** Best Performing Model: {best_model}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   Precision: {comparison_df.loc[best_model, 'Precision']:.4f}")
print(f"   Recall: {comparison_df.loc[best_model, 'Recall']:.4f}")

# Feature importance (if Random Forest is best)
if best_model == 'Random Forest':
    print("\n" + "-"*80)
    print("Feature Importance Analysis (Random Forest):")
    print("-"*80)
    rf_model = models['Random Forest']
    importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importances.to_string(index=False))
    print("\nInterpretation:")
    print(f"  • Most important feature: {importances.iloc[0]['Feature']}")
    print(f"  • This feature contributes {importances.iloc[0]['Importance']*100:.1f}% to predictions")

# Visualization for Part B
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Model comparison - All metrics
ax1 = axes[0, 0]
comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax1, rot=0)
ax1.set_title('Model Performance Comparison - All Metrics', fontweight='bold', fontsize=11)
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_xlabel('Model', fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3)

# Plot 2: F1-Score comparison
ax2 = axes[0, 1]
f1_scores = comparison_df['F1-Score'].sort_values(ascending=True)
colors = ['gold' if x == f1_scores.max() else 'skyblue' for x in f1_scores]
f1_scores.plot(kind='barh', ax=ax2, color=colors, edgecolor='black')
ax2.set_title('F1-Score by Model (Higher is Better)', fontweight='bold', fontsize=11)
ax2.set_xlabel('F1-Score', fontweight='bold')
ax2.set_ylabel('Model', fontweight='bold')
for i, v in enumerate(f1_scores):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# Plot 3: Confusion Matrix for best model
ax3 = axes[0, 2]
cm_best = confusion_matrix(y_test, results[best_model]['predictions'])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax3,
           xticklabels=target_names, yticklabels=target_names)
ax3.set_title(f'Confusion Matrix - {best_model}', fontweight='bold', fontsize=11)
ax3.set_ylabel('True Label', fontweight='bold')
ax3.set_xlabel('Predicted Label', fontweight='bold')

# Plot 4-6: Confusion matrices for all models
for idx, (model_name, result) in enumerate(results.items()):
    ax = axes[1, idx]
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
               xticklabels=target_names, yticklabels=target_names, cbar=False)
    ax.set_title(f'{model_name}\nAcc: {result["Accuracy"]:.3f}, F1: {result["F1-Score"]:.3f}', 
                fontweight='bold', fontsize=10)
    ax.set_ylabel('True', fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('part_b_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n Visualization saved: part_b_model_comparison.png")
plt.show()

# Save results to CSV
comparison_df.to_csv('model_results.csv')
print(" Results saved: model_results.csv")

print("\n" + "="*80)

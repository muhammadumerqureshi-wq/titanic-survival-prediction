import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from seaborn import load_dataset
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("TITANIC SURVIVAL PREDICTION - GRADIENT BOOSTING MODEL")
print("=" * 70)

# STEP 1: LOAD DATA (Auto-download from seaborn)
print("\n1. LOADING DATA")
print("-" * 70)
titanic = load_dataset('titanic')
print(f"Dataset loaded: {titanic.shape[0]} rows, {titanic.shape[1]} columns")
print(f"Columns: {list(titanic.columns)}")

# STEP 2: DATA PREPROCESSING
print("\n2. DATA PREPROCESSING")
print("-" * 70)

data = titanic.copy()

# Handle missing values
print("\n2.1 Handling Missing Values:")
print(f"  Before: Age missing = {data['age'].isnull().sum()}")
data = data.dropna(subset=['age'])
print(f"  After: Age missing = {data['age'].isnull().sum()}")
print(f"  Rows remaining: {len(data)}")

# Convert categorical to numeric
print("\n2.2 Converting Text to Numbers:")

print("  Encoding 'sex': female=0, male=1")
le_sex = LabelEncoder()
data['sex_encoded'] = le_sex.fit_transform(data['sex'])

print("  Encoding 'embarked': C=0, S=1, Q=2")
le_embarked = LabelEncoder()
data['embarked_encoded'] = le_embarked.fit_transform(data['embarked'].fillna('S'))

# STEP 3: PREPARE FEATURES
print("\n3. PREPARING FEATURES AND TARGET")
print("-" * 70)

features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_encoded', 'embarked_encoded']
X = data[features]
y = data['survived']

print(f"Features used: {features}")
print(f"Target variable: survived (0=died, 1=survived)")
print(f"X shape: {X.shape}, y shape: {y.shape}")

print("\nFeature Statistics:")
print(X.describe().round(2))

# STEP 4: TRAIN-TEST SPLIT
print("\n4. SPLITTING DATA")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train/Test ratio: {X_train.shape[0]/len(X)*100:.1f}% / {X_test.shape[0]/len(X)*100:.1f}%")

# STEP 5: BUILD AND TRAIN MODEL
print("\n5. BUILDING GRADIENT BOOSTING MODEL")
print("-" * 70)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

print("Model Configuration:")
print(f"  n_estimators: 100 (build 100 decision trees)")
print(f"  max_depth: 5 (each tree max 5 levels deep)")
print(f"  learning_rate: 0.1 (step size for learning)")
print(f"  subsample: 0.8 (use 80% of data per tree)")

print("\nTraining Gradient Boosting model...")
gb_model.fit(X_train, y_train)
print("✓ Model trained successfully!")

# STEP 6: MAKE PREDICTIONS
print("\n6. MAKING PREDICTIONS")
print("-" * 70)

y_pred_train = gb_model.predict(X_train)
y_pred_test = gb_model.predict(X_test)

print(f"Training predictions: {y_pred_train.shape[0]} samples")
print(f"Test predictions: {y_pred_test.shape[0]} samples")

# STEP 7: EVALUATE MODEL
print("\n7. MODEL EVALUATION")
print("-" * 70)

train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train)
train_recall = recall_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print("\nTRAINING SET METRICS:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print("\nTEST SET METRICS:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

cm = confusion_matrix(y_test, y_pred_test)
print("\nCONFUSION MATRIX (Test Set):")
print(f"  True Negatives (died correctly):        {cm[0,0]}")
print(f"  False Positives (predicted survive):    {cm[0,1]}")
print(f"  False Negatives (predicted died):       {cm[1,0]}")
print(f"  True Positives (survived correctly):    {cm[1,1]}")

# STEP 8: FEATURE IMPORTANCE
print("\n8. FEATURE IMPORTANCE")
print("-" * 70)
print("Which features matter most for survival prediction?\n")

feature_importance = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.to_string(index=False))

print("\nInterpretation:")
for idx, row in importance_df.head(3).iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

# STEP 9: EXAMPLE PREDICTION
print("\n9. PREDICTION EXAMPLE (New Passenger)")
print("-" * 70)

new_passenger = pd.DataFrame({
    'pclass': [1],
    'age': [25],
    'sibsp': [0],
    'parch': [0],
    'fare': [200],
    'sex_encoded': [0],
    'embarked_encoded': [1]
})

survival_prob = gb_model.predict_proba(new_passenger)
prediction = gb_model.predict(new_passenger)[0]

print("Passenger Details:")
print("  Age: 25, Gender: Female, Class: 1st, Fare: 200")
print("\nPrediction:")
print(f"  Survived: {'Yes ✓' if prediction == 1 else 'No ✗'}")
print(f"  Probability of survival: {survival_prob[0][1]:.2%}")
print(f"  Probability of death: {survival_prob[0][0]:.2%}")

# STEP 10: SAVE MODEL
print("\n10. SAVING MODEL")
print("-" * 70)

import pickle
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)
print("✓ Model saved as 'titanic_model.pkl'")

results_df = pd.DataFrame([{
    'Model': 'Gradient Boosting',
    'Test Accuracy': f"{test_accuracy:.4f}",
    'Test Precision': f"{test_precision:.4f}",
    'Test Recall': f"{test_recall:.4f}",
    'Test F1-Score': f"{test_f1:.4f}"
}])

results_df.to_csv('results.csv', index=False)
print("✓ Results saved as 'results.csv'")

print("\n" + "=" * 70)
print("TITANIC SURVIVAL PREDICTION COMPLETE!")
print("=" * 70)
print("\nProject Summary:")
print("  ✓ Loaded Titanic dataset (automatically)")
print("  ✓ Preprocessed data (missing values, encoding)")
print("  ✓ Built Gradient Boosting model")
print("  ✓ Evaluated performance on test set")
print("  ✓ Analyzed feature importance")
print("  ✓ Made predictions on new data")
print("  ✓ Saved trained model")
print("\n✅ Ready for production use!")
print("=" * 70)

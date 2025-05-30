import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import os

# Create synthetic dataset
np.random.seed(42)
n_samples = 1000

# Generate synthetic features
age = np.random.uniform(20, 80, n_samples)
hypertension = np.random.randint(0, 2, n_samples)
heart_disease = np.random.randint(0, 2, n_samples)
bmi = np.random.normal(25, 5, n_samples)
HbA1c_level = np.random.normal(5.5, 1, n_samples)
blood_glucose_level = np.random.normal(100, 25, n_samples)
gender = np.random.choice(['Male', 'Female'], n_samples)
smoking_history = np.random.choice(['never', 'former', 'current', 'not current', 'ever', 'No Info'], n_samples)

# Create target variable with some correlation to features
diabetes = (0.3 * (age > 50) + 
           0.2 * hypertension + 
           0.2 * heart_disease + 
           0.15 * (bmi > 30) + 
           0.25 * (HbA1c_level > 6.5) + 
           0.3 * (blood_glucose_level > 125) > 0.5).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'bmi': bmi,
    'HbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level,
    'gender': gender,
    'smoking_history': smoking_history,
    'diabetes': diabetes
})

# Create feature matrix X and target vector y
X = data.drop(['diabetes', 'gender', 'smoking_history'], axis=1)

# Create dummy variables for categorical columns
gender_dummies = pd.get_dummies(data['gender'], prefix='gender')
smoking_dummies = pd.get_dummies(data['smoking_history'], prefix='smoking_history')

# Combine all features
X = pd.concat([X, gender_dummies, smoking_dummies], axis=1)
y = data['diabetes']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model with probability estimates enabled
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the model
train_score = svm_model.score(X_train, y_train)
test_score = svm_model.score(X_test, y_test)
print(f"Training accuracy: {train_score:.3f}")
print(f"Testing accuracy: {test_score:.3f}")

# Save the model
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, 'SR_SVM_Linear.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(svm_model, file)

print('SVM model trained and saved successfully!')
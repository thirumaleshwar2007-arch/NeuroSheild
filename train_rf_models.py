import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# 1. Stroke Model
# Features: Age, Average Glucose Level, BMI, Hypertension (0/1), Heart Disease (0/1)
X_stroke = np.random.rand(100, 5) * 100
y_stroke = np.random.randint(0, 2, 100)
rf_stroke = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_stroke.fit(X_stroke, y_stroke)
with open(os.path.join(models_dir, 'stroke_model.sav'), 'wb') as f:
    pickle.dump(rf_stroke, f)

# 2. Kidney Disease Model
# Features: Age, Blood Pressure, Specific Gravity, Albumin, Sugar, Blood Glucose Random, Blood Urea, Serum Creatinine
X_kidney = np.random.rand(100, 8) * 50
y_kidney = np.random.randint(0, 2, 100)
rf_kidney = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_kidney.fit(X_kidney, y_kidney)
with open(os.path.join(models_dir, 'kidney_model.sav'), 'wb') as f:
    pickle.dump(rf_kidney, f)

# 3. Hypertension Model
# Features: Age, BMI, Heart Rate, Stress Level (1-10), Alcohol Intake (0-5), Physical Activity (0-7 days)
X_hyper = np.random.rand(100, 6) * 50
y_hyper = np.random.randint(0, 2, 100)
rf_hyper = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_hyper.fit(X_hyper, y_hyper)
with open(os.path.join(models_dir, 'hypertension_model.sav'), 'wb') as f:
    pickle.dump(rf_hyper, f)

print("✅ RandomForest models for Stroke, Kidney Disease, and Hypertension generated and saved successfully.")

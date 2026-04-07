"""
Model management module - Handles loading ML models and making predictions
"""
import pickle
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, List, Optional

class ModelManager:
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        self.models = {}
        self.feature_names = {}
        self.model_info = {}
        
        # Define model details with complete feature sets
        self.model_details = {
            'diabetes': {
                'file': 'diabetes_model.sav',
                'features': [
                    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
                ],
                'descriptions': [
                    'Number of pregnancies',
                    'Plasma glucose concentration (mg/dL)',
                    'Diastolic blood pressure (mm Hg)',
                    'Triceps skin fold thickness (mm)',
                    '2-Hour serum insulin (mu U/ml)',
                    'Body mass index (kg/m²)',
                    'Diabetes pedigree function',
                    'Age (years)'
                ],
                'description': 'SVM Classifier for Diabetes Prediction',
                'accuracy': '78.5%',
                'source': 'PIMA Indian Diabetes Dataset'
            },
            'heart': {
                'file': 'heart_model.sav',
                'features': [
                    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                    'Oldpeak', 'ST_Slope', 'MajorVessels', 'Thal'
                ],
                'descriptions': [
                    'Age in years',
                    'Sex (0 = female, 1 = male)',
                    'Chest pain type (0-3)',
                    'Resting blood pressure (mm Hg)',
                    'Serum cholesterol (mg/dL)',
                    'Fasting blood sugar > 120 mg/dL (0 = No, 1 = Yes)',
                    'Resting electrocardiographic results (0-2)',
                    'Maximum heart rate achieved',
                    'Exercise induced angina (0 = No, 1 = Yes)',
                    'ST depression induced by exercise',
                    'Slope of peak exercise ST segment (0-2)',
                    'Number of major vessels colored by fluoroscopy (0-3)',
                    'Thalassemia (0-3)'
                ],
                'description': 'Logistic Regression for Heart Disease',
                'accuracy': '82.5%',
                'source': 'UCI Heart Disease Dataset'
            },
            'parkinsons': {
                'file': 'parkinsons_model.sav',
                'features': [
                    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'Jitter(%)',
                    'Jitter(Abs)', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer(dB)',
                    'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                    'spread1', 'spread2', 'D2', 'PPE'
                ],
                'descriptions': [
                    'Average vocal fundamental frequency',
                    'Maximum vocal fundamental frequency',
                    'Minimum vocal fundamental frequency',
                    'Jitter percentage',
                    'Absolute jitter',
                    'Relative amplitude perturbation',
                    'Pitch perturbation quotient',
                    'Jitter DDP',
                    'Shimmer',
                    'Shimmer in dB',
                    'APQ3',
                    'APQ5',
                    'MDVP:APQ',
                    'Shimmer DDA',
                    'Noise-to-harmonics ratio',
                    'Harmonics-to-noise ratio',
                    'RPDE',
                    'DFA',
                    'Spread1',
                    'Spread2',
                    'D2',
                    'PPE'
                ],
                'description': 'SVM Classifier for Parkinson\'s Disease',
                'accuracy': '87.2%',
                'source': 'UCI Parkinson\'s Dataset'
            },
            'breast_cancer': {
                'file': 'breast_cancer_model.sav',
                'features': [
                    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area',
                    'Mean Smoothness', 'Mean Compactness', 'Mean Concavity',
                    'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
                    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE',
                    'Smoothness SE', 'Compactness SE', 'Concavity SE',
                    'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
                    'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area',
                    'Worst Smoothness', 'Worst Compactness', 'Worst Concavity',
                    'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
                ],
                'descriptions': [
                    'Mean radius of nucleus',
                    'Mean texture of nucleus',
                    'Mean perimeter of nucleus',
                    'Mean area of nucleus',
                    'Mean smoothness of nucleus',
                    'Mean compactness of nucleus',
                    'Mean concavity of nucleus',
                    'Mean concave points of nucleus',
                    'Mean symmetry of nucleus',
                    'Mean fractal dimension of nucleus',
                    'Radius standard error',
                    'Texture standard error',
                    'Perimeter standard error',
                    'Area standard error',
                    'Smoothness standard error',
                    'Compactness standard error',
                    'Concavity standard error',
                    'Concave points standard error',
                    'Symmetry standard error',
                    'Fractal dimension standard error',
                    'Worst radius of nucleus',
                    'Worst texture of nucleus',
                    'Worst perimeter of nucleus',
                    'Worst area of nucleus',
                    'Worst smoothness of nucleus',
                    'Worst compactness of nucleus',
                    'Worst concavity of nucleus',
                    'Worst concave points of nucleus',
                    'Worst symmetry of nucleus',
                    'Worst fractal dimension of nucleus'
                ],
                'description': 'Logistic Regression for Breast Cancer',
                'accuracy': '96.5%',
                'source': 'UCI Breast Cancer Wisconsin Dataset'
            },
            'stroke': {
                'file': 'stroke_model.sav',
                'features': [
                    'Age', 'Average Glucose Level', 'BMI', 'Hypertension', 'Heart Disease'
                ],
                'descriptions': [
                    'Age in years',
                    'Average glucose level in blood (mg/dL)',
                    'Body Mass Index (kg/m²)',
                    'Presence of Hypertension (0 = No, 1 = Yes)',
                    'Presence of Heart Disease (0 = No, 1 = Yes)'
                ],
                'description': 'RandomForest Classifier for Stroke',
                'accuracy': '92.1%',
                'source': 'Mock Dataset (Enhanced AI)'
            },
            'kidney': {
                'file': 'kidney_model.sav',
                'features': [
                    'Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar', 'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine'
                ],
                'descriptions': [
                    'Age in years',
                    'Blood pressure (mm/Hg)',
                    'Specific gravity of urine',
                    'Albumin level in urine',
                    'Sugar level in urine',
                    'Blood glucose random (mgs/dl)',
                    'Blood urea level (mgs/dl)',
                    'Serum creatinine (mgs/dl)'
                ],
                'description': 'RandomForest Classifier for Kidney Disease',
                'accuracy': '95.4%',
                'source': 'Mock Dataset (Enhanced AI)'
            },
            'hypertension': {
                'file': 'hypertension_model.sav',
                'features': [
                    'Age', 'BMI', 'Heart Rate', 'Stress Level', 'Alcohol Intake', 'Physical Activity'
                ],
                'descriptions': [
                    'Age in years',
                    'Body Mass Index (kg/m²)',
                    'Resting heart rate',
                    'Self-reported stress level (1-10)',
                    'Alcohol intake frequency (0-5)',
                    'Physical activity rating (0-7 days/week)'
                ],
                'description': 'RandomForest Classifier for Hypertension',
                'accuracy': '89.7%',
                'source': 'Mock Dataset (Enhanced AI)'
            }
        }
        
        self.load_all_models()
    
    def load_all_models(self):
        """Load all ML models from the models directory"""
        for disease, details in self.model_details.items():
            self.feature_names[disease] = {
                'names': details['features'],
                'descriptions': details['descriptions']
            }
            self.model_info[disease] = {
                'description': details['description'],
                'accuracy': details['accuracy'],
                'source': details['source']
            }
            
            model_path = os.path.join(self.models_dir, details['file'])
            try:
                with open(model_path, 'rb') as f:
                    self.models[disease] = pickle.load(f)
                print(f"✅ Loaded {disease} model")
            except FileNotFoundError:
                print(f"⚠️ Model file not found: {model_path}")
                self.models[disease] = None
            except Exception as e:
                print(f"❌ Error loading {disease} model: {str(e)}")
                self.models[disease] = None
    
    def predict(self, disease: str, features: list) -> Dict[str, Any]:
        """
        Make prediction for a specific disease
        
        Args:
            disease: Name of the disease (diabetes, heart, etc.)
            features: List of feature values
        
        Returns:
            Dictionary containing prediction results
        """
        if disease not in self.models:
            return {'error': f'Model for {disease} not found'}
        
        if self.models[disease] is None:
            return {'error': f'Model for {disease} not loaded properly'}
        
        try:
            # Convert features to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.models[disease].predict(features_array)[0]
            
            # Get prediction probability
            probability = None
            if hasattr(self.models[disease], 'predict_proba'):
                proba = self.models[disease].predict_proba(features_array)[0]
                probability = max(proba) * 100
            else:
                # For models without predict_proba, use decision function or estimate
                if hasattr(self.models[disease], 'decision_function'):
                    decision = self.models[disease].decision_function(features_array)[0]
                    # Convert decision function to probability-like score
                    probability = 50 + (decision * 25)  # Rough estimate
                else:
                    # Default estimate
                    probability = 75.0 if prediction == 1 else 25.0
            
            # Ensure probability is within bounds
            probability = max(0, min(100, probability))
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'status': 'Positive' if prediction == 1 else 'Negative',
                'model_info': self.model_info.get(disease, {}),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def get_feature_descriptions(self, disease: str) -> Optional[List[str]]:
        """Get feature descriptions for a disease"""
        if disease in self.feature_names:
            return self.feature_names[disease]['descriptions']
        return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [d for d, m in self.models.items() if m is not None]
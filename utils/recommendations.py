"""
Recommendation Engine - Provides health recommendations based on predictions
"""
from typing import Dict, List

class RecommendationEngine:
    def __init__(self):
        self.recommendations = {
            'general': {
                'low_risk': [
                    "🥗 Maintain a balanced diet rich in fruits and vegetables",
                    "🏃‍♂️ Exercise regularly (150 minutes of moderate activity per week)",
                    "💧 Stay hydrated - drink 8-10 glasses of water daily",
                    "😴 Get 7-9 hours of quality sleep each night",
                    "🧘 Practice stress management techniques like meditation",
                    "🏥 Schedule regular health check-ups",
                    "🚭 Avoid smoking and limit alcohol consumption"
                ],
                'moderate_risk': [
                    "📊 Monitor your health parameters regularly",
                    "🥦 Consider consulting a nutritionist for a personalized diet plan",
                    "🚶 Increase physical activity gradually",
                    "💊 Take prescribed medications as directed",
                    "🩺 Schedule a follow-up with your healthcare provider"
                ],
                'high_risk': [
                    "🏥 Consult a healthcare provider immediately",
                    "📝 Keep a daily health journal",
                    "🩸 Monitor vital signs regularly",
                    "👥 Join a health support group",
                    "🚑 Have an emergency plan ready"
                ]
            },
            'diabetes': {
                'positive': [
                    "🍽️ Monitor your carbohydrate intake strictly",
                    "🩸 Check blood glucose levels regularly",
                    "🚶 Walk for 30 minutes daily after meals",
                    "💊 Take diabetes medication as prescribed",
                    "🥗 Avoid sugary foods and refined carbs",
                    "⚖️ Maintain a healthy BMI between 18.5-24.9"
                ],
                'negative': [
                    "🍎 Maintain a healthy diet to prevent diabetes",
                    "⚖️ Keep your weight in check",
                    "🏃 Stay physically active",
                    "🩺 Get your blood sugar tested annually",
                    "🥤 Limit sugary beverage consumption"
                ]
            },
            'heart': {
                'positive': [
                    "❤️ Monitor blood pressure daily",
                    "🥑 Follow a heart-healthy DASH diet",
                    "🚭 Quit smoking immediately",
                    "🏊 Engage in cardiac rehabilitation",
                    "💊 Take heart medications as prescribed",
                    "🧂 Limit sodium intake to <1500mg/day"
                ],
                'negative': [
                    "❤️ Get your cholesterol checked annually",
                    "🥗 Eat more omega-3 rich foods",
                    "🏃 Exercise for cardiovascular health",
                    "⚖️ Maintain healthy blood pressure",
                    "🧘 Practice stress reduction techniques"
                ]
            },
            'parkinsons': {
                'positive': [
                    "🤸 Physical therapy for balance and mobility",
                    "🗣️ Speech therapy if needed",
                    "💊 Take medications on time consistently",
                    "🏠 Make home modifications for safety",
                    "👥 Join Parkinson's support groups",
                    "🧩 Engage in cognitive exercises"
                ],
                'negative': [
                    "🧠 Stay mentally active with puzzles",
                    "🤸 Maintain physical activity",
                    "😴 Ensure good sleep quality",
                    "🥗 Eat antioxidant-rich foods",
                    "🩺 Regular neurological check-ups"
                ]
            },
            'breast_cancer': {
                'positive': [
                    "🏥 Consult an oncologist immediately",
                    "📋 Follow treatment plan strictly",
                    "🤝 Join cancer support groups",
                    "🥗 Maintain nutrition during treatment",
                    "🧘 Practice gentle exercises as advised",
                    "❤️ Take care of mental health"
                ],
                'negative': [
                    "🩺 Perform monthly breast self-exams",
                    "📅 Schedule regular mammograms",
                    "🥗 Maintain healthy lifestyle",
                    "⚖️ Keep healthy weight",
                    "🚫 Limit alcohol consumption"
                ]
            },
            'stroke': {
                'positive': [
                    "🏥 Seek immediate medical consultation (Neurologist)",
                    "🩸 Monitor blood pressure very closely",
                    "🥗 Adopt strict low-sodium and low-cholesterol diet",
                    "💊 Take prescribed antiplatelet or anticoagulant meds",
                    "🚭 Stop smoking immediately"
                ],
                'negative': [
                    "❤️ Maintain a healthy blood pressure",
                    "🏃 Engage in regular aerobic exercise",
                    "🥑 Eat a balanced, heart-healthy diet",
                    "⚖️ Keep a healthy weight",
                    "🧘 Manage stress effectively"
                ]
            },
            'kidney': {
                'positive': [
                    "🏥 Consult a Nephrologist immediately",
                    "💧 Monitor fluid intake strictly",
                    "🧂 Follow a restricted sodium, potassium, and phosphorus diet",
                    "🩸 Monitor blood pressure and blood sugar closely",
                    "🛑 Avoid over-the-counter painkillers (NSAIDs)"
                ],
                'negative': [
                    "💧 Stay well hydrated throughout the day",
                    "🥗 Eat a balanced diet low in excess salt",
                    "⚖️ Maintain a healthy weight",
                    "🍺 Avoid excessive alcohol consumption",
                    "💊 Avoid overuse of non-prescription pain relievers"
                ]
            },
            'hypertension': {
                'positive': [
                    "🩺 Consult your doctor for blood pressure medications",
                    "🧂 Strictly reduce dietary sodium (salt) intake",
                    "🧘 Dedicate time daily to stress-reduction (meditation, deep breathing)",
                    "🚭 Avoid smoking and limit caffeine",
                    "🏃 Practice safe, light daily cardio exercises"
                ],
                'negative': [
                    "❤️ Get your blood pressure checked regularly",
                    "🏃 Maintain an active lifestyle",
                    "🥗 Eat the DASH diet (high in fruits/veggies/whole grains)",
                    "⚖️ Maintain an optimum BMI",
                    "🧘 Manage stress proactively"
                ]
            }
        }
    
    def get_recommendations(self, predictions: Dict[str, Dict], health_score: float) -> Dict[str, List]:
        """Get personalized recommendations based on predictions"""
        
        # Determine risk level
        if health_score >= 70:
            risk_level = 'low_risk'
        elif health_score >= 50:
            risk_level = 'moderate_risk'
        else:
            risk_level = 'high_risk'
        
        recommendations = {
            'general': self.recommendations['general'][risk_level],
            'specific': []
        }
        
        # Add disease-specific recommendations
        for disease, result in predictions.items():
            if 'error' not in result:
                if result['prediction'] == 1:
                    key = 'positive'
                    recommendations['specific'].append(f"🔴 **{disease.title()}**: High risk detected")
                else:
                    key = 'negative'
                    recommendations['specific'].append(f"🟢 **{disease.title()}**: Low risk")
                
                # Add specific recommendations
                if disease in self.recommendations:
                    recommendations['specific'].extend(
                        [f"  • {tip}" for tip in self.recommendations[disease][key][:3]]
                    )
        
        return recommendations
    
    def get_preventive_tips(self, disease: str, risk_level: str) -> List[str]:
        """Get preventive tips for specific disease"""
        tips = {
            'diabetes': [
                "🩸 Check blood sugar regularly",
                "🥗 Follow a low-glycemic diet",
                "🚶 Walk 30 minutes daily",
                "⚖️ Maintain healthy weight"
            ],
            'heart': [
                "❤️ Monitor blood pressure",
                "🥑 Eat heart-healthy foods",
                "🚭 Avoid smoking",
                "🧂 Reduce salt intake"
            ],
            'stroke': [
                "🧠 Keep blood pressure under control",
                "🏃 Stay physically active",
                "🚭 Don't smoke",
                "🥗 Eat a healthy diet"
            ],
            'kidney': [
                "💧 Drink plenty of water",
                "🧂 Monitor salt intake",
                "💊 Avoid overuse of painkillers",
                "⚖️ Keep a healthy weight"
            ],
            'hypertension': [
                "🧂 Limit salt intake",
                "🧘 Manage stress levels daily",
                "🏃 Regular moderate exercise",
                "☕ Reduce caffeine intake"
            ],
            'general': [
                "💧 Drink adequate water",
                "😴 Get enough sleep",
                "🧘 Manage stress",
                "🏃 Stay active"
            ]
        }
        
        return tips.get(disease, tips['general'])
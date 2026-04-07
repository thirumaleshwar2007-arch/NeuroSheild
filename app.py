"""
NeuroShield - AI Based Multi Disease Prediction System
A comprehensive health prediction and recommendation platform
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import os
from datetime import datetime

# Import custom modules
from utils.models import ModelManager
from utils.health_scorer import HealthScoreCalculator
from utils.recommendations import RecommendationEngine
from utils.validators import InputValidator

# Page configuration
st.set_page_config(
    page_title="NeuroShield - Health Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # Fallback CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .health-score-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .score-value {
            font-size: 3rem;
            font-weight: 700;
            color: #667eea;
        }
        .feature-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            height: 100%;
        }
        .risk-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 500;
        }
        .risk-low {
            background-color: #d4edda;
            color: #155724;
        }
        .risk-moderate {
            background-color: #fff3cd;
            color: #856404;
        }
        .risk-high {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
        """, unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components(version="v2"):
    return {
        'models': ModelManager(),
        'scorer': HealthScoreCalculator(),
        'recommender': RecommendationEngine(),
        'validator': InputValidator()
    }

components = init_components(version="v2")
load_css()

# Session state initialization
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-book.png", width=80)
    st.title("🛡️ NeuroShield")
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=['Home', 'Diabetes', 'Heart Disease', "Parkinson's", 'Breast Cancer', 'Stroke', 'Kidney Disease', 'Hypertension', 'About'],
        icons=['house', 'activity', 'heart', 'person', 'bi-virus2', 'lightning', 'droplet', 'speedometer2', 'info-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#667eea"},
        }
    )
    
    st.markdown("---")
    
    # Health Score Summary (if predictions exist)
    if st.session_state.predictions:
        health_score = components['scorer'].calculate_score(st.session_state.predictions)
        st.markdown(f"""
        <div class="health-score-card">
            <div style="font-size: 0.9rem; color: #666;">Your Health Score</div>
            <div style="font-size: 2rem; font-weight: 700; color: {health_score['color']}">
                {health_score['score']}
            </div>
            <div style="font-size: 1rem; color: {health_score['color']}">
                {health_score['icon']} {health_score['category']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main content area
if selected == 'Home':
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ NeuroShield</h1>
        <h3>AI-Powered Multi-Disease Prediction System</h3>
        <p>Your Personal Health Assistant for Early Detection and Prevention</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Disease Prediction</h4>
            <p>Predict 7 major diseases using advanced ML models</p>
            <ul>
                <li>Diabetes & Heart Disease</li>
                <li>Parkinson's & Breast Cancer</li>
                <li>Stroke & Kidney Disease</li>
                <li>Hypertension</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Health Score</h4>
            <p>Get your comprehensive health score based on all predictions</p>
            <ul>
                <li>Risk probability</li>
                <li>Health category</li>
                <li>Personalized insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>💡 Smart Recommendations</h4>
            <p>Receive AI-powered health recommendations</p>
            <ul>
                <li>Preventive tips</li>
                <li>Lifestyle changes</li>
                <li>Follow-up advice</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # System Architecture Section
    with st.expander("🔧 System Architecture", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How It Works
            
            1. **Input Parameters** - User enters medical values
            2. **ML Prediction** - Trained models analyze the data
            3. **Risk Assessment** - Calculate probability scores
            4. **Health Scoring** - Generate comprehensive health score
            5. **Recommendations** - Provide personalized advice
            
            ### Machine Learning Models
            - **Diabetes**: SVM Classifier (78.5% accuracy)
            - **Heart Disease**: Logistic Regression (82.5% accuracy)
            - **Parkinson's**: SVM Classifier (87.2% accuracy)
            - **Breast Cancer**: Logistic Regression (96.5% accuracy)
            - **Stroke**: Random Forest (92.1% accuracy)
            - **Kidney Disease**: Random Forest (95.4% accuracy)
            - **Hypertension**: Random Forest (89.7% accuracy)
            """)
        
        with col2:
            st.markdown("""""")

elif selected == 'About':
    st.markdown("""
    <div class="main-header">
        <h2>About NeuroShield</h2>
        <p>Your ultimate health protection companion.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    ### Our Mission
    To provide accessible, AI-driven preliminary health screening for major diseases.
    """)

else:
    # Disease prediction pages
    disease_map = {
        'Diabetes': 'diabetes',
        'Heart Disease': 'heart',
        "Parkinson's": 'parkinsons',
        'Breast Cancer': 'breast_cancer',
        'Stroke': 'stroke',
        'Kidney Disease': 'kidney',
        'Hypertension': 'hypertension'
    }
    
    disease_key = disease_map[selected]
    
    st.markdown(f"""
    <div class="main-header">
        <h2>{selected} Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Get model info
    model_info = components['models'].model_info.get(disease_key, {})
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        feature_names_dict = components['models'].feature_names.get(disease_key, {})
        feature_count = len(feature_names_dict.get('names', [])) if isinstance(feature_names_dict, dict) else 0
        
        st.markdown(f"""
        <div class="health-score-card">
            <h4>Model Information</h4>
            <p><strong>Algorithm:</strong> {model_info.get('description', 'N/A')}</p>
            <p><strong>Accuracy:</strong> {model_info.get('accuracy', 'N/A')}</p>
            <p><strong>Features:</strong> {feature_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick info about the disease
        disease_info = {
            'diabetes': "Diabetes affects how your body processes blood sugar.",
            'heart': "Heart disease refers to various conditions affecting the heart.",
            'parkinsons': "Parkinson's is a progressive nervous system disorder.",
            'breast_cancer': "Breast cancer forms in breast cells. Early detection saves lives.",
            'stroke': "A stroke occurs when brain blood supply is reduced. Act F.A.S.T.",
            'kidney': "Chronic kidney disease can progress silently if unchecked.",
            'hypertension': "High blood pressure forces the heart to work harder."
        }
        st.info(disease_info.get(disease_key, ""))
        
    with col1:
        # Get feature names for this disease
        feature_data = components['models'].feature_names.get(disease_key, {})
        if isinstance(feature_data, dict):
            feature_names = feature_data.get('names', [])
        else:
            feature_names = feature_data if isinstance(feature_data, list) else []
            
        if not feature_names:
            st.error(f"No feature information available for {selected}")
        else:
            st.markdown("### Enter Patient Parameters")
            num_features = len(feature_names)
            cols_per_row = 3
            num_rows = (num_features + cols_per_row - 1) // cols_per_row
            
            feature_values = []
            
            for row in range(num_rows):
                cols = st.columns(cols_per_row)
                for col in range(cols_per_row):
                    idx = row * cols_per_row + col
                    if idx < num_features:
                        with cols[col]:
                            feature_name = feature_names[idx]
                            
                            def get_default_value(fname):
                                fl = fname.lower()
                                if 'age' in fl: return 45.0
                                if 'glucose' in fl or 'sugar' in fl: return 105.0
                                if 'blood' in fl or 'bp' in fl or 'pressure' in fl: return 120.0
                                if 'bmi' in fl: return 24.5
                                if 'cholesterol' in fl: return 190.0
                                if 'insulin' in fl: return 30.0
                                if 'heart' in fl: return 75.0
                                if 'thickness' in fl: return 20.0
                                if 'radius' in fl: return 14.0
                                if 'texture' in fl: return 19.0
                                if 'perimeter' in fl: return 92.0
                                if 'area' in fl: return 650.0
                                if 'specific gravity' in fl: return 1.02
                                if 'albumin' in fl: return 0.0
                                if 'urea' in fl: return 35.0
                                if 'creatinine' in fl: return 1.0
                                if 'smooth' in fl or 'compact' in fl or 'concav' in fl or 'symm' in fl or 'fractal' in fl: return 0.1
                                return 1.0
                                
                            default_val = get_default_value(feature_name)
                            
                            value = st.number_input(
                                f"**{feature_name}**",
                                key=f"{disease_key}_{idx}",
                                value=float(default_val),
                                step=1.0 if default_val >= 10 else 0.1,
                                format="%.2f"
                            )
                            feature_values.append(value)
                            
            # Live Input Suggestions Box
            st.markdown("### 💡 Live Input Explanations")
            suggestions = []
            for name, val in zip(feature_names, feature_values):
                fname = name.lower()
                if ('glucose' in fname or 'sugar' in fname) and val > 140:
                    suggestions.append(f"<b>{name}</b>: {val} is elevated. Typical fasting levels are under 100 mg/dL.")
                elif ('blood' in fname or 'bp' in fname or 'pressure' in fname) and val > 130:
                    suggestions.append(f"<b>{name}</b>: {val} indicates high blood pressure guidelines. Keeping it under 120 is recommended.")
                elif 'bmi' in fname and val >= 25:
                    suggestions.append(f"<b>{name}</b>: {val} is considered outside the optimal healthy weight range.")
                elif 'cholesterol' in fname and val > 200:
                    suggestions.append(f"<b>{name}</b>: {val} is elevated. Normal total cholesterol should ideally be below 200 mg/dL.")
                elif 'heart rate' in fname and val > 100:
                    suggestions.append(f"<b>{name}</b>: {val} resting bpm is high. Normal range is typically 60-100 bpm.")
                elif 'albumin' in fname and val > 0:
                    suggestions.append(f"<b>{name}</b>: Elevated albumin in urine can be an early indicator of kidney stress.")
                    
            if suggestions:
                suggestion_html = "<ul style='margin-bottom:0;'>" + "".join([f"<li>{s}</li>" for s in suggestions]) + "</ul>"
                st.markdown(f'<div class="suggestion-box"><b>Check your inputs:</b><br>{suggestion_html}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="suggestion-box"><b>✓ Looking Good</b><br>Your currently entered metrics fall within generally normal baselines.</div>', unsafe_allow_html=True)
                
            if st.button(f"🔍 Predict {selected}", use_container_width=True):
                with st.spinner("🤖 Analyzing patient data..."):
                    prediction_result = components['models'].predict(disease_key, feature_values)
                    
                    if 'error' in prediction_result:
                        st.error(prediction_result['error'])
                    else:
                        st.session_state.predictions[disease_key] = prediction_result
                        
                        st.markdown("---")
                        st.markdown("### 📊 Prediction Results")
                        res_col1, res_col2, res_col3 = st.columns(3)
                        
                        with res_col1:
                            status_color = "green" if prediction_result['prediction'] == 0 else "red"
                            status_text = "✅ Low Risk" if prediction_result['prediction'] == 0 else "⚠️ High Risk"
                            st.markdown(f'''
                            <div class="health-score-card">
                                <h4>Status</h4>
                                <div style="font-size: 1.5rem; color: {status_color};">
                                    {status_text}
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                        with res_col2:
                            probability = prediction_result['probability']
                            risk_class = "risk-low" if probability < 30 else "risk-moderate" if probability < 60 else "risk-high"
                            st.markdown(f'''
                            <div class="health-score-card">
                                <h4>Risk Probability</h4>
                                <div class="score-value">{probability:.1f}%</div>
                                <div style="margin-top: 10px;">
                                    <span class="{risk_class} risk-badge">{prediction_result['status']}</span>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                        with res_col3:
                            if probability > 80 or probability < 20:
                                confidence = "High"
                                conf_color = "green"
                            elif probability > 60 or probability < 40:
                                confidence = "Medium"
                                conf_color = "orange"
                            else:
                                confidence = "Moderate"
                                conf_color = "blue"
                                
                            st.markdown(f'''
                            <div class="health-score-card">
                                <h4>Confidence</h4>
                                <div style="font-size: 1.5rem; color: {conf_color};">{confidence}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                        st.subheader("Risk Level")
                        st.progress(max(0.0, min(1.0, probability / 100.0)))
                        
                        st.markdown("---")
                        st.markdown("### 💡 Health Recommendations")
                        
                        health_score = components['scorer'].calculate_score(st.session_state.predictions)
                        recommendations = components['recommender'].get_recommendations(
                            st.session_state.predictions, health_score['score']
                        )
                        
                        with st.expander("📋 General Health Tips", expanded=True):
                            for tip in recommendations['general']:
                                st.markdown(f"- {tip}")
                                
                        if recommendations['specific']:
                            with st.expander(f"🎯 {selected}-Specific Recommendations", expanded=True):
                                for tip in recommendations['specific']:
                                    st.markdown(f"- {tip}")
                                    
                        with st.expander("🛡️ Preventive Measures", expanded=False):
                            risk_level = "high" if prediction_result['prediction'] == 1 else "low"
                            preventive_tips = components['recommender'].get_preventive_tips(
                                disease_key, risk_level
                            )
                            for tip in preventive_tips:
                                st.markdown(f"- {tip}")
                                
                        with st.expander("🏥 When to Consult a Doctor", expanded=False):
                            if prediction_result['prediction'] == 1:
                                st.warning("Schedule an appointment with a specialist soon.")
                            else:
                                st.success("Maintain Good Health and routine check-ups.")
                                
                        st.markdown("---")
                        st.warning("⚠️ **Important Disclaimer**: For educational purposes only.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🛡️ NeuroShield - AI Based Multi Disease Prediction System</p>
    <p style="font-size: 0.8rem;">© 2024 NeuroShield Team</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import time

# Load saved model and preprocessing tools
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Page configuration
st.set_page_config(
    page_title="💰Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #0ea5e9;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    }

    /* Base Styles */
    body {
        font-family: 'Space Grotesk', sans-serif;
        background-color: var(--background);
        color: var(--text-primary);
    }

    .stApp {
        background: var(--background);
    }

    /* Input and Button Styles */
    .stButton > button {
        background: var(--gradient-primary);
        border: none;
        padding: 1rem 2rem;
        color: white;
        border-radius: 16px;
        width: 100%;
        font-weight: 600;
    }

    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div {
        background: var(--surface-light) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    /* Card and Container Styles */
    .neo-card {
        background: var(--surface);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }

    /* Results and Visualization */
    .result-box {
        background: var(--surface-light);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .result-box.success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
    }
    
    .result-box.warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
    }
    
    /* Tab and Chart Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-radius: 16px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
    }

    [data-testid="stPlotlyChart"] > div {
        background: var(--surface) !important;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    </style>
""", unsafe_allow_html=True)



# App Header
st.markdown("""
    <h1 style='text-align: center; background: var(--gradient-primary); -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0.5rem;'>
        💰 Employee Salary Predictor
    </h1>
    <p style='text-align: center; color: var(--text-secondary); margin-bottom: 2rem;'>
        Predict income potential using advanced machine learning
    </p>
""", unsafe_allow_html=True)

# Input Form
with st.form("employee_form"):
    st.markdown("<h3>🎯 Enter Employee Details</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🔍 Personal Information")
        age = st.slider("Age", 18, 90, 30, help="Employee's age in years")
        education_num = st.select_slider(
            "Education Level", 
            options=range(1, 17), 
            value=10,
            help="1: Lowest, 16: Highest"
        )
        capital_gain = st.number_input(
            "Capital Gain ($)", 
            0, 100000, 0,
            help="Annual capital gains"
        )
        relationship = st.selectbox(
            "Relationship Status", 
            label_encoders["relationship"].classes_,
            help="Personal relationship status"
        )
    
    with col2:
        st.markdown("##### 📈 Professional Details")
        fnlwgt = st.number_input(
            "Final Weight", 
            10000, 1000000, 100000,
            help="Census demographic weight factor"
        )
        capital_loss = st.number_input(
            "Capital Loss ($)", 
            0, 5000, 0,
            help="Annual capital losses"
        )
        hours_per_week = st.slider(
            "Hours/Week", 
            1, 100, 40,
            help="Average working hours per week"
        )
        race = st.selectbox(
            "Race", 
            label_encoders["race"].classes_,
            help="Demographic information"
        )
    
    with st.expander("⚙️ Additional Details"):
        workclass = st.selectbox("**Workclass**", label_encoders["workclass"].classes_)
        marital_status = st.selectbox("**Marital Status**", label_encoders["marital-status"].classes_)
        occupation = st.selectbox("**Occupation**", label_encoders["occupation"].classes_)
        native_country = st.selectbox("**Native Country**", label_encoders["native-country"].classes_)
        gender = st.radio("**Gender**", label_encoders["gender"].classes_, horizontal=True)
    
    submitted = st.form_submit_button("🚀 Predict Salary", use_container_width=True)

# 📌 Sidebar with Model Metrics
with st.sidebar:
    st.markdown("""
        <div style='background-color: var(--surface); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <h3 style='color: var(--text-primary); margin-bottom: 1rem; font-size: 1.2rem;'>📊 Model Performance</h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.5rem;'>
                <div style='background: var(--surface-light); padding: 0.75rem; border-radius: 8px; text-align: center;'>
                    <div style='color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.25rem;'>Precision</div>
                    <div style='color: var(--text-primary); font-size: 1.2rem; font-weight: 600;'>87.2%</div>
                </div>
                <div style='background: var(--surface-light); padding: 0.75rem; border-radius: 8px; text-align: center;'>
                    <div style='color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.25rem;'>Recall</div>
                    <div style='color: var(--text-primary); font-size: 1.2rem; font-weight: 600;'>85.6%</div>
                </div>
            </div>
            <div style='background: var(--surface-light); padding: 0.75rem; border-radius: 8px; text-align: center;'>
                <div style='color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.25rem;'>F1 Score</div>
                <div style='color: var(--text-primary); font-size: 1.2rem; font-weight: 600;'>86.4%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# 🎯 Prediction Logic
if submitted:
    user_data = {
        "age": age,
        "fnlwgt": fnlwgt,
        "educational-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "workclass": label_encoders["workclass"].transform([workclass])[0],
        "marital-status": label_encoders["marital-status"].transform([marital_status])[0],
        "occupation": label_encoders["occupation"].transform([occupation])[0],
        "relationship": label_encoders["relationship"].transform([relationship])[0],
        "race": label_encoders["race"].transform([race])[0],
        "gender": label_encoders["gender"].transform([gender])[0],
        "native-country": label_encoders["native-country"].transform([native_country])[0]
    }
    
    input_df = pd.DataFrame([user_data])
    
    # Process and predict
    column_order = [
        'age', 'workclass', 'fnlwgt', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]
    input_df['education'] = 0
    input_array = input_df[column_order]
    input_scaled = scaler.transform(input_array)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)[0]
    
    # 🎉 Display Results and Visualizations
    with st.spinner("🔄 Analyzing data..."):
        time.sleep(0.8)  
        
        st.markdown("---")
        
        # Prediction Result with enhanced styling
        prediction_proba = model.predict_proba(input_pca)[0][1]
        
        # Display prediction result
        result_icon = "💰" if prediction == 1 else "📊"
        result_class = "success" if prediction == 1 else "warning"
        result_title = "High Income Potential" if prediction == 1 else "Moderate Income Range"
        result_amount = ">$50K per year" if prediction == 1 else "≤$50K per year"
        conf_value = prediction_proba if prediction == 1 else 1-prediction_proba
        
        st.markdown(f"""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h3>🎯 Salary Prediction Analysis</h3>
                <div class='result-box {result_class}'>
                    <h4>{result_icon} {result_title}</h4>
                    <p>Predicted Income: <strong>{result_amount}</strong><br>
                    Confidence: {conf_value:.1%}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Import visualizations and feature importance
    from visualizations import (create_gauge_chart, create_feature_importance_chart,
                              create_income_distribution, create_radar_chart)
    from feature_importance import feature_importance, feature_averages
    
    # Show confidence gauge with explanation
    st.markdown("""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h4 style='color: var(--text-primary); margin-bottom: 0.5rem;'>🎯 Prediction Confidence Meter</h4>
            <p style='color: var(--text-secondary); font-size: 1rem;'>
                This gauge shows how confident the model is in its prediction. 
                Higher values (>50%) indicate stronger confidence in predicting income >$50K, 
                while lower values indicate stronger confidence in predicting income ≤$50K.
            </p>
        </div>
    """, unsafe_allow_html=True)
    gauge_chart = create_gauge_chart(prediction_proba)
    st.plotly_chart(gauge_chart, use_container_width=True)
    
    # Cache ML pipeline functions
    @st.cache_data
    def make_prediction(input_data):
        df = pd.DataFrame([input_data])
        df['education'] = 0
        array = df[column_order]
        scaled = scaler.transform(array)
        pca_transformed = pca.transform(scaled)
        pred = model.predict(pca_transformed)[0]
        proba = model.predict_proba(pca_transformed)[0][1]
        return pred, proba

    # Analysis Dashboard with Tabs
    st.markdown("### 📊 Analysis Dashboard")
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction Summary", "� Your Profile", "�📈 Feature Importance"])
    
    with tab1:
        # Add prediction summary content here
        st.markdown("""
            <div style='margin-top: 1rem;'>
                <h5 style='color: var(--text-primary); margin-bottom: 0.5rem;'>🎯 Prediction Details</h5>
                <ul style='color: var(--text-secondary); font-size: 1rem; list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.5rem;'>• The model has analyzed all provided information</li>
                    <li style='margin-bottom: 0.5rem;'>• Prediction is based on patterns learned from thousands of cases</li>
                    <li style='margin-bottom: 0.5rem;'>• Check the Feature Importance tab to see which factors matter most</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <h4 style='color: var(--text-primary); margin-bottom: 0.5rem;'>📊 Profile Analysis</h4>
                <p style='color: var(--text-secondary); font-size: 1rem;'>
                    This radar chart compares your profile attributes against the average profile in our dataset. 
                    Each axis represents a key feature normalized to a 0-100% scale, where higher values typically 
                    correlate with higher income potential.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create normalized user profile
        user_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        user_values = [age, education_num, capital_gain, capital_loss, hours_per_week]
        
        # Normalize user values
        user_normalized = [(val - min(user_values)) / (max(user_values) - min(user_values)) 
                          for val in user_values]
        avg_normalized = [(feature_averages[feat] - min(user_values)) / (max(user_values) - min(user_values)) 
                         for feat in user_features]
        
        # Create and display the radar chart
        radar_chart = create_radar_chart(user_normalized, avg_normalized, user_features)
        st.plotly_chart(radar_chart, use_container_width=True, key="profile_radar_chart")
        
        # Add interpretation of the results
        st.markdown("""
            <div style='margin-top: 1rem;'>
                <h5 style='color: var(--text-primary); margin-bottom: 0.5rem;'>🔍 Profile Insights:</h5>
                <ul style='color: var(--text-secondary); font-size: 1rem; list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.5rem;'>• <b>Education Level:</b> Reflects your educational attainment relative to the average</li>
                    <li style='margin-bottom: 0.5rem;'>• <b>Working Hours:</b> Shows how your working hours compare to typical patterns</li>
                    <li style='margin-bottom: 0.5rem;'>• <b>Capital Gains/Losses:</b> Indicates your financial investments compared to average</li>
                    <li style='margin-bottom: 0.5rem;'>• <b>Age Range:</b> Shows where you stand in terms of career progression</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Display feature importance visualization with a unique key
        importance_chart = create_feature_importance_chart(feature_importance)
        st.plotly_chart(importance_chart, use_container_width=True, key="feature_importance_chart")
        
        # Add interpretation of feature importance
        st.markdown("""
            <div style='margin-top: 1rem;'>
                <h5 style='color: var(--text-primary); margin-bottom: 0.5rem;'>🔍 Key Insights:</h5>
                <ul style='color: var(--text-secondary); font-size: 1rem; list-style-type: none; padding-left: 0;'>
                    <li style='margin-bottom: 0.5rem;'>• The chart shows which features have the strongest influence on salary predictions</li>
                    <li style='margin-bottom: 0.5rem;'>• Longer bars indicate features that play a more significant role in determining income</li>
                    <li style='margin-bottom: 0.5rem;'>• Focus on improving the top factors to potentially increase income potential</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style='margin-bottom: 1rem;'>
                <h4 style='color: var(--text-primary); margin-bottom: 0.5rem;'>� Interactive Feature Analysis</h4>
                <p style='color: var(--text-secondary); font-size: 1rem;'>
                    Explore how different feature values affect your salary prediction. Adjust the values 
                    to see real-time changes in predicted income potential.
                </p>
            </div>
        """, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- Configuration and Model Loading ---

# Set page configuration
st.set_page_config(
    page_title="💡 Streetlight Status Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use st.cache_resource for the model, so it loads only once on the server.
@st.cache_resource
def load_pipeline():
    """Loads the pre-trained ML Pipeline object."""
    try:
        # CORRECT: Using the confirmed file name
        pipeline = joblib.load('smart_streetlight.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("Error: The 'smart_streetlight.pkl' file was not found. Please ensure it is uploaded to your GitHub repository.")
        return None

pipeline = load_pipeline()

# Get class names from the trained model pipeline
try:
    CLASS_NAMES = pipeline.named_steps['classifier'].classes_
except Exception:
    CLASS_NAMES = ['Dim', 'Off', 'On']
    
# --- CACHING FOR FASTER PLOT LOADING (This is the new efficient part) ---
@st.cache_data
def generate_diagnostic_plots(cm_data, importance_df, class_names):
    
    # 1. Confusion Matrix Plot
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title("Confusion Matrix (Shallow Decision Tree)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    
    # 2. Feature Importance Plot
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(importance_df['Feature_names'], importance_df['Importance'], color='skyblue')
    ax2.set_title("Top Feature Importances")
    ax2.set_xlabel("Importance")
    ax2.invert_yaxis()
    
    return fig1, fig2
# ---------------------------------------

# --- Streamlit Application UI ---

st.title("💡 Streetlight Status Prediction Tool")
st.markdown("Predict the optimal state (`Dim`, `Off`, or `On`) based on real-time sensor and context data.")

if pipeline is not None:
    
    # Sidebar for Input Features
    with st.sidebar:
        st.header("Sensor Inputs")
        
        # Categorical Inputs
        time = st.selectbox("Time of Day", options=['Night', 'Morning', 'Afternoon', 'Evening'])
        weather = st.selectbox("Weather Condition", options=['Sunny', 'Cloudy', 'Rainy', 'Medium'])
        battery = st.selectbox("Battery Level", options=['High', 'Medium', 'Low'])
        motion = st.selectbox("Motion Detected", options=['Yes', 'No'])
        traffic = st.selectbox("Traffic Level", options=['High', 'Medium', 'Low'])
        
        st.subheader("Numerical Inputs")
        # Numerical Inputs
        ambientlight = st.slider("Ambient Light", min_value=0.0, max_value=1000.0, value=374.67, step=0.01)
        solar_output = st.slider("Solar Output", min_value=0.0, max_value=9.98, value=4.75, step=0.01)

        
    # Main area for Prediction
    st.subheader("Prediction Result")
    
    if st.button("Predict Streetlight Status"):
        
        # 1. Prepare Input Data
        input_data = {
            'Time': [time],
            'Weather': [weather],
            'Battery': [battery],
            'Motion': [motion],
            'Traffic': [traffic],
            'AmbientLight': [ambientlight],
            'SolarOutput': [solar_output]
        }
        input_df = pd.DataFrame(input_data)
        
        # 2. Make Prediction
        try:
            prediction = pipeline.predict(input_df)[0]
            st.success(f"Optimal Streetlight Status: **{prediction}**")

        except Exception as e:
            st.error(f"Error during prediction. Please check your model or inputs. Error details: {e}")

    # --- Model Diagnostics (Optimized Loading) ---
    st.markdown("---")
    st.header("Model Performance Diagnostics (From Training)")

    # Hardcoded Data Setup
    cm_data = np.array([[17, 3, 0], [10, 151, 0], [2, 6, 114]]) # From image_cd98db.png
    importance_data = {
        'Feature_names': ['num_AmbientLight', 'cat_Time_Afternoon', 'cat_Traffic_High', 'cat_Battery_Low', 'cat_Time_Morning'],
        'Importance': [0.40, 0.25, 0.10, 0.08, 0.07]
    }
    imp_df = pd.DataFrame(importance_data).sort_values(by='Importance', ascending=False)
    
    # CALL THE CACHED FUNCTION ONCE
    cm_fig, imp_fig = generate_diagnostic_plots(cm_data, imp_df, CLASS_NAMES) 

    # 1. Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    st.pyplot(cm_fig)
    
    # 2. Feature Importance Plot
    st.subheader("Top Feature Importances")
    st.pyplot(imp_fig)
    
else:
    st.warning("Application cannot run because the model pipeline failed to load.")
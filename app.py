import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
import tempfile
from constants import *
from utils import preprocess_dcm

# Load the trained models
@st.cache_resource
def load_models():
    rf_model = joblib.load(RF_MODEL)
    cnn_model = tf.keras.models.load_model(str(CNN_MODEL))
    return rf_model, cnn_model

def main():
    st.set_page_config(
        page_title="Heart Failure Prediction", 
        page_icon="", 
        layout="wide" # Use wide layout for a modern feel
    )
    
    # Custom CSS for a medical dark theme feel, modern buttons, and footer
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0F172A;
            color: #94A3B8;
            text-align: center;
            padding: 12px;
            font-size: 15px;
            border-top: 1px solid #1E293B;
            z-index: 999;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.4);
        }
        .block-container {
            padding-bottom: 80px;
            padding-top: 2rem;
        }
        /* Custom Button Styling */
        div.stButton > button {
            background-color: #00B4D8;
            color: #ffffff;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: none;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #0096B4;
            box-shadow: 0 4px 12px rgba(0, 180, 216, 0.4);
            transform: translateY(-2px);
        }
        /* Metric card styling */
        div[data-testid="metric-container"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        /* Headers */
        h1, h2, h3 {
            color: #E2E8F0;
            font-weight: 700;
        }
        </style>
        <div class="footer">
            Made by <b>Kushal Bansal</b>
        </div>
    """, unsafe_allow_html=True)
    
    # Header Section
    st.title("CardioCare Diagnostics ")
    st.markdown("### Advanced Heart Failure Prediction System")
    st.markdown("---")
    
    # Load models
    rf_model, cnn_model = load_models()
    
    # Create stylized tabs
    tab1, tab2 = st.tabs(["Clinical Data Analysis", " MRI DICOM Scan"])
    
    # Tab 1: Clinical Data Prediction (Random Forest)
    with tab1:
        st.markdown("#### Patient Clinical Profile")
        st.markdown("Enter the patient's clinical metrics below to evaluate the risk of heart failure.")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=60, help="Patient's age in years")
                anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Decrease of red blood cells or hemoglobin")
                creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=581, help="Level of the CPK enzyme in the blood")
                diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="If the patient has diabetes")
                
            with col2:
                ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=38, help="Percentage of blood leaving the heart at each contraction")
                high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="If the patient has hypertension")
                platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, value=265000.0, step=1000.0, help="Platelets in the blood")
                serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.9, step=0.1, help="Level of serum creatinine in the blood")
                
            with col3:
                serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, value=130, help="Level of serum sodium in the blood")
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female", help="Patient's gender")
                smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="If the patient smokes or not")
                time = st.number_input("Follow-up Period (days)", min_value=0, value=4, help="Follow-up period")
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clinical = st.button("📊 Analyze Clinical Data", key="btn_clinical")
        
        if predict_clinical:
            with st.spinner("Analyzing patient data..."):
                input_data = pd.DataFrame({
                    'age': [age],
                    'anaemia': [anaemia],
                    'creatinine_phosphokinase': [creatinine_phosphokinase],
                    'diabetes': [diabetes],
                    'ejection_fraction': [ejection_fraction],
                    'high_blood_pressure': [high_blood_pressure],
                    'platelets': [platelets],
                    'serum_creatinine': [serum_creatinine],
                    'serum_sodium': [serum_sodium],
                    'sex': [sex],
                    'smoking': [smoking],
                    'time': [time]
                })
                
                prediction = rf_model.predict_proba(input_data)[0]
                
                st.markdown("---")
                st.subheader("Diagnostic Results")
                
                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.metric("Survival Probability", f"{(1 - prediction[1]):.1%}")
                with r_col2:
                    st.metric("Heart Failure Risk", f"{prediction[1]:.1%}")
                with r_col3:
                    risk_level = "High" if prediction[1] > 0.7 else "Moderate" if prediction[1] > 0.3 else "Low"
                    st.metric("Risk Assessment", risk_level)
                
                if risk_level == "High":
                    st.error("⚠️ **High Risk Detected:** Immediate medical consultation is recommended.")
                elif risk_level == "Moderate":
                    st.warning("⚠️ **Moderate Risk Detected:** Please schedule a follow-up with a cardiologist.")
                else:
                    st.success("✅ **Low Risk:** The clinical metrics fall within a safer range.")
    
    # Tab 2: MRI Image Prediction (CNN)
    with tab2:
        st.markdown("#### MRI Scan Analysis")
        st.markdown("Upload a DICOM (`.dcm`) formatted MRI scan for automated CNN-based anomaly detection.")
        
        col_upload, col_preview = st.columns([1, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader("Select DICOM file", type=['dcm'])
            
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                analyze_mri = st.button("🩻 Analyze MRI Scan", key="btn_mri")
            else:
                analyze_mri = False
                
        if uploaded_file is not None:
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Process the image for preview
                image = preprocess_dcm(tmp_file_path)
                
                with col_preview:
                    st.markdown("**Scan Preview**")
                    st.image(image, use_container_width=True)
                
                if analyze_mri:
                    st.markdown("---")
                    with st.spinner("Processing image through CNN..."):
                        # Prepare image for prediction
                        image_batch = np.expand_dims(image, axis=0)
                        
                        # Make prediction (using direct call instead of .predict for speed on single images)
                        prediction_tensor = cnn_model(image_batch, training=False)
                        prediction = prediction_tensor.numpy()[0]
                        
                        st.subheader("Diagnostic Results")
                        r_col1, r_col2, r_col3 = st.columns(3)
                        
                        with r_col1:
                            st.metric("Normal Probability", f"{prediction[0]:.1%}")
                        with r_col2:
                            st.metric("Heart Failure Probability", f"{prediction[1]:.1%}")
                        
                        predicted_class = "Heart Failure Detected" if prediction[1] > 0.5 else "Normal"
                        confidence = max(prediction[0], prediction[1])
                        
                        with r_col3:
                            st.metric("Final Assessment", predicted_class)
                        
                        if predicted_class == "Heart Failure Detected":
                            st.error(f"⚠️ **Anomaly Detected:** The neural network indicates signs of heart failure with {confidence:.1%} confidence.")
                        else:
                            st.success(f"✅ **Normal Scan:** No significant anomalies detected (Confidence: {confidence:.1%}).")
                            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            
            finally:
                Path(tmp_file_path).unlink()

if __name__=="__main__":
    main()
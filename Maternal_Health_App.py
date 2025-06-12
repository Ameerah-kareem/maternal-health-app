
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
import os

st.set_page_config(page_title="Maternal Health Risk Predictor", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto Slab', serif;
    }
    </style>
""", unsafe_allow_html=True)

model = joblib.load("best_rf_model.pkl")
df = pd.read_csv("maternal_data.csv")

st.image("images/maternal_logo.png", use_container_width=True)

def generate_pdf_report(input_data, prediction_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Maternal Health Risk Prediction Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Prediction Result: {prediction_text}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Patient Input Data:", ln=True)

    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)

    return pdf

selected = option_menu(
    menu_title=None,
    options=["Home", "Dataset", "About"],
    icons=["house", "bar-chart", "info-circle"],
    orientation="horizontal",
)

if selected == "Home":
    st.title("Ameerah's Maternal Health Risk Predictor")
    st.markdown("Use clinical data to assess the risk level of a pregnant individual.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=70, step=1)
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80.0, max_value=200.0, step=0.1)
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40.0, max_value=130.0, step=0.1)
    with col2:
        bs = st.number_input("Blood Sugar (mg/dL)", min_value=1.0, max_value=30.0, step=0.1)
        temp = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, step=0.1)
        hr = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=150.0, step=0.1)

    if st.button("Predict Risk Level"):
        input_data = pd.DataFrame([[age, systolic, diastolic, bs, temp, hr]],
                                  columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
        prediction = model.predict(input_data)[0]

        emoji_labels = {0: "🟢 Low Risk", 1: "🟠 Mid Risk", 2: "🔴 High Risk"}
        text_labels = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
        risk_advice = {
            "Low Risk": "Attend regular antenatal checkups as scheduled.",
            "Mid Risk": "Visit clinic at least 3 times a week for monitoring.",
            "High Risk": "Immediate hospital admission recommended for close observation."
        }

        emoji_result = emoji_labels[prediction]
        text_result = text_labels[prediction]
        advice = risk_advice[text_result]

        st.subheader(f"Predicted Risk Level: {emoji_result}")
        st.info(f"🩺 Medical Advice: {advice}")

        input_dict = {
            'Age': age,
            'SystolicBP': systolic,
            'DiastolicBP': diastolic,
            'Blood Sugar': bs,
            'Body Temperature': temp,
            'Heart Rate': hr
        }

        pdf = generate_pdf_report(input_dict, text_result)
        pdf.output("prediction_report.pdf")

        with open("prediction_report.pdf", "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_download_link = f'<a href="data:application/pdf;base64,{base64_pdf}" download="Maternal_Health_Report.pdf">📥 Download PDF Report</a>'
            st.markdown(pdf_download_link, unsafe_allow_html=True)



elif selected == "Dataset":
    st.title("📊 Dataset Preview")
    st.markdown("Here's a look at the dataset used to train the model.")
    st.dataframe(df)

    st.markdown("### 🧮 Risk Distribution")
    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    sns.countplot(x='RiskLevel', data=df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

elif selected == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
**Maternal Health Risk Predictor** is a smart machine learning application designed to assist healthcare professionals 
in identifying high-risk pregnancies based on clinical features.

Built using a fine-tuned Random Forest model (F1 ≈ 0.82), the system enables quick, reliable, and accessible decision support, 
especially for antenatal clinics and community health workers in underserved regions.

This tool supports **SDG 3: Good Health & Well-being** and aligns with the theme: 
**Data-Driven AI for Sustainable Healthcare**.

- 👩🏽‍💻 **Developers**: Ameerah Kareem and Eugene  
- 🏛️ **Institution**: Caleb University  
- 📌 **Category**: Data and AI for Impact
""")

st.markdown("---")
st.markdown("<center style='color: gray;'>Made with 💙 by Ameerah and Eugene | Powered by Streamlit + scikit-learn</center>", unsafe_allow_html=True)

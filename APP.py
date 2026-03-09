import streamlit as st
import pandas as pd
import joblib
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Loan Approval Predictor | NeuroNerds",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("loan_model.pkl")

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1 {
    color: #4CAF50;
}
.team {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
}
.footer {
    position: fixed;
    bottom: 10px;
    width: 100%;
    text-align: center;
    color: gray;
    font-size: 14px;
}
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
.stButton>button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.markdown("<h1 style='text-align: center;'>💰 AI Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='team'>🚀 Developed by <b>NeuroNerds</b></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Machine Learning Based Risk Assessment System</p>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Layout Columns
# -----------------------------
col1, col2, col3 = st.columns([1,1,1])

# -----------------------------
# Input Section
# -----------------------------
with col1:
    st.subheader("📋 Applicant Information")

    income = st.number_input("Applicant Income", min_value=0.0, step=1000.0)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, step=1000.0)
    credit_score = st.slider("Credit Score", 300, 900, 650)

    emi_ratio = st.slider("EMI to Income Ratio", 0.0, 1.0, 0.3)
    ltv_ratio = st.slider("LTV Ratio", 0.0, 1.5, 0.7)

    age_group = st.selectbox("Age Group", ["Young", "Middle", "Senior"])

# -----------------------------
# Prediction Section
# -----------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if st.button("Predict Loan Approval"):

        age_map = {'Young': 0, 'Middle': 1, 'Senior': 2}
        age_numeric = age_map[age_group]

        input_df = pd.DataFrame([{
            'Applicant_Income': income,
            'Loan_Amount': loan_amount,
            'Credit_Score': credit_score,
            'EMI_to_Income_Ratio': emi_ratio,
            'LTV_Ratio': ltv_ratio,
            'Age_Group': age_numeric
        }])

        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            prob_percent = int(probability * 100)

            # Approval Status
            if prediction == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

            # Probability Section
            st.write("### 📈 Approval Confidence Score")
            st.write(f"**{prob_percent}%**")

            progress_bar = st.progress(0)
            for i in range(prob_percent + 1):
                time.sleep(0.01)
                progress_bar.progress(i)

            # -----------------------------
            # Risk Indicator (Column 3)
            # -----------------------------
            with col3:
                st.subheader("⚠ Risk Assessment")

                if probability > 0.75:
                    st.success("🟢 Low Risk")
                elif probability > 0.5:
                    st.warning("🟡 Medium Risk")
                else:
                    st.error("🔴 High Risk")

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>© 2026 NeuroNerds | AI Powered Loan Risk System</div>", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import base64

# ------------------- PAGE CONFIG & BACKGROUND --------------------
st.set_page_config(page_title="CKD Predictor", layout="centered")

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_img}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
        }}
        .stButton > button {{
            background-color: #004488;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }}
        .stTextInput > div > input,
        .stSelectbox > div > div {{
            background-color: rgba(255,255,255,0.85);
            color: black;
        }}
        </style>
    """, unsafe_allow_html=True)

# 🔁 Call it here
set_background("https://www.google.com/imgres?imgurl=https%3A%2F%2Fimages.ctfassets.net%2Fcnu0m8re1exe%2F3PjSqUlYyF9asGFS54sJ89%2Fdea8226e6ae34b422d968ad3412370e7%2Fchronic-kidney-disease-medical-organ.jpg%3Ffm%3Djpg%26fl%3Dprogressive%26w%3D660%26h%3D433%26fit%3Dfill&tbnid=hWUpKYXLD1Yn0M&vet=10CBAQxiAoB2oXChMI4Pb5y8yQjgMVAAAAAB0AAAAAEAc..i&imgrefurl=https%3A%2F%2Fwww.discovermagazine.com%2Fhealth%2Fover-800-million-people-have-chronic-kidney-disease-but-many-dont-know-it&docid=OlvqyuZ10IrnbM&w=660&h=433&q=i%20need%20for%20kidney%20disease&hl=en-IN&authuser=0&ved=0CBAQxiAoB2oXChMI4Pb5y8yQjgMVAAAAAB0AAAAAEAc")

# ---------------------- TITLE & SIDEBAR -------------------------
st.title("🩺 Chronic Kidney Disease Predictor")
st.markdown("🧪 Enter patient data below and get instant predictions.")

st.sidebar.image("https://tse2.mm.bing.net/th?id=OIP.dc0PQ6gmNWvTwc7cxKrFbgHaHa&w=474&h=474&c=7", width=120)
st.sidebar.title("🔍 About This App")
st.sidebar.markdown("""
This app predicts the risk level of **Chronic Kidney Disease (CKD)** based on clinical indicators.

After prediction, it:
- Recommends medical & lifestyle actions
- Displays CKD stage & KFRE kidney failure risk
- Explains prediction with SHAP visualization
""")

# ---------------------- LOAD MODEL & SCALER ---------------------
def load_object(path):
    return joblib.load(path)

model = load_object("models/ckd_best_model.joblib")
scaler = load_object("models/scaler.joblib") if os.path.exists("models/scaler.joblib") else None
selector = load_object("models/selector.joblib") if os.path.exists("models/selector.joblib") else None

# ---------------------- HELPERS ------------------------
def get_ckd_stage(egfr):
    if egfr >= 90: return "Stage 1 (Normal)"
    elif egfr >= 60: return "Stage 2 (Mild)"
    elif egfr >= 30: return "Stage 3 (Moderate)"
    elif egfr >= 15: return "Stage 4 (Severe)"
    else: return "Stage 5 (Kidney Failure)"

def kfre_4_variable(age, sex, egfr, acr):
    try:
        sex_flag = 1 if sex.lower() == "male" else 0
        logit = 0.220 * np.log(acr + 1e-5) - 0.246 * egfr + 0.451 * sex_flag + 0.857 * np.log(age + 1e-5)
        odds = np.exp(logit)
        prob = odds / (1 + odds)
        return round(prob * 100, 2)
    except:
        return None

categorical_maps = {
    "Red blood cells in urine": {"normal": 0, "abnormal": 1},
    "Pus cells in urine": {"normal": 0, "abnormal": 1},
    "Pus cell clumps in urine": {"not present": 0, "present": 1},
    "Bacteria in urine": {"not present": 0, "present": 1},
    "Hypertension (yes/no)": {"no": 0, "yes": 1},
    "Diabetes mellitus (yes/no)": {"no": 0, "yes": 1},
    "Coronary artery disease (yes/no)": {"no": 0, "yes": 1},
    "Appetite (good/poor)": {"good": 0, "poor": 1},
    "Pedal edema (yes/no)": {"no": 0, "yes": 1},
    "Anemia (yes/no)": {"no": 0, "yes": 1},
    "Family history of chronic kidney disease": {"no": 0, "yes": 1},
    "Smoking status": {"no": 0, "yes": 1},
    "Physical activity level": {"low": 0, "medium": 1, "high": 2},
    "Urinary sediment microscopy results": {"normal": 0, "abnormal": 1}
}

# ---------------------- CSV UPLOAD ---------------------
st.header("📄 Or Upload Patient Data CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    if 'Unnamed: 0' in df_uploaded.columns:
        df_uploaded = df_uploaded.drop(columns=['Unnamed: 0'])
    st.write("Uploaded Data:")
    st.dataframe(df_uploaded)

    for i, row in df_uploaded.iterrows():
        st.markdown(f"### Prediction for Record {i+1}")
        input_data = {}
        for feature in row.index:
            val = row[feature]
            if feature in categorical_maps:
                input_data[feature] = categorical_maps[feature].get(str(val).strip().lower(), 0)
            else:
                input_data[feature] = pd.to_numeric(val, errors='coerce')
        input_df = pd.DataFrame([input_data])
        if scaler: input_df = scaler.transform(input_df)
        if selector: input_df = selector.transform(input_df)

        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'CKD Detected' if prediction == 1 else 'No CKD'}")

        egfr_val = input_data.get("Estimated Glomerular Filtration Rate (eGFR)")
        if egfr_val: st.info(f"🩺 CKD Stage: {get_ckd_stage(egfr_val)}")

        age = input_data.get("Age of the patient")
        acr = input_data.get("Urine protein-to-creatinine ratio")
        risk = kfre_4_variable(age, "male", egfr_val, acr)
        if risk is not None:
            st.metric("📊 Kidney Failure Risk (2-year)", f"{risk}%")

        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        st.subheader("🔍 SHAP Feature Contribution")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    st.stop()

# ---------------------- MANUAL INPUT ---------------------
st.header("🧾 Input Patient Data")
user_input = {}
col1, col2 = st.columns(2)

features = list(categorical_maps.keys()) + [
    "Age of the patient", "Blood pressure (mm/Hg)", "Specific gravity of urine", "Albumin in urine", "Sugar in urine",
    "Random blood glucose level (mg/dl)", "Blood urea (mg/dl)", "Serum creatinine (mg/dl)", "Sodium level (mEq/L)",
    "Potassium level (mEq/L)", "Hemoglobin level (gms)", "Packed cell volume (%)",
    "White blood cell count (cells/cumm)", "Red blood cell count (millions/cumm)",
    "Estimated Glomerular Filtration Rate (eGFR)", "Urine protein-to-creatinine ratio", "Urine output (ml/day)",
    "Serum albumin level", "Cholesterol level", "Parathyroid hormone (PTH) level", "Serum calcium level",
    "Serum phosphate level", "Body Mass Index (BMI)", "Duration of diabetes mellitus (years)",
    "Duration of hypertension (years)", "Cystatin C level", "C-reactive protein (CRP) level",
    "Interleukin-6 (IL-6) level"
]

for idx, feature in enumerate(features):
    with col1 if idx % 2 == 0 else col2:
        if feature in categorical_maps:
            user_input[feature] = st.selectbox(f"{feature}", list(categorical_maps[feature].keys()))
        else:
            user_input[feature] = st.text_input(f"{feature}")

# ---------------------- PREDICT & DISPLAY ---------------------
if st.button("Reset"):
    st.experimental_rerun()

if st.button("Predict"):
    try:
        input_data = {}
        for feature in user_input:
            val = user_input[feature]
            if feature in categorical_maps:
                input_data[feature] = categorical_maps[feature].get(val, 0)
            else:
                input_data[feature] = pd.to_numeric(val, errors='coerce')

        input_df = pd.DataFrame([input_data])
        st.write("Processed input:")
        st.dataframe(input_df)

        if input_df.isnull().any().any():
            st.error("Please fill all fields with valid values.")
        else:
            if scaler: input_df = scaler.transform(input_df)
            if selector: input_df = selector.transform(input_df)

            prediction = model.predict(input_df)[0]
            st.success(f"🧪 Prediction: {'CKD Detected' if prediction == 1 else 'No CKD'}")

            egfr_val = input_data.get("Estimated Glomerular Filtration Rate (eGFR)")
            if egfr_val: st.info(f"🩺 CKD Stage: {get_ckd_stage(egfr_val)}")

            age = input_data.get("Age of the patient")
            acr = input_data.get("Urine protein-to-creatinine ratio")
            risk = kfre_4_variable(age, "male", egfr_val, acr)
            if risk is not None:
                st.metric("📊 Kidney Failure Risk (2-year)", f"{risk}%")

            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            st.subheader("🔍 SHAP Feature Contribution")
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# === Utility functions ===
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def sanitize_df(df: pd.DataFrame):
    # binary & expected numeric columns
    binary_cols = ["sex", "address", "famsize", "Pstatus",
                   "schoolsup", "famsup", "paid", "activities", "higher",
                   "internet", "romantic", "nursery"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric_expect = ["age", "traveltime", "studytime", "failures",
                      "famrel", "freetime", "goout", "Dalc", "Walc",
                      "health", "absences", "G1", "G2"]
    for col in numeric_expect:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.isnull().any().any():
        st.warning("Some inputs were coerced to NaN due to type issues; please verify your entries.")
    return df

def prepare_example_df():
    example = {
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "health",
        "Fjob": "services",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 4,
        "nursery": "yes",
        "G1": 14,
        "G2": 15,
        "school": "GP"
    }
    df = pd.DataFrame([example])
    bin_map = {"yes":1,"no":0,"F":0,"M":1,"U":1,"R":0,"LE3":0,"GT3":1,"T":1,"A":0}
    df["sex"] = df["sex"].map(bin_map)
    df["address"] = df["address"].map(bin_map)
    df["famsize"] = df["famsize"].map(bin_map)
    df["Pstatus"] = df["Pstatus"].map(bin_map)
    for col in ["schoolsup","famsup","paid","activities","higher","internet","romantic","nursery"]:
        df[col] = df[col].map({"yes":1,"no":0})
    return sanitize_df(df)

# === Streamlit UI ===
st.set_page_config(page_title="Student Grade Predictor", layout="centered")
st.title("Final Grade (G3) Predictor")

# Load model
model_path = "D:\\propertyverify_final\\visual\\best_random_forest_pipeline.joblib"
if not Path(model_path).exists():
    st.error(f"Model not found at '{model_path}'. Make sure you ran training and saved the pipeline.")
    st.stop()

model = load_model(model_path)

st.sidebar.header("Input Mode")
input_mode = st.sidebar.selectbox("Mode", ["Example student", "Manual input"])

if input_mode == "Example student":
    st.subheader("Example student")
    df_input = prepare_example_df()
    st.markdown("**Input features (processed):**")
    st.dataframe(df_input.T, use_container_width=True)

    if st.button("Predict for example"):
        pred = model.predict(df_input)[0]
        st.success(f"Predicted final grade (G3): **{pred:.2f}**")

elif input_mode == "Manual input":
    st.subheader("Fill student features manually")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.selectbox("Sex", options=["F", "M"], index=0)
        age = st.number_input("Age", min_value=15, max_value=22, value=17)
        address = st.selectbox("Address", options=["U", "R"], index=0)
        famsize = st.selectbox("Famsize", options=["LE3", "GT3"], index=1)
        Pstatus = st.selectbox("Pstatus", options=["T", "A"], index=0)
        Medu = st.selectbox("Mother's education (Medu)", options=[0,1,2,3,4], index=2)
        Fedu = st.selectbox("Father's education (Fedu)", options=[0,1,2,3,4], index=2)
        Mjob = st.selectbox("Mother's job", options=["at_home","health","other","services","teacher"], index=1)
        Fjob = st.selectbox("Father's job", options=["at_home","health","other","services","teacher"], index=1)
        reason = st.selectbox("Reason to choose school", options=["home","reputation","course","other"], index=2)
        guardian = st.selectbox("Guardian", options=["mother","father","other"], index=0)
    with col2:
        traveltime = st.selectbox("Travel time", options=[1,2,3,4], index=0)
        studytime = st.selectbox("Study time", options=[1,2,3,4], index=1)
        failures = st.number_input("Past failures", min_value=0, max_value=4, value=0)
        schoolsup = st.selectbox("School support", options=["yes","no"], index=1)
        famsup = st.selectbox("Family support", options=["yes","no"], index=0)
        paid = st.selectbox("Extra paid classes", options=["yes","no"], index=1)
        activities = st.selectbox("Extra-curricular activities", options=["yes","no"], index=0)
        higher = st.selectbox("Wants higher education", options=["yes","no"], index=0)
        internet = st.selectbox("Internet access", options=["yes","no"], index=0)
        romantic = st.selectbox("In a romantic relationship", options=["yes","no"], index=1)
        famrel = st.slider("Family relationship quality", 1, 5, 4)
        freetime = st.slider("Free time", 1, 5, 3)
        goout = st.slider("Going out", 1, 5, 2)
        Dalc = st.selectbox("Workday alcohol consumption", options=[1,2,3,4,5], index=0)
        Walc = st.selectbox("Weekend alcohol consumption", options=[1,2,3,4,5], index=0)
        health = st.selectbox("Current health", options=[1,2,3,4,5], index=4)
        absences = st.number_input("Absences", min_value=0, max_value=100, value=4)
        nursery = st.selectbox("Attended nursery", options=["yes","no"], index=0)
        G1 = st.number_input("G1 (first period grade)", min_value=0, max_value=20, value=14)
        G2 = st.number_input("G2 (second period grade)", min_value=0, max_value=20, value=15)
        school = st.selectbox("School", options=["GP","MS"], index=0)

    raw = {
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "nursery": nursery,
        "G1": G1,
        "G2": G2,
        "school": school
    }
    df_input = pd.DataFrame([raw])
    bin_map = {"yes":1,"no":0,"F":0,"M":1,"U":1,"R":0,"LE3":0,"GT3":1,"T":1,"A":0}
    df_input["sex"] = df_input["sex"].map(bin_map)
    df_input["address"] = df_input["address"].map(bin_map)
    df_input["famsize"] = df_input["famsize"].map(bin_map)
    df_input["Pstatus"] = df_input["Pstatus"].map(bin_map)
    for col in ["schoolsup","famsup","paid","activities","higher","internet","romantic","nursery"]:
        df_input[col] = df_input[col].map({"yes":1,"no":0})
    df_input = sanitize_df(df_input)

    st.markdown("**Processed input used for prediction:**")
    st.dataframe(df_input.T, use_container_width=True)

    if st.button("Predict"):
        pred = model.predict(df_input)[0]
        st.success(f"Predicted final grade (G3): **{pred:.2f}**")

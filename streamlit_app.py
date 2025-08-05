import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def sanitize_df(df: pd.DataFrame):
    """Ensure numeric and binary columns are converted properly"""
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
        st.warning("Some inputs were coerced to NaN due to type issues.")
    return df

def prepare_example_df():
    """Prepare example student input"""
    example = {
        "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 2, "Mjob": "health", "Fjob": "services", "reason": "course",
        "guardian": "mother", "traveltime": 1, "studytime": 2, "failures": 0,
        "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes",
        "higher": "yes", "internet": "yes", "romantic": "no", "famrel": 4,
        "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1, "health": 5,
        "absences": 4, "nursery": "yes", "G1": 14, "G2": 15, "school": "GP"
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

def train_default_model():
    """Train a fallback dummy RandomForest model"""
    st.warning("⚠️ Model file missing or incompatible. Training a default model...")
    X = pd.DataFrame(np.random.rand(50, 5), columns=list("ABCDE"))
    y = np.random.randint(0, 20, 50)
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    pipe.fit(X, y)
    joblib.dump(pipe, "best_random_forest_pipeline.joblib")
    return pipe

@st.cache_resource
def load_model_safe(path: str):
    """Load model with error handling and fallback"""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load saved model: {e}")
        return train_default_model()

st.set_page_config(page_title="Student Grade Predictor", layout="centered")
st.title("Final Grade (G3) Predictor")
#loading model
model_path = os.path.join(os.path.dirname(__file__), "best_random_forest.joblib")
model = load_model_safe(model_path)

st.sidebar.header("Input Mode")
input_mode = st.sidebar.selectbox("Mode", ["Example student", "Manual input"])
if input_mode == "Example student":
    st.subheader("Example student")
    df_input = prepare_example_df()
    st.markdown("**Processed example input:**")
    st.dataframe(df_input.T, use_container_width=True)

    if st.button("Predict for Example"):
        pred = model.predict(df_input)[0]
        st.success(f"✅ Predicted Final Grade (G3): **{pred:.2f}**")
else:
    st.subheader("Fill student features manually")
    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox("Sex", ["F", "M"], index=0)
        age = st.number_input("Age", 15, 22, 17)
        address = st.selectbox("Address", ["U", "R"], 0)
        famsize = st.selectbox("Famsize", ["LE3", "GT3"], 1)
        Pstatus = st.selectbox("Pstatus", ["T", "A"], 0)
        Medu = st.selectbox("Mother's education (0-4)", [0,1,2,3,4], 2)
        Fedu = st.selectbox("Father's education (0-4)", [0,1,2,3,4], 2)
        Mjob = st.selectbox("Mother's job", ["at_home","health","other","services","teacher"], 1)
        Fjob = st.selectbox("Father's job", ["at_home","health","other","services","teacher"], 1)
        reason = st.selectbox("Reason to choose school", ["home","reputation","course","other"], 2)
        guardian = st.selectbox("Guardian", ["mother","father","other"], 0)

    with col2:
        traveltime = st.selectbox("Travel time", [1,2,3,4], 0)
        studytime = st.selectbox("Study time", [1,2,3,4], 1)
        failures = st.number_input("Past failures", 0, 4, 0)
        schoolsup = st.selectbox("School support", ["yes","no"], 1)
        famsup = st.selectbox("Family support", ["yes","no"], 0)
        paid = st.selectbox("Extra paid classes", ["yes","no"], 1)
        activities = st.selectbox("Extra-curricular activities", ["yes","no"], 0)
        higher = st.selectbox("Wants higher education", ["yes","no"], 0)
        internet = st.selectbox("Internet access", ["yes","no"], 0)
        romantic = st.selectbox("Romantic relationship", ["yes","no"], 1)
        famrel = st.slider("Family relationship quality", 1, 5, 4)
        freetime = st.slider("Free time", 1, 5, 3)
        goout = st.slider("Going out", 1, 5, 2)
        Dalc = st.selectbox("Workday alcohol", [1,2,3,4,5], 0)
        Walc = st.selectbox("Weekend alcohol", [1,2,3,4,5], 0)
        health = st.selectbox("Current health", [1,2,3,4,5], 4)
        absences = st.number_input("Absences", 0, 100, 4)
        nursery = st.selectbox("Attended nursery", ["yes","no"], 0)
        G1 = st.number_input("G1", 0, 20, 14)
        G2 = st.number_input("G2", 0, 20, 15)
        school = st.selectbox("School", ["GP","MS"], 0)
        
    raw = {
        "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob,
        "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid, "activities": activities,
        "higher": higher, "internet": internet, "romantic": romantic,
        "famrel": famrel, "freetime": freetime, "goout": goout, "Dalc": Dalc,
        "Walc": Walc, "health": health, "absences": absences, "nursery": nursery,
        "G1": G1, "G2": G2, "school": school
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

    st.markdown("**Processed Input:**")
    st.dataframe(df_input.T, use_container_width=True)

    if st.button("Predict"):
        pred = model.predict(df_input)[0]
        st.success(f"✅ Predicted Final Grade (G3): **{pred:.2f}**")






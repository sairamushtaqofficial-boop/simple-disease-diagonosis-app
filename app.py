import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Page config
st.set_page_config(page_title="AI Disease Predictor", page_icon="🧠", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🧠 AI Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select your symptoms and get prediction</p>", unsafe_allow_html=True)

st.write("---")

# Dataset
data = {
    'fever': [1,1,0,1,0,0,1,0],
    'cough': [1,1,0,0,1,0,1,0],
    'fatigue': [1,0,1,1,0,1,1,0],
    'headache': [1,1,0,1,0,1,1,0],
    'disease': ['Flu','Flu','Cold','Flu','Cold','Cold','Flu','Healthy']
}

df = pd.DataFrame(data)

X = df[['fever','cough','fatigue','headache']]
y = df['disease']

model = DecisionTreeClassifier()
model.fit(X, y)

# Card-style box
st.subheader("📝 Select Symptoms")

col1, col2 = st.columns(2)

with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")

with col2:
    fatigue = st.checkbox("Fatigue")
    headache = st.checkbox("Headache")

st.write("")

# Button
if st.button("🔍 Predict Disease"):
    input_data = [[int(fever), int(cough), int(fatigue), int(headache)]]
    prediction = model.predict(input_data)

    st.success(f"🩺 Predicted Disease: {prediction[0]}")
    st.info("📌 This is a basic AI prediction for learning purposes.")
    st.warning("⚠️ Not a medical diagnosis")

# Footer
st.write("---")
st.caption("Built by Saira Mushtaq | AI + Microbiology Project")
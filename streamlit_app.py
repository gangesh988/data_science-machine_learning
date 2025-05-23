import streamlit as st
import pickle
import numpy as np

# Load the trained logistic regression model
# Load trained model with full path
with open("C:/Users/victus/Data_Science/Code/Logistic.pkl", "rb") as f:
    model = pickle.load(f)


# Streamlit app title
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict the probability of survival on the Titanic.")

# User input section
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare Paid (in $)", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs to match the model's training
sex_encoded = 0 if sex == "female" else 1
embarked_mapping = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_mapping[embarked]

# Create the input array
input_features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    if prediction == 1:
        st.success(f"🛟 The passenger would SURVIVE! (Probability: {probability:.2%})")
    else:
        st.error(f"⚓ The passenger would NOT survive. (Probability: {probability:.2%})")


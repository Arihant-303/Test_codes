import streamlit as st
import numpy as np
import joblib


# Load the trained model
model = joblib.load("model_titanic.pkl")

st.title("Titanic Survival Predictor")
st.write("Predict whether a passenger survived the Titanic disaster based on their characteristics.")

# Inputs
age = st.number_input("Age", 0.0, 100.0, 25.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

sex = st.selectbox("Sex", ["male", "female"])
pclass = st.selectbox("Pclass", [1, 2, 3])

# Convert categorical inputs to dummy variables
sex_female = 1 if sex == "female" else 0
sex_male = 1 if sex == "male" else 0

pclass_1 = 1 if pclass == 1 else 0
pclass_2 = 1 if pclass == 2 else 0
pclass_3 = 1 if pclass == 3 else 0

# Combine all inputs into a single array
input_data = np.array([[age, sibsp, parch, fare,
                        sex_female, sex_male,
                        pclass_1, pclass_2, pclass_3]])

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    st.write("Survived" if prediction[0] == 1 else "Did not survive")
    st.balloons() if prediction[0] == 1 else st.error("Better luck next time!")
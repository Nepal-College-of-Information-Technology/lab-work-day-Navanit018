import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data (you can replace this with a more comprehensive dataset)
data = {
    'symptom': [
        "fever cough",
        "fever headache",
        "headache fatigue",
        "cough fatigue",
        "nausea vomiting",
        "fatigue fever"
    ],
    'condition': [
        "flu",
        "migraine",
        "stress",
        "cold",
        "stomach flu",
        "viral infection"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['symptom'])
y = df['condition']

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

def predict_condition(symptoms):
    symptoms_vec = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vec)
    return prediction[0]

# Streamlit UI
st.title("Advanced Telehealth Symptom Checker")
st.write("Enter symptoms separated by commas:")

# Input for symptoms
symptom_input = st.text_input("Symptoms:")

if st.button("Check Symptoms"):
    # Split the input into a list
    symptoms = ' '.join([symptom.strip() for symptom in symptom_input.split(",")])
    
    predicted_condition = predict_condition(symptoms)
    
    # Display predictions
    st.write("### Predicted Condition:")
    st.write(f"- **Condition**: {predicted_condition}")
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
try:
    with open('spam_detection_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check the file paths.")
    st.stop()

# Set up the Streamlit app
st.title("Spam Detection Email Classifier")
st.write("Enter your email text below:")

# Text input for the email
input_mail = st.text_area("Email Text")

# Classify the email when the button is pressed
if st.button("Classify"):
    if input_mail:
        # Transform the input using the loaded vectorizer
        input_data_features = vectorizer.transform([input_mail])
        
        # Make prediction
        prediction = model.predict(input_data_features)
        
        # Display result
        if prediction[0] == 1:
            st.error("This email is classified as **Spam**.")
        else:
            st.success("This email is classified as **Not Spam**.")
    else:
        st.warning("Please enter some text for classification.")

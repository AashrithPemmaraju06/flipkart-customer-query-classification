import streamlit as st
import joblib
import pickle

# Load model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# UI
st.title("Customer Query Classifier")
st.write("Enter a customer query and classify it into a category.")

query = st.text_area("Customer Query")

if st.button("Classify"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        vec = vectorizer.transform([query])
        pred_num = model.predict(vec)[0]  # numeric label
        pred_label = label_encoder.inverse_transform([pred_num])[0]  # convert to category name
        st.success(f"Predicted Category: **{pred_label}**")

import streamlit as st
import joblib

model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Customer Query Classifier")
st.write("Enter a customer query and classify it into a category.")

query = st.text_area("Customer Query")

if st.button("Classify"):
    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        vec = vectorizer.transform([query])
        pred = model.predict(vec)[0]
        st.success(f"Predicted Category: **{pred}**")

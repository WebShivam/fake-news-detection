# app/app.py


def load_accuracies():
    try:
        with open("model_files/nb_accuracy.txt", "r") as f:
            nb_acc = f.read()
        with open("model_files/lr_accuracy.txt", "r") as f:
            lr_acc = f.read()
        with open("model_files/svm_accuracy.txt", "r") as f:
            svm_acc = f.read()
        return nb_acc, lr_acc, svm_acc
    except:
        return "?", "?", "?"


import streamlit as st
import pickle

# Load models and vectorizer
@st.cache_resource
def load_models():
    with open("./model_files/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("./model_files/nb_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open("./model_files/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("./model_files/svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    return vectorizer, nb_model, lr_model, svm_model

vectorizer, nb_model, lr_model, svm_model = load_models()

# Load model accuracy from saved .txt files
# This function reads the accuracy from .txt files
def load_accuracies():
    try:
        with open("./model_files/nb_accuracy.txt", "r") as f:
            nb_acc = f.read()
        with open("./model_files/lr_accuracy.txt", "r") as f:
            lr_acc = f.read()
        with open("./model_files/svm_accuracy.txt", "r") as f:
            svm_acc = f.read()
        return nb_acc, lr_acc, svm_acc
    except:
        return "?", "?", "?"


# Title and description
st.title("📰 Fake News Detector")
st.markdown("Paste any news content or upload a `.txt` file to see predictions from multiple models.")

# Input options
input_option = st.radio("Choose input method:", ["Paste text", "Upload .txt file"])

if input_option == "Paste text":
    user_input = st.text_area("Paste the news article here:")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    user_input = uploaded_file.read().decode("utf-8") if uploaded_file else ""

st.markdown("## 📊 Model Accuracy Comparison")

nb_acc, lr_acc, svm_acc = load_accuracies()

st.write(f"**Naive Bayes Accuracy**: {nb_acc}%")
st.write(f"**Logistic Regression Accuracy**: {lr_acc}%")
st.write(f"**SVM Accuracy**: {svm_acc}%")



# Predict button
check_button = st.button("Check if it's Fake or Real", key="check_button")

if check_button:
    if user_input:
        # Transform input
        X_input = vectorizer.transform([user_input])

        # Predictions
        nb_pred = nb_model.predict(X_input)[0]
        lr_pred = lr_model.predict(X_input)[0]
        svm_pred = svm_model.predict(X_input)[0]

        label_map = {0: "🟢 Real", 1: "🔴 Fake"}

        # Results
        st.markdown("### 🔍 Predictions")
        st.write(f"**Naive Bayes**: {label_map[nb_pred]}")
        st.write(f"**Logistic Regression**: {label_map[lr_pred]}")
        st.write(f"**SVM (LinearSVC)**: {label_map[svm_pred]}")
    else:
        st.warning("Please enter or upload some text.")

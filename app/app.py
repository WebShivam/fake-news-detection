# app/app.py

import streamlit as st
import joblib

# ----------------- Load models and vectorizer -----------------
@st.cache_resource
def load_models():
    with open("model_files/vectorizer.pkl", "rb") as f:
        vectorizer = joblib.load(f)
    with open("model_files/nb_model.pkl", "rb") as f:
        nb_model = joblib.load(f)
    with open("model_files/lr_model.pkl", "rb") as f:
        lr_model = joblib.load(f)
    with open("model_files/svm_model.pkl", "rb") as f:
        svm_model = joblib.load(f)
    return vectorizer, nb_model, lr_model, svm_model

vectorizer, nb_model, lr_model, svm_model = load_models()

# ----------------- Load accuracy from .txt files -----------------
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

# ----------------- UI -----------------
st.title("📰 Fake News Detector")
st.markdown("Paste any news content or upload a `.txt` file to see predictions from multiple models.")

# Input method
input_option = st.radio("Choose input method:", ["Paste text", "Upload .txt file"])

if input_option == "Paste text":
    user_input = st.text_area("Paste the news article here:")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    user_input = uploaded_file.read().decode("utf-8") if uploaded_file else ""

# Show model accuracies
st.markdown("## 📊 Model Accuracy Comparison")
nb_acc, lr_acc, svm_acc = load_accuracies()
st.write(f"**Naive Bayes Accuracy**: {nb_acc}%")
st.write(f"**Logistic Regression Accuracy**: {lr_acc}%")
st.write(f"**SVM Accuracy**: {svm_acc}%")

# Analyze button — shows prediction + confidence bars
if st.button("Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        vec = vectorizer.transform([user_input])

        models = {
            "Naive Bayes": nb_model,
            "Logistic Regression": lr_model,
            "SVM (LinearSVC)": svm_model
        }

        label_map = {0: "🟢 Real", 1: "🔴 Fake"}

        for name, model in models.items():
            pred = model.predict(vec)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vec)[0]
                confidence = max(prob)
            else:
                decision_score = model.decision_function(vec)[0]
                confidence = 1 / (1 + abs(decision_score))  # mapped to 0–1 range

            st.subheader(f"🔍 {name}")
            st.write(f"**Prediction:** {label_map[pred]}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")
            st.progress(int(confidence * 100))

# Optional: Compact label summary
if st.button("Check if it's Fake or Real", key="check_button"):
    if user_input.strip():
        vec = vectorizer.transform([user_input])

        nb_pred = nb_model.predict(vec)[0]
        lr_pred = lr_model.predict(vec)[0]
        svm_pred = svm_model.predict(vec)[0]

        label_map = {0: "🟢 Real", 1: "🔴 Fake"}

        st.markdown("### 🧾 Summary Predictions")
        st.write(f"**Naive Bayes:** {label_map[nb_pred]}")
        st.write(f"**Logistic Regression:** {label_map[lr_pred]}")
        st.write(f"**SVM (LinearSVC):** {label_map[svm_pred]}")
    else:
        st.warning("⚠️ Please enter or upload some news content.")

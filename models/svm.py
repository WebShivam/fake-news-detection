

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load the data
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels
true_df["label"] = 1  # Real news
fake_df["label"] = 0  # Fake news

# Combine the datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features and labels
X = df["text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)

# Predict & accuracy
svm_predictions = svm_model.predict(X_test_vec)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Save model and accuracy
os.makedirs("../model_files", exist_ok=True)

joblib.dump(svm_model, "model_files/svm_model.pkl")
with open("model_files/svm_accuracy.txt", "w") as f:
    f.write(str(round(svm_accuracy * 100, 2)))

print(f"SVM Model trained. Accuracy: {svm_accuracy * 100:.2f}%")

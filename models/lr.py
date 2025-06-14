import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Create directory if not exists
os.makedirs("../model_files", exist_ok=True)

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 0
fake_df["label"] = 1

df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict & accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

# Save model and accuracy
os.makedirs("model_files", exist_ok=True)

joblib.dump(model, "../model_files/lr_model.pkl")
with open("../model_files/lr_accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))

print(f"Logistic Regression Model trained. Accuracy: {accuracy * 100:.2f}%")




'''with open("model_files/lr_accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))'''


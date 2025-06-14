import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Make sure the directory exists
os.makedirs("model_files", exist_ok=True)

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Labeling: 0 = True, 1 = Fake
true_df["label"] = 0
fake_df["label"] = 1

# Combine & shuffle
df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict & accuracy
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)


# Save everything: model & vectorizer
os.makedirs("models", exist_ok=True)

joblib.dump(model, "model_files/nb_model.pkl")
joblib.dump(vectorizer, "model_files/vectorizer.pkl")

# Save accuracy
with open("model_files/nb_accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))

print(f"✅ Naive Bayes Model trained and saved successfully with accuracy: {accuracy * 100:.2f}%")

'''#Save Section
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/nb_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
with open("models/nb_accuracy.txt", "w") as f:
    f.write(str(round(accuracy * 100, 2)))

print(f"Naive Bayes Model trained. Accuracy: {accuracy * 100:.2f}%")'''
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_combine_data(true_path, fake_path):
    """
    Load True.csv and Fake.csv, label them, and combine.
    """
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    true_df['label'] = 0  # Real = 0
    fake_df['label'] = 1  # Fake = 1

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

    return df

def clean_text(text):
    """
    Clean the input text by removing links, punctuation, and making it lowercase.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def preprocess_data(df):
    """
    Apply cleaning to title and text columns, and combine them into a single feature column.
    """
    df['text'] = df['title'] + " " + df['text']
    df['text'] = df['text'].apply(clean_text)
    return df

def vectorize_and_split(df):
    """
    Convert text to TF-IDF features and split into train and test sets.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, vectorizer

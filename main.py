from utils.preprocess import load_and_combine_data, preprocess_data, vectorize_and_split

# Step 1: Load data
df = load_and_combine_data("data/True.csv", "data/Fake.csv")

# Step 2: Clean and prepare
df = preprocess_data(df)

# Step 3: TF-IDF + Train-Test split
X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)

print("✅ Data ready for model training")

'''

from utils.preprocess import load_and_combine_data, preprocess_data, vectorize_and_split
from models.naive_bayes import train_and_save_naive_bayes

# Load and prepare data
df = load_and_combine_data("data/True.csv", "data/Fake.csv")
df = preprocess_data(df)
X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)

# Train and save NB model
train_and_save_naive_bayes(X_train, y_train, X_test, y_test, vectorizer)



from utils.preprocess import load_and_combine_data, preprocess_data, vectorize_and_split
from models.logistic_regression import train_and_save_logistic_regression

# Load and prepare data
df = load_and_combine_data("data/True.csv", "data/Fake.csv")
df = preprocess_data(df)
X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)

# Train and save Logistic Regression model
train_and_save_logistic_regression(X_train, y_train, X_test, y_test, vectorizer)
'''

from utils.preprocess import load_and_combine_data, preprocess_data, vectorize_and_split
from models.svm import train_and_save_svm

# Load and prepare data
df = load_and_combine_data("data/True.csv", "data/Fake.csv")
df = preprocess_data(df)
X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)

# Train and save SVM model
train_and_save_svm(X_train, y_train, X_test, y_test, vectorizer)

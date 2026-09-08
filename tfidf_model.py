import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load processed data
df = pd.read_csv("dataset/processed_data.csv")

# Combine resume and job description
df["combined_text"] = df["resume_text"] + " " + df["job_text"]

# Create TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["combined_text"])

y = df["match"]

print("TF-IDF created successfully!")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target shape:", y.shape)
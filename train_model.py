import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Load processed dataset
df = pd.read_csv("dataset/processed_dataset.csv")

# TF-IDF for resume and job description
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

resume_tfidf = vectorizer.fit_transform(df["resume_text"])
job_tfidf = vectorizer.transform(df["job_text"])

# Calculate text similarity
similarity = cosine_similarity(resume_tfidf, job_tfidf)

text_similarity = similarity.diagonal() * 100

# Create meaningful numerical features
X = pd.DataFrame({
    "text_similarity": text_similarity,
    "skill_match_percentage": df["skill_match_percentage"],
    "matched_skill_count": df["matched_skills"].fillna("").apply(
        lambda x: len(x.split(",")) if x else 0
    ),
    "missing_skill_count": df["missing_skills"].fillna("").apply(
        lambda x: len(x.split(",")) if x else 0
    )
})

y = df["match"]

print("Features created successfully!")
print("\nFeatures:")
print(X.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Completed!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Poor Match", "Good Match"]
    )
)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
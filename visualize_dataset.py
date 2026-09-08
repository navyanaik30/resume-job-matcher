import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("dataset/job_resume_fit.csv")

# Create graphs folder
os.makedirs("graphs", exist_ok=True)


# -----------------------------------------
# 1. Job Category Distribution
# -----------------------------------------

category_counts = df["category"].value_counts()

plt.figure(figsize=(10, 7))
category_counts.sort_values().plot(kind="barh")

plt.title("Job Category Distribution")
plt.xlabel("Number of Records")
plt.ylabel("Job Category")
plt.tight_layout()

plt.savefig("graphs/01_job_category_distribution.png", dpi=300)
plt.show()


# -----------------------------------------
# 2. Good Match vs Poor Match
# -----------------------------------------

df["match"] = (df["ai_match_score"] >= 40).astype(int)

match_counts = df["match"].map({
    0: "Poor Match",
    1: "Good Match"
}).value_counts()

plt.figure(figsize=(7, 5))
match_counts.plot(kind="bar")

plt.title("Good Match vs Poor Match")
plt.xlabel("Match Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("graphs/02_match_distribution.png", dpi=300)
plt.show()


# -----------------------------------------
# 3. AI Match Score Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["ai_match_score"],
    bins=20
)

plt.title("AI Match Score Distribution")
plt.xlabel("AI Match Score")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("graphs/03_ai_match_score_distribution.png", dpi=300)
plt.show()


# -----------------------------------------
# 4. Skill Match Score Distribution
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["skill_string_match_score"],
    bins=20
)

plt.title("Skill Match Score Distribution")
plt.xlabel("Skill Match Score")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("graphs/04_skill_match_distribution.png", dpi=300)
plt.show()


# -----------------------------------------
# 5. AI Match Score vs Fuzzy Match Score
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["ai_match_score"],
    df["fuzzy_match_score"],
    alpha=0.5
)

plt.title("AI Match Score vs Fuzzy Match Score")
plt.xlabel("AI Match Score")
plt.ylabel("Fuzzy Match Score")
plt.tight_layout()

plt.savefig("graphs/05_ai_vs_fuzzy_score.png", dpi=300)
plt.show()


# -----------------------------------------
# 6. Average Match Score by Job Category
# -----------------------------------------

category_score = (
    df.groupby("category")["ai_match_score"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(10, 7))

category_score.plot(kind="barh")

plt.title("Average AI Match Score by Job Category")
plt.xlabel("Average AI Match Score")
plt.ylabel("Job Category")
plt.tight_layout()

plt.savefig("graphs/06_match_score_by_category.png", dpi=300)
plt.show()


print("\nAll graphs created successfully!")

print("\nGraphs saved inside:")
print("graphs/")
import pandas as pd
from skill_gap import get_skill_gap

# Load original dataset
df = pd.read_csv("dataset/job_resume_fit.csv")

# Create target
df["match"] = (df["ai_match_score"] >= 40).astype(int)

# Lists to store skill information
matched_list = []
missing_list = []
skill_percent = []

# Calculate skill gap for every row
for i in range(len(df)):
    matched, missing, percent = get_skill_gap(
        df.loc[i, "job_text"],
        df.loc[i, "resume_text"]
    )

    matched_list.append(",".join(matched))
    missing_list.append(",".join(missing))
    skill_percent.append(percent)

# Add new columns
df["matched_skills"] = matched_list
df["missing_skills"] = missing_list
df["skill_match_percentage"] = skill_percent

# Save processed dataset
df.to_csv("dataset/processed_dataset.csv", index=False)

print("Processed dataset created successfully!")

print("\nShape:")
print(df.shape)

print("\nClass Distribution:")
print(df["match"].value_counts())

print("\nNew Columns:")
print([
    "matched_skills",
    "missing_skills",
    "skill_match_percentage",
    "match"
])
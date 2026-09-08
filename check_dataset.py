import pandas as pd

df = pd.read_csv("dataset/job_resume_fit.csv")

for threshold in [40, 45, 50, 55, 60]:
    good = (df["ai_match_score"] >= threshold).sum()
    poor = (df["ai_match_score"] < threshold).sum()

    print(f"\nThreshold: {threshold}")
    print("Good Match:", good)
    print("Poor Match:", poor)
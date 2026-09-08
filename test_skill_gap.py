from skill_gap import get_skill_gap

job_skills = "['Python', 'SQL', 'Pandas', 'Flask', 'Docker']"

resume_skills = "['Python', 'SQL', 'Pandas']"

matched, missing, percentage = get_skill_gap(
    job_skills,
    resume_skills
)

print("Matched Skills:")
print(matched)

print("\nMissing Skills:")
print(missing)

print("\nSkill Match:")
print(percentage, "%")
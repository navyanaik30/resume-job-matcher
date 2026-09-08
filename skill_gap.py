import re

SKILLS = [
    "python", "java", "c", "c++", "javascript", "typescript",
    "html", "css", "sql", "mysql", "mongodb",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "machine learning", "deep learning", "data science",
    "data analysis", "nlp", "computer vision",
    "flask", "django", "spring boot",
    "git", "github", "docker", "aws", "azure",
    "power bi", "tableau", "excel",
    "rest api", "api", "communication",
    "problem solving", "data structures", "algorithms",
    "oop", "linux", "cloud computing"
]


def extract_skills(text):

    text = text.lower()

    found_skills = []

    for skill in SKILLS:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):
            found_skills.append(skill)

    return found_skills


def get_skill_gap(job_text, resume_text):

    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)

    matched = [
        skill for skill in job_skills
        if skill in resume_skills
    ]

    missing = [
        skill for skill in job_skills
        if skill not in resume_skills
    ]

    if len(job_skills) > 0:
        skill_match_percentage = (
            len(matched) / len(job_skills)
        ) * 100
    else:
        skill_match_percentage = 0

    return (
        matched,
        missing,
        round(skill_match_percentage, 2)
    )

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import tempfile

from sklearn.metrics.pairwise import cosine_similarity

from pdf_reader import extract_text_from_pdf
from skill_gap import get_skill_gap

app = Flask(__name__)
CORS(app)

# Load trained model and TF-IDF vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return "Resume Job Matcher is Running!"


@app.route("/analyze", methods=["POST"])
def analyze():

    if "resume" not in request.files:
        return jsonify({"error": "Resume PDF is required"}), 400

    resume = request.files["resume"]
    job_description = request.form.get("job_description", "")

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        resume.save(temp.name)
        pdf_path = temp.name

    # Extract resume text
    resume_text = extract_text_from_pdf(pdf_path)

    # -----------------------------
    # 1. Text similarity
    # -----------------------------
    resume_vector = vectorizer.transform([resume_text])
    job_vector = vectorizer.transform([job_description])

    similarity = cosine_similarity(
        resume_vector,
        job_vector
    )[0][0]

    text_similarity = similarity * 100

    # -----------------------------
    # 2. Skill gap
    # -----------------------------
    matched_skills, missing_skills, skill_match_percentage = get_skill_gap(
        job_description,
        resume_text
    )

    matched_skill_count = len(matched_skills)
    missing_skill_count = len(missing_skills)

    # -----------------------------
    # 3. Create model features
    # -----------------------------
    features = [[
        text_similarity,
        skill_match_percentage,
        matched_skill_count,
        missing_skill_count
    ]]

    # -----------------------------
    # 4. Random Forest prediction
    # -----------------------------
    prediction = model.predict(features)[0]

    if prediction == 1:
        match_result = "Good Match"
    else:
        match_result = "Poor Match"

    probability = float(model.predict_proba(features)[0][1])

    # Ensure probability stays between 0 and 1
    probability = max(0.0, min(probability, 1.0))

    ml_score = probability * 100

    # -----------------------------
    # 5. Overall match percentage
    # -----------------------------
    match_percentage = round(
        (ml_score * 0.50)
        + (skill_match_percentage * 0.30)
        + (text_similarity * 0.20),
        2
    )

    # Keep overall score within 0–100
    match_percentage = max(0, min(match_percentage, 100))

    # -----------------------------
    # 6. Recommendations
    # -----------------------------
    recommendations = missing_skills

    # -----------------------------
    # 7. Send result to frontend
    # -----------------------------
    return jsonify({
        "match_result": match_result,
        "match_percentage": match_percentage,
        "skill_match_percentage": skill_match_percentage,
        "text_similarity": round(text_similarity, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
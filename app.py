# app.py

import os
import pickle
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from ats_model import calculate_ats_score, pdf_to_text

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


# 1) Point this to the pickle you created above:
MODEL_PATH = os.environ.get("ATS_KEYWORDS_PATH", "learned_keywords.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Cannot find learned‑keywords at '{MODEL_PATH}'.\n"
        "Run: python train_model.py --csv your_dataset.csv"
    )

with open(MODEL_PATH, "rb") as f:
    learned_keywords = pickle.load(f)


@app.route("/score", methods=["POST"])
def score_resume():
    """
    Expects multipart/form-data:
      - resume: the PDF file
      - job_title (optional): string
    Returns JSON:
      { "score": int, "suggestions": [ str, … ] }
    """
    if "resume" not in request.files:
        return jsonify(error="Missing form‑field 'resume'"), 400

    pdf_file = request.files["resume"]
    job_title = request.form.get("job_title")

    # write to temp so pdfminer can read
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        pdf_file.save(tmp_path)

    try:
        text = pdf_to_text(tmp_path)
        if not text:
            return jsonify(error="Failed to extract text from PDF"), 500

        score, suggestions = calculate_ats_score(text, learned_keywords, job_title)
        return jsonify(score=score, suggestions=suggestions)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    # In production, bind HOST/PORT via env vars or use a WSGI server
    app.run(host="0.0.0.0", port=5000, debug=False)

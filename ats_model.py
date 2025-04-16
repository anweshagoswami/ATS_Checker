# ats_checker.py

import re
import string
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

GENERIC_WORDS = {
    'including', 'implementing', 'using', 'learning', 'projects', 'developer',
    'building', 'applications', 'technologies', 'and', 'or', 'to', 'for', 'with'
}

def clean_text(text):
    """
    Lowercases the text, removes digits and punctuation.
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def learn_job_keywords(df, top_n=10):
    """
    return a dictionary mapping each job title (lowercase) to its top 'n' keywords
    """
    learned_keywords = {}
    job_groups = df.groupby('JOB TITLE')
    
    for job, group in job_groups:
        job_lower = job.lower().strip()
        
        combined_text = " ".join(group['RESUME TEXT'].astype(str).tolist())
        combined_text = clean_text(combined_text)
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=300)
        tfidf_matrix = tfidf.fit_transform([combined_text])
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        term_scores = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
       
        filtered_terms = [term for term, score in term_scores if term not in GENERIC_WORDS]
        
        if len(filtered_terms) < top_n:
            top_keywords = []
            seen = set()
            for term, score in term_scores:
                if term not in seen:
                    top_keywords.append(term)
                    seen.add(term)
                if len(top_keywords) == top_n:
                    break
        else:
            top_keywords = filtered_terms[:top_n]
            
        learned_keywords[job_lower] = top_keywords
    
    return learned_keywords

def match_job_title(user_title, learned_keywords):
    """
    finding the closest matching job title from the learned keywords.
    Returns the best match if cosine similarity is above 0.6, otherwise None.
    """
    if not user_title:
        return None

    user_embedding = embedding_model.encode(user_title.lower(), convert_to_tensor=True)
    best_match = None
    highest_score = 0.0

    for job in learned_keywords.keys():
        job_embedding = embedding_model.encode(job.lower(), convert_to_tensor=True)
        score = util.pytorch_cos_sim(user_embedding, job_embedding).item()
        if score > highest_score:
            highest_score = score
            best_match = job

    return best_match if highest_score > 0.6 else None

def calculate_keyword_score(resume_text, job_title, learned_keywords, weight=30):
    """
    Calculate the keyword matching score.
    Suggests role-specific keywords by filtering out generic words.
    """
    suggestions = []
    resume_clean = clean_text(resume_text)

    matched_title = match_job_title(job_title, learned_keywords)
    if matched_title:
        keywords = learned_keywords[matched_title]
    else:
        keywords = list({kw for kws in learned_keywords.values() for kw in kws})

    found_keywords = [kw for kw in keywords if kw in resume_clean]
    count = len(found_keywords)
    possible = len(keywords)
    score = (count / possible) * weight if possible > 0 else 0

    missing = set(keywords) - set(found_keywords)
    refined_missing = missing - GENERIC_WORDS
    
    if refined_missing:
        suggestions.append("Consider including more targeted keywords such as: " + ", ".join(sorted(refined_missing)))
    else:
        suggestions.append("Your resume could benefit by further emphasizing role-specific technical keywords.")

    return score, suggestions

def calculate_skill_score(resume_text, weight=20):
    """
    Calculate the skill relevance score (20 points)
    """
    suggestions = []
    resume_clean = clean_text(resume_text)
    words = set(resume_clean.split())
    possible_skills = {word for word in words if len(word) > 3}
    found_skills = [skill for skill in possible_skills if skill in resume_clean]
    
    count = len(found_skills)
    possible = len(possible_skills)
    score = (count / possible) * weight if possible > 0 else 0

    if count < possible * 0.5:
        suggestions.append("Enhance your resume by emphasizing specific technical skills and relevant tools.")
    
    return score, suggestions

def calculate_section_score(resume_text, weight=20):
    """
    Check for essential sections in the resume and return a score with detailed suggestions.
    """
    suggestions = []
    resume_clean = clean_text(resume_text)
    expected_sections = ["work experience", "skills", "education", "projects", "summary"]
    sections_found = [section for section in expected_sections if section in resume_clean]
    
    count = len(sections_found)
    possible = len(expected_sections)
    score = (count / possible) * weight if possible > 0 else 0

    missing = set(expected_sections) - set(sections_found)
    if missing:
        detailed_sections = {
            "work experience": "provide detailed work experience and accomplishments",
            "skills": "list your specific skills and proficiencies",
            "education": "include your academic qualifications and certifications",
            "projects": "highlight notable projects and contributions",
            "summary": "offer a brief professional summary emphasizing your strengths"
        }
        suggestions_text = "; ".join(detailed_sections.get(section, section) for section in missing)
        suggestions.append("Improve your resume by ensuring these sections are complete: " + suggestions_text)
    
    return score, suggestions

def calculate_formatting_score(resume_text, weight=15):
    """
    Evaluate the resume formatting and readability.
    """
    suggestions = []
    bullet_count = resume_text.count("-") + resume_text.count("•")
    line_count = len(resume_text.splitlines())

    bullet_score = min(bullet_count / 5, 1)
    line_score = min(line_count / 10, 1)
    score = ((bullet_score + line_score) / 2) * weight

    if bullet_count < 5:
        suggestions.append("Improve readability by using bullet points to organize key information.")
    if line_count < 10:
        suggestions.append("Increase the number of line breaks to structure your content more clearly.")
    
    return score, suggestions

def calculate_word_count_score(resume_text, weight=15):
    """
    Award points for adequate resume length and the inclusion of quantifiable achievements.
    """
    suggestions = []
    words = resume_text.split()
    word_count = len(words)
    measurable_results = len(re.findall(r'\d+', resume_text))

    if word_count >= 300 and measurable_results >= 3:
        score = weight
    else:
        word_score = min(word_count / 300, 1)
        meas_score = min(measurable_results / 3, 1)
        score = ((word_score + meas_score) / 2) * weight

        if word_count < 300:
            suggestions.append("Expand your resume to at least 300 words to fully articulate your experience.")
        if measurable_results < 3:
            suggestions.append("Include specific achievements with numbers or percentages to illustrate your impact.")
    
    return score, suggestions

def calculate_ats_score(resume_text, learned_keywords, job_title=None):
    """
    Combine all individual scores to compute a final ATS score (maximum of 100)
    and aggregate all improvement suggestions.
    """
    total_score = 0
    all_suggestions = []

    kw_score, kw_sugg = calculate_keyword_score(resume_text, job_title, learned_keywords)
    total_score += kw_score
    all_suggestions.extend(kw_sugg)

    skill_score, skill_sugg = calculate_skill_score(resume_text)
    total_score += skill_score
    all_suggestions.extend(skill_sugg)

    section_score, section_sugg = calculate_section_score(resume_text)
    total_score += section_score
    all_suggestions.extend(section_sugg)

    format_score, format_sugg = calculate_formatting_score(resume_text)
    total_score += format_score
    all_suggestions.extend(format_sugg)

    wc_score, wc_sugg = calculate_word_count_score(resume_text)
    total_score += wc_score
    all_suggestions.extend(wc_sugg)

    final_score = min(round(total_score), 100)
    all_suggestions = list(dict.fromkeys(all_suggestions))  # remove duplicate suggestions
    return final_score, all_suggestions

def pdf_to_text(pdf_path):
    """
    Convert a PDF file to text using pdfminer.
    """
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error converting {pdf_path} to text: {e}")
        return ""

def process_dataset(csv_file, top_n_keywords=10):
    """
    learn job-specific keywords, and compute ATS scores for all resumes from csv dataset.
    """
    df = pd.read_csv(csv_file)
    if 'JOB TITLE' not in df.columns or 'RESUME TEXT' not in df.columns:
        raise ValueError("CSV must contain columns 'JOB TITLE' and 'RESUME TEXT'.")

    learned_keywords = learn_job_keywords(df, top_n=top_n_keywords)
    scores = []
    suggestions_list = []

    for idx, row in df.iterrows():
        job_title = row['JOB TITLE'] if pd.notnull(row['JOB TITLE']) else None
        resume_text = row['RESUME TEXT']
        score, suggestions = calculate_ats_score(resume_text, learned_keywords, job_title)
        scores.append(score)
        suggestions_list.append("; ".join(suggestions) if suggestions else "None")

    df['ATS SCORE'] = scores
    df['IMPROVEMENT SUGGESTIONS'] = suggestions_list
    return df, learned_keywords

#Main#

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATS Resume Scoring System")
    parser.add_argument("--csv", type=str, help="Path to the CSV dataset file.")
    parser.add_argument("--pdf", type=str, help="(Optional) Path to a PDF resume.")
    parser.add_argument("--job_title", type=str, default=None, help="(Optional) Job title for the PDF.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top keywords to extract.")
    args = parser.parse_args()

    if args.csv:
        if not os.path.exists(args.csv):
            print(f"CSV file {args.csv} does not exist.")
        else:
            results_df, learned_keywords = process_dataset(args.csv, top_n_keywords=args.top_n)
            output_csv = "ats_scored_results.csv"
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}.")
            print("\nLearned Job Keywords:")
            for job, keywords in learned_keywords.items():
                print(f" - {job}: {', '.join(keywords)}")

    if args.pdf:
        resume_text = pdf_to_text(args.pdf)
        if resume_text:
            score, suggestions = calculate_ats_score(resume_text, learned_keywords if args.csv else {}, job_title=args.job_title)
            print(f"\nPDF Resume ATS Score: {score}/100")
            if suggestions:
                print("Improvement Suggestions:")
                for s in suggestions:
                    print(f" - {s}")
            else:
                print("No suggestions. Resume looks good!")

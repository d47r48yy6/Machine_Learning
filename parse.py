import docx2txt
import fitz  # PyMuPDF
import spacy
import re

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

# Extract name using NLP
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

# Email extractor
def extract_email(text):
    match = re.search(r'\S+@\S+', text)
    return match.group() if match else "Not Found"

# Phone extractor
def extract_phone(text):
    match = re.search(r'(\+?\d[\d\s\-\(\)]{9,}\d)', text)
    return match.group() if match else "Not Found"

# Skills extractor
def extract_skills(text):
    skill_keywords = [
        'React', 'Node.js', 'Python', 'MongoDB', 'Java', 'C++', 'SQL',
        'HTML', 'CSS', 'JavaScript', 'Django', 'Flask', 'AWS', 'Git'
    ]
    found = [skill for skill in skill_keywords if skill.lower() in text.lower()]
    return list(set(found))  # remove duplicates

# Education extractor (basic pattern-based)
def extract_education(text):
    education_keywords = ['B.Tech', 'M.Tech', 'Bachelor', 'Master', 'B.E.', 'M.E.', 'BSc', 'MSc']
    for line in text.split('\n'):
        for keyword in education_keywords:
            if keyword.lower() in line.lower():
                return line.strip()
    return "Not Found"

# Experience extractor (basic pattern-based)
def extract_experience(text):
    match = re.search(r'(\d+)\s+years?', text, re.IGNORECASE)
    return match.group() if match else "Not Found"

# Project extractor (based on "Project"/"Projects" keyword)
def extract_projects(text):
    lines = text.splitlines()
    projects = []
    capture = False
    for line in lines:
        if re.search(r'\bprojects?\b', line, re.IGNORECASE):
            capture = True
        elif capture:
            if line.strip() == "":
                break
            projects.append(line.strip())
    return projects if projects else ["Not Found"]

# Master parse function
def parse_resume(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise Exception("Unsupported file format")

    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "projects": extract_projects(text)
    }

# Entry point
if __name__ == "__main__":
    path = input("Enter path to resume (PDF/DOCX): ").strip()
    data = parse_resume(path)

    print("\n--- Parsed Resume Data ---")
    from pprint import pprint
    pprint(data)
